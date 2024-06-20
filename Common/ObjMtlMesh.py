from typing import Optional

from OpenGL.GL import (glBindVertexArray, glDrawArrays, glDeleteVertexArrays,
                       glDeleteBuffers)
from OpenGL.GL import GL_TRIANGLES
import numpy as np

from Common.Player import Player
from Common.ShinyMaterial import ShinyMaterial
from Common.EulerMotion import EulerMotion
from Helpers.ReadObj import parse_obj
from Helpers.VertexDataOperations import (normalize_l1, get_bbox_2d,
                                          centroid_weighted_by_face, convex_volume)
from Helpers.MemoryUtil import generate_vertex_buffers, layout_position_texture_normal
from Helpers.Globals import IMAGE_SIZE


class ObjMtlMesh:
    image_size = np.array(IMAGE_SIZE, dtype=np.float32)
    __slots__ = 'centroid', 'bbox', 'vao', 'vbo', 'texture_data', 'materials', \
                'draw_iterator', 'motion', 'globals', 'volume', 'screen_bbox', 'screen_centroid'

    def __init__(self, file_path: str, motion: EulerMotion, 
                 normalize_scale: float, color_variation: Optional[dict] = None, **kwargs):

        if color_variation is None:
            color_variation = {}

        self.motion = motion
        vertices, self.texture_data, mtl_dict = parse_obj(file_path)
        
        self.globals = kwargs

        normalize_l1(vertices[:, :3], normalize_scale)
        self.centroid = centroid_weighted_by_face(vertices[:, :3])
        vertices[:, :3] -= self.centroid
        self.volume = convex_volume(vertices[:, :3])

        self.bbox = get_bbox_2d(vertices[:, :3])

        self.vao, self.vbo = generate_vertex_buffers(vertices)
        layout_position_texture_normal()

        self.materials = []
        self.draw_iterator = []
        for material_data in mtl_dict.values():
            hue_offset = color_variation.get(material_data['texture'])
            material = ShinyMaterial(hue_offset, **material_data)
            self.materials.append(material)

            texture = self.texture_data.get(material_data['texture'])
            if texture is None:
                offset = self.draw_iterator[-1][2] if self.draw_iterator else 0
                self.draw_iterator.append((material, 0, offset))
            else:
                self.draw_iterator.append((material, texture['count'], texture['offset']))

        self.materials = tuple(self.materials)
        self.draw_iterator = tuple(self.draw_iterator)

        self.screen_bbox = np.zeros((4, 2), dtype=np.int32) # Actual coordinates
        self.screen_centroid = np.zeros(2, dtype=np.int32)

    def draw(self):
        ''' Draw object with materials '''
        glBindVertexArray(self.vao)
        self.motion.set_motion_to_global()

        for material, count, offset in self.draw_iterator:
            material.use()
            glDrawArrays(GL_TRIANGLES, offset, count)

    def bind_global_variable_names(self, shader):
        ''' Prepare object for drawing '''
        for material in self.materials:
            material.assign_global_slots(shader, **self.globals)
        self.motion.bind_global_variable_names(shader)

    def transform_vertex_to_screen(self, vertex: np.ndarray, player: Player):
        ''' Turn vertex from static 4d sobject plane to 2d screen coordinates. '''
        vertex = self.motion.transform_vertex(vertex)
        vertex = player.transform_vertex(vertex)
        vertex /= vertex[3]
        vertex = vertex[:2] * -1
        vertex = vertex * 0.5 + 0.5
        vertex *= self.image_size
        return vertex

    def update_bbox_and_centroid(self, player: Player):
        ''' Updated bounding box and centroid screen coordinates '''
        v_min, v_max = self.bbox
        v_min = self.transform_vertex_to_screen(v_min, player)
        v_max = self.transform_vertex_to_screen(v_max, player)
        object_cog = np.array([0, 0, 0, 1], dtype=np.float32)
        centroid = self.transform_vertex_to_screen(object_cog, player)

        bbox = np.vstack([v_min, v_max])
        new_v_min = bbox.min(axis=0)
        new_v_max = bbox.max(axis=0)
        bbox = np.vstack([
            [new_v_min, [new_v_min[0], new_v_max[1]], new_v_max, [new_v_max[0], new_v_min[1]]]
        ])

        rot = self.motion.angles[1]
        rot_mat = np.array([
            [np.cos(rot), -np.sin(rot)],
            [np.sin(rot), np.cos(rot)]
        ], dtype=np.float32)
        bbox = (bbox - centroid) @ rot_mat + centroid

        self.screen_bbox = bbox.astype(np.int32)
        self.screen_centroid = centroid.astype(np.int32)

    def destroy(self):
        # Free memory of buffers
        for material in self.materials:
            material.destroy()

        glDeleteVertexArrays(1, (self.vao, ))
        glDeleteBuffers(1, (self.vbo, ))
