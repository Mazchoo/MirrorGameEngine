
from OpenGL.GL import (glBindVertexArray, glDrawArrays, glDeleteVertexArrays,
                       glDeleteBuffers)
from OpenGL.GL import GL_TRIANGLES

from Common.ShinyMaterial import ShinyMaterial
from Common.EulerMotion import EulerMotion
from Helpers.ReadObj import parse_obj
from Helpers.VertexDataOperations import normalize_l1, get_bbox
from Helpers.MemoryUtil import generate_vertex_buffers, layout_position_texture_normal


class ObjMtlMesh:

    __slots__ = 'centroid', 'bbox', 'vao', 'vbo', 'texture_data', 'materials', \
                'draw_iterator', 'motion', 'globals'

    def __init__(self, file_path: str, motion: EulerMotion, normalize_scale=2, **kwargs):

        self.motion = motion
        vertices, self.texture_data, mtl_dict = parse_obj(file_path)
        self.globals = kwargs

        normalize_l1(vertices[:, :3], normalize_scale)
        self.centroid = vertices[:, :3].mean(axis=0)
        self.bbox = get_bbox(vertices)

        self.vao, self.vbo = generate_vertex_buffers(vertices)
        layout_position_texture_normal()

        self.materials = []
        self.draw_iterator = []
        for material_data in mtl_dict.values():
            material = ShinyMaterial(**material_data)
            self.materials.append(material)

            texture = self.texture_data.get(material_data['texture'])
            if texture is None:
                offset = self.draw_iterator[-1][2] if self.draw_iterator else 0
                self.draw_iterator.append((material, 0, offset))
            else:
                self.draw_iterator.append((material, texture['count'], texture['offset']))

        self.materials = tuple(self.materials)
        self.draw_iterator = tuple(self.draw_iterator)

    def draw(self):
        glBindVertexArray(self.vao)
        self.motion.set_motion_to_global()

        for material, count, offset in self.draw_iterator:
            material.use()
            glDrawArrays(GL_TRIANGLES, offset, count)

    def destroy(self):
        # Free memory of buffers
        for material in self.materials:
            material.destroy()

        glDeleteVertexArrays(1, (self.vao, ))
        glDeleteBuffers(1, (self.vbo, ))

    def bind_global_variable_names(self, shader):
        for material in self.materials:
            material.assign_global_slots(shader, **self.globals)
        self.motion.bind_global_variable_names(shader)
