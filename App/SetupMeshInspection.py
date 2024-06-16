from Common.ObjMtlMesh import ObjMtlMesh
from App.MeshInspection import InspectionApp
from Common.EulerMotion import EulerMotion
from Common.Player import Player
from Common.PositionCamera import PositionCamera
from Common.ReflectiveLight import ReflectiveLight
from Helpers.Globals import LIGHT_DEFAULT_GLOBAL_DICT, MATERIAL_DEFAULT_GLOBAL_DICT


def update_camera_position(app):
    app.player.camera.position = app.player.position
    app.player.camera.set_position_to_global()


def mesh_view(mesh_name: str, fragement_file: str, light_intensity: float, scale=2.):

    motion_model = EulerMotion([0, -1, -3], [0, 0, 0], object_id="motion")

    shape_factory = lambda: ObjMtlMesh(mesh_name, motion_model, scale, {}, **MATERIAL_DEFAULT_GLOBAL_DICT)

    camera = PositionCamera(fovy=45, aspect=640 / 480, near=.1, far=10, position=(0, 0, 0),
                            object_id="projection", position_glob_id="cameraPosition")
    player = Player(camera, object_id="view")

    light = ReflectiveLight([0, 2, -3], [1, 1, 1], light_intensity, 1., 1., 8., **LIGHT_DEFAULT_GLOBAL_DICT)

    app = InspectionApp(shape_factory, 'Shaders/model.vert', fragement_file, player, light,
                        limit_frame_rate=True, main_loop_command=update_camera_position)
    return app


if __name__ == '__main__':
    mesh_view('ObjFiles/Bulbasaur/Bulbasaur.obj')