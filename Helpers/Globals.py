
from OpenGL.GL import glGetUniformLocation


def get_global_object_id(obj, attr_name: str, shader: int = None, var_name: str = None):
    if shader and var_name:
        global_object_id = glGetUniformLocation(shader, var_name)
    else:
        global_object_id = getattr(obj, attr_name)
        if global_object_id is None:
            raise AttributeError("Color Id has not been set.")

    return global_object_id


def bind_globals_to_object(obj, shader: int):
    for var_name, global_name in obj.globals.items():
        global_uniform = glGetUniformLocation(shader, global_name)
        setattr(obj, var_name, global_uniform)


MATERIAL_DEFAULT_GLOBAL_DICT = {
    'ambient_weighting_glob_id': 'currentMaterial.ambientWeighting',
    'diffuse_weighting_glob_id': 'currentMaterial.diffuseWeighting',
    'specular_weighting_glob_id': 'currentMaterial.specularWeighting',
    'specular_exponent_glob_id': 'currentMaterial.specularExponent',
    'opacicty_glob_id': 'currentMaterial.opacity',
    'specular_tint_glob_id': 'currentMaterial.specularTint',
}

LIGHT_DEFAULT_GLOBAL_DICT = {
    "position_glob_id": "lightSource.position",
    "color_glob_id": "lightSource.color",
    "strength_glob_id": "lightSource.strength",
    "ambient_strength_glob_id": "lightSource.ambientStrength",
    "min_dist_glob_id": "lightSource.minDistance",
    "max_dist_glob_id": "lightSource.maxDistance"
}

LIGHT_SIMPLE_DICT = {
    "position_glob_id": "lightSource.position",
    "color_glob_id": "lightSource.color",
    "strength_glob_id": "lightSource.strength"
}

IMAGE_VERTICES = (
    -1., -1., 0., 1.,
     1., -1., 1., 1.,
     1.,  1., 1., 0.,
    -1., -1., 0., 1.,
     1.,  1., 1., 0.,
    -1.,  1., 0., 0.
)

RELEASE_MODE = False
