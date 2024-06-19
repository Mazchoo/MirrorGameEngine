from OpenGL.GL import glGetUniformLocation


def get_global_object_id(obj, attr_name: str, shader: int = None, var_name: str = None):
    '''
        Get global id from shader when shader and uniform name are known.
        During the drawing loop, the object should already have a global is associated with
        its variable.
    '''
    if shader and var_name:
        global_object_id = glGetUniformLocation(shader, var_name)
    else:
        global_object_id = getattr(obj, attr_name)
        if global_object_id is None:
            raise AttributeError(f"Global Id {attr_name} has not been set.")

    return global_object_id


def bind_globals_to_object(obj, shader: int):
    ''' Bind all global variables in an object to the GPU. '''
    for var_name, global_name in obj.globals.items():
        global_uniform = glGetUniformLocation(shader, global_name)
        setattr(obj, var_name, global_uniform)
