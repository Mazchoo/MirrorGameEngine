
import numpy as np

from Helpers.ReadMtl import parse_mtl, parse_vertex


def invert_negative_vertex(value):
    if value < 0:
        value *= -1
    if value > 1:
        value -= int(value)
    return value


def add_to_negative_vertex(value):
    if value < 0:
        value += 1
    if value > 1:
        value -= int(value)
    return value


def parse_texture_coord(line, invert_negative):
    vertex = [float(x) for x in line.split(' ')]

    if len(vertex) != 2:
        raise ValueError(f'Texture Coord {line} is wrong length.')

    if invert_negative:
        vertex[0] = invert_negative_vertex(vertex[0])
        vertex[1] = invert_negative_vertex(vertex[1])
    else:
        vertex[0] = add_to_negative_vertex(vertex[0])
        vertex[1] = add_to_negative_vertex(vertex[1])

    return vertex


def parse_face(line):
    face = [[int(x) for x in f.split('/')] for f in line.split(' ')]

    # ToDo - Support converting four faces into triangles
    if len(face) == 4:
        face = face[:3]

    if len(face) != 3:
        raise ValueError(f'Face {line} is not a triangle.')

    return face


def convert_parsed_data_to_numpy(faces, vertices, textures, normals):
    texture_data = {
        path: {'count': sum([len(face) for face in face_list]),
               'offset': 0} for path, face_list in faces.items()
    }

    total_vertices = sum([texture['count'] for texture in texture_data.values()])
    vertext_data = np.empty((total_vertices, 8), dtype=np.float32)

    ind = 0
    for path, face_list in faces.items():
        texture_data[path]['offset'] = ind

        for face in face_list:
            for vert_ind, tex_ind, normal_ind in face:
                vertext_data[ind, :3] = vertices[vert_ind - 1]
                vertext_data[ind, 3:5] = textures[tex_ind - 1]
                vertext_data[ind, 5:] = normals[normal_ind - 1]
                ind += 1

    return vertext_data, texture_data


def parse_obj(file_path, invert_negative=True):
    mtl_dict = parse_mtl(file_path)

    faces = {}
    current_texture = ''

    vertices = []
    textures = []
    normals = []
    with open(file_path, 'r') as f:
        while line := f.readline():
            # ToDo Inverting the texture file could be an option
            line = line.strip()
            flag = line[:line.find(' ')]
            line_content = line[len(flag) + 1:]

            if flag == 'usemtl':
                current_texture = mtl_dict[line_content]['texture']
                if current_texture not in faces:
                    faces[current_texture] = []
            elif flag == 'v':
                vertices.append(parse_vertex(line_content))
            elif flag == 'vt':
                textures.append(parse_texture_coord(line_content, invert_negative))
            elif flag == 'vn':
                normals.append(parse_vertex(line_content))
            elif flag == 'f':
                faces[current_texture].append(parse_face(line_content))

    vertext_data, texture_data = convert_parsed_data_to_numpy(
        faces, vertices, textures, normals
    )
    return vertext_data, texture_data, mtl_dict