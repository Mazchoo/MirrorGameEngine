
from pathlib import Path


def parse_material(line, file_path):
    ''' Read material line and check it refers to an image e.g. Texture_2.png '''
    material_path = Path(file_path).parent / line

    if not material_path.exists():
        raise FileNotFoundError(f'Material {line} does not exist')
    if material_path.suffix != '.png':
        raise AttributeError('Only .png image type supported.')

    return str(material_path)


def parse_vertex(line):
    ''' Parse a vertex line e.g. -5.490000 20.340000 4.410002 '''
    vertex = [float(x) for x in line.split(' ')]

    if len(vertex) != 3:
        raise ValueError(f'Vertex {line} is wrong length.')

    return vertex


def get_texture(current_dict):
    ''' Texture will either be a previously read image or a colour from the diffuse weighting '''
    if texture := current_dict.get('texture'):
        return texture
    if texture := current_dict.get('diffuse_weighting'):
        return tuple(texture)


def parse_material_name(current_material, current_dict, mtl_dict):
    ''' Texture will be a coordinate or an image or an explicit color vt 0.491723 -0.123703 '''
    if current_material:
        current_dict['texture'] = get_texture(current_dict)

        if current_dict['texture'] is None:
            raise KeyError('End of material reached with no texture present.')

        mtl_dict[current_material] = current_dict


def check_mtl_file_exists(obj_path: str) -> str:
    ''' Get corresponding .mtl for current .obj file '''
    obj_path = Path(obj_path)
    mtl_path = obj_path.parent / (obj_path.stem + '.mtl')

    if not mtl_path.exists():
        raise FileNotFoundError(f"Mtl {mtl_path} cannot be found.")

    return str(mtl_path)


def parse_mtl(obj_path: str):
    ''' Read mtl file into a dictionary '''
    mtl_path = check_mtl_file_exists(obj_path)

    current_dict = {}
    current_material = ''

    mtl_dict = {}
    with open(mtl_path, 'r') as f:
        while line := f.readline():
            line = line.strip()
            flag = line[:line.find(' ')]
            line_content = line[len(flag) + 1:]

            if flag == 'newmtl':
                parse_material_name(current_material, current_dict, mtl_dict)

                current_material = line_content
                current_dict = {}
            elif flag == 'Ns':
                current_dict['specular_exponent'] = float(line_content)
            elif flag == 'Ka':
                current_dict['ambient_weighting'] = parse_vertex(line_content)
            elif flag == 'Kd':
                current_dict['diffuse_weighting'] = parse_vertex(line_content)
            elif flag == 'Ks':
                current_dict['specular_weighting'] = parse_vertex(line_content)
            elif flag == 'Ke':
                current_dict['emission_weighting'] = parse_vertex(line_content)
            elif flag == 'Ni':
                current_dict['refractive_index'] = float(line_content)
            elif flag == 'd':
                current_dict['opacicty'] = float(line_content)
            elif flag == 'illum':
                current_dict['illumination_model'] = int(line_content)
            elif flag == 'map_Kd':
                current_dict['texture'] = parse_material(line_content, mtl_path)
            elif flag == 'Ti':
                current_dict['specular_tint'] = float(line_content)

        parse_material_name(current_material, current_dict, mtl_dict)

    return mtl_dict
