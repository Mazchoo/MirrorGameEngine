import numpy as np

MATERIAL_DEFAULT_GLOBAL_DICT = {
    "ambient_weighting_glob_id": "currentMaterial.ambientWeighting",
    "diffuse_weighting_glob_id": "currentMaterial.diffuseWeighting",
    "specular_weighting_glob_id": "currentMaterial.specularWeighting",
    "specular_exponent_glob_id": "currentMaterial.specularExponent",
    "opacicty_glob_id": "currentMaterial.opacity",
    "specular_tint_glob_id": "currentMaterial.specularTint",
}

LIGHT_DEFAULT_GLOBAL_DICT = {
    "position_glob_id": "lightSource.position",
    "color_glob_id": "lightSource.color",
    "strength_glob_id": "lightSource.strength",
    "ambient_strength_glob_id": "lightSource.ambientStrength",
    "min_dist_glob_id": "lightSource.minDistance",
    "max_dist_glob_id": "lightSource.maxDistance",
}

LIGHT_SIMPLE_DICT = {
    "position_glob_id": "lightSource.position",
    "color_glob_id": "lightSource.color",
    "strength_glob_id": "lightSource.strength",
}

IMAGE_VERTICES = (
    -1.0,
    -1.0,
    0.0,
    1.0,
    1.0,
    -1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    0.0,
    -1.0,
    -1.0,
    0.0,
    1.0,
    1.0,
    1.0,
    1.0,
    0.0,
    -1.0,
    1.0,
    0.0,
    0.0,
)

RELEASE_MODE = False
SCREEN_SIZE = (960, 720)
IMAGE_SIZE = (640, 480)

# Balloon constants
CEILING_LEVEL = -200
GRAVITY_CONSTANT = 0.0001
DESPAWN_LEVEL = 200
WALL_MOMENTUM_DROP = 0.75
CEILING_MOMENTUM_DROP = 0.5
FRAME_YZ_SPEED_DROP = 0.8
FRAME_XY_DISTURBANCE_RATIO = 0.01
FRAME_XZ_SPEED_DROP = 0.8

# Light cycle
LIGHT_FREQUENCY = 1 / (8 * np.pi)
LIGHT_AMPLITUDE = 0.1

# Light parameters
LIGHT_POSITION = (0.0, 2.0, -3.0)
LIGHT_COLOUR = (1.0, 1.0, 1.0)
LIGHT_STRENGTH = 0.8
LIGHT_AMBIENT_STRENGTH = 1.0
LIGHT_MIN_DISTANCE = 1.0
LIGHT_MAX_DISTANCE = 8.0

# Material parameters
SPECULAR_EXPONENT_WEIGHTING = 1 / 255.0
MIN_SPECULAR_EXPONENT = 0.2
DEFAULT_AMBIENT_WEIGHTING = (0.2, 0.2, 0.2)
DEFAULT_DIFFUSE_WEIGHTING = (1.5, 1.5, 1.5)
DEFAULT_SPECULAR_WEIGHTING = (1, 1, 1)
DEFAULT_SPECULAR_EXPONENT = 0.75
DEFAULT_OPACITY = 1.0
DEFAULT_SPECULAR_TINT = 0.0

# Balloon spawn parameters
NR_BALLOONS = 3
PITCH_RANGE = 0.1
ROLL_RANGE = np.pi * 2
START_X_RANGE = 2
START_Y = 2.25
START_Z = -4
MIN_START_DELAY = 120
MAX_START_DELAY = 240
POSSIBLE_HUE_OFFSETS = [12, 20, 0, 0, 0, 60, 25]
BALLOON_SCALE = 0.5
BALLOON_TERMINAL_VELOCITY = 0.075
BALLOON_DENSITY = 1.0

# Camera parameters
FIELD_OF_VIEW = 45
ASPECT_RATIO = 640 / 480
CAM_MIN_DISTANCE = 0.1
CAM_MAX_DISTANCE = 10
CAM_POSITION = (0, 0, 0)
