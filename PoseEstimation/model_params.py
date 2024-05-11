
TORCH_PATH = './PoseEstimation/models/checkpoint.pth'
ONNX_PATH = './PoseEstimation/models/onyx.onnx'
TRT_PATH = './PoseEstimation/models/turtwig.trt'
INPUT_HEIGHT = 210
INPUT_WIDTH = 280
LOAD_CUDA = True
STRIDE = 8
UPSAMPLE_RATIO = 2 * 16 / 7
IMAGE_MEAN = (128, 128, 128)
IMAGE_SCALE = 1/256
