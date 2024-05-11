from time import perf_counter

import cv2
import numpy as np
import torch

from ComputerVision.CameraThread import VideoThread
from PoseEstimation.mobilenet import PoseEstimationWithMobileNet
from PoseEstimation.keypoints import extract_keypoints, group_keypoints
from PoseEstimation.load_state import load_state
from PoseEstimation.pose import Pose, track_poses

import onnxruntime as ort

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

print(f'Cuda enabled {torch.cuda.is_available()}')
print(f'Cudnn enabled {torch.backends.cudnn.enabled}')

TORCH_PATH = './PoseEstimation/models/checkpoint.pth'
ONNX_PATH = './PoseEstimation/models/onyx.onnx'
TRT_PATH = './PoseEstimation/models/turtwig.trt'
INPUT_HEIGHT = 210
INPUT_WIDTH = 280
LOAD_CUDA = False
STRIDE = 8
UPSAMPLE_RATIO = 4
IMAGE_MEAN = (128, 128, 128)
IMAGE_SCALE = 1/256


def infer_fast(net, img, net_target_y, net_target_x, upsample_ratio, 
               img_mean, img_mult, cuda):
    height, width, _ = img.shape
    scale_x = net_target_x / width
    scale_y = net_target_y / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

    tensor_img = torch.from_numpy(scaled_img)
    if cuda:
        tensor_img = tensor_img.cuda()
    tensor_img = tensor_img.permute(2, 0, 1).unsqueeze(0).float()
    tensor_img = (tensor_img - img_mean) * img_mult

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    if not isinstance(stage2_heatmaps, np.ndarray):
        stage2_heatmaps = stage2_heatmaps.cpu().data.numpy()
    heatmaps = np.transpose(stage2_heatmaps.squeeze(), (1, 2, 0))
    if heatmaps.dtype != np.float32:
        heatmaps = np.float32(heatmaps)

    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    if not isinstance(stage2_pafs, np.ndarray):
        stage2_pafs = stage2_pafs.cpu().data.numpy()
    pafs = np.transpose(stage2_pafs.squeeze(), (1, 2, 0))
    if pafs.dtype != np.float32:
        pafs = np.float32(pafs)
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale_x, scale_y


def infer_on_image(net, img, height, width, cuda, img_mean, img_mult,
                   stride, upsample_ratio, previous_poses):
    num_keypoints = Pose.num_kpts

    heatmaps, pafs, scale_x, scale_y = infer_fast(net, img, height, width, upsample_ratio, 
                                       img_mean, img_mult, cuda)

    total_keypoints_num = 0
    all_keypoints_by_type = []
    for kpt_idx in range(num_keypoints):  # 19th for bg
        total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

    pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
    for kpt_id in range(all_keypoints.shape[0]):
        all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio) / scale_x
        all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio) / scale_y

    current_poses = []
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
        for kpt_id in range(num_keypoints):
            if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
        pose = Pose(pose_keypoints, pose_entries[n][18])
        current_poses.append(pose)

    track_poses(previous_poses, current_poses, smooth=True)

    for pose in current_poses:
        pose.draw(img)

    img = cv2.addWeighted(img, 0.6, img, 0.4, 0)
    for pose in current_poses:
        cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                        (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))

        cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))

    return img, current_poses


class PytorchModel:
    def __init__(self, path: str, cuda=True, **kwargs):
        self.net = PoseEstimationWithMobileNet()
        self.load_cuda = cuda
        self.half = False

        checkpoint = torch.load(path, map_location='cuda' if self.load_cuda else 'cpu')
        load_state(self.net, checkpoint)
        self.net = self.net.eval()
    
        if self.load_cuda:
            self.net = self.net.cuda()
            if kwargs.get("half"):
                self.net = self.net.half()
                self.half = True

    def __call__(self, inp: torch.Tensor):
        inp = inp.half() if self.half else inp
        return self.net(inp)


class OnnxModel:
    def __init__(self, path: str, **kwargs):
        providers = [("CUDAExecutionProvider",
                      {"device_id": torch.cuda.current_device()})]
        sess_options = ort.SessionOptions()
        self.session = ort.InferenceSession(path, sess_options=sess_options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
    
    def __call__(self, inp: torch.Tensor):
        np_input = inp.cpu().numpy()
        outputs = self.session.run(
            self.output_names,
            {self.input_name: np_input}              
        )
        return outputs


class TensorRTModel:
    def __init__(self, path: str, **kwargs):
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.trt_logger)
        with open(path, 'rb') as f:
            engine_data = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)

        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()

        self.context = self.engine.create_execution_context()
        self.context.set_optimization_profile_async(0, self.stream.handle)

        for binding in self.engine:
            shape = tuple(self.engine.get_tensor_shape(binding))
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding))

            host_mem = cuda.pagelocked_empty(shape, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(cuda_mem))

            if binding == 'data':
                self.inputs.append({'host': host_mem, 'device': cuda_mem,
                                    'shape': shape, 'name': binding})
            else:
                self.outputs.append({'host': host_mem, 'device': cuda_mem,
                                     'shape': shape, 'name': binding})

        self.success = None # Don't know yet if model run was successful

    def __call__(self, inp: torch.Tensor):
        np_input = inp.cpu().numpy()
        np.copyto(self.inputs[0]['host'], np_input)
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.success = self.context.execute_v2(self.bindings)
        cuda.memcpy_dtoh_async(self.outputs[2]['host'], self.outputs[2]['device'], self.stream)
        cuda.memcpy_dtoh_async(self.outputs[3]['host'], self.outputs[3]['device'], self.stream)
        self.stream.synchronize()
        return self.outputs[2]['host'], self.outputs[3]['host']

class CheckPointMobileNet:
    def __init__(self, net, cuda=LOAD_CUDA):
        self.net = net
        self.load_cuda = cuda

        self.image_mean = torch.FloatTensor(IMAGE_MEAN)
        self.image_mean = self.image_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        self.image_scale = torch.FloatTensor([IMAGE_SCALE])
        self.poses = []

        if self.load_cuda:
            self.image_mean = self.image_mean.cuda()
            self.image_scale = self.image_scale.cuda()

    def __call__(self, image: np.ndarray, height=INPUT_HEIGHT, width=INPUT_WIDTH):
        out_image, self.poses = infer_on_image(self.net, image, height, width, self.load_cuda,
                                               self.image_mean, self.image_scale,
                                               STRIDE, UPSAMPLE_RATIO, self.poses)
        return out_image


def main(network):
    capture = VideoThread().start()
    model = CheckPointMobileNet(network)
    cv2.namedWindow("Camera")

    total_time = 0
    total_measurements = 0
    while True:
        start = perf_counter()
        pred_frame = model(capture.frame)
        total_time += perf_counter() - start
        total_measurements += 1

        if total_measurements != 0:
            avg_time_ms = np.round(1000 * total_time / total_measurements, 3)
            avg_fps = 1000 / avg_time_ms
            print(f'Avg time {total_time/total_measurements} fps {avg_fps}')

        cv2.imshow("Camera", pred_frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:
            break

    capture.stop()

if __name__ == '__main__':
    main(TensorRTModel(TRT_PATH))
