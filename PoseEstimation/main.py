import math

import cv2
import numpy as np
import torch
from torchvision.transforms import Pad

from ComputerVision.CameraThread import VideoThread
from PoseEstimation.mobilenet import PoseEstimationWithMobileNet
from PoseEstimation.keypoints import extract_keypoints, group_keypoints
from PoseEstimation.load_state import load_state
from PoseEstimation.pose import Pose, track_poses
from time import perf_counter

MODEL_PATH = './PoseEstimation/models/checkpoint.pth'
INPUT_SIZE = 256
LOAD_CUDA = True
SMOOTH_POSES = False
TRACK_OBJECTS = False


def normalize(img, img_mean, img_scale):
    img = np.array(img, dtype=np.float32)
    img = (img - img_mean) * img_scale
    return img


def pad_width(img, stride, min_dims):
    _, _, h, w = img.shape
    h = min(min_dims[0], h)
    min_dims[0] = math.ceil(min_dims[0] / float(stride)) * stride
    min_dims[1] = max(min_dims[1], w)
    min_dims[1] = math.ceil(min_dims[1] / float(stride)) * stride
    pad = []
    pad.append(int(math.floor((min_dims[0] - h) / 2.0)))
    pad.append(int(math.floor((min_dims[1] - w) / 2.0)))
    pad.append(int(min_dims[0] - h - pad[0]))
    pad.append(int(min_dims[1] - w - pad[1]))
    padded_img = Pad(pad)(img)
    return padded_img, pad


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cuda,
               img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
    height, _, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]

    tensor_img = torch.from_numpy(scaled_img)
    if cuda:
        tensor_img = tensor_img.cuda()
    tensor_img = tensor_img.permute(2, 0, 1).unsqueeze(0).float()
    padded_img, pad = pad_width(tensor_img, stride, min_dims)
    print(pad)

    stages_output = net(padded_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad


def infer_on_image(net, img, height_size, cuda, track, smooth):
    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []

    heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cuda)

    total_keypoints_num = 0
    all_keypoints_by_type = []
    for kpt_idx in range(num_keypoints):  # 19th for bg
        total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

    pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
    for kpt_id in range(all_keypoints.shape[0]):
        all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
        all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale

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

    if track:
        track_poses(previous_poses, current_poses, smooth=smooth)
        previous_poses = current_poses

    for pose in current_poses:
        pose.draw(img)

    img = cv2.addWeighted(img, 0.6, img, 0.4, 0)
    for pose in current_poses:
        cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                        (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
        if track:
            cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))

    return img


class CheckPointMobileNet:
    def __init__(self, model_path=MODEL_PATH, cuda=LOAD_CUDA):
        self.net = PoseEstimationWithMobileNet()
        self.load_cuda = cuda
        checkpoint = torch.load(model_path, map_location='cuda' if cuda else 'cpu')
        load_state(self.net, checkpoint)
        self.net = self.net.eval()

        if LOAD_CUDA:
            self.net = self.net.cuda()

    def __call__(self, image: np.ndarray, height=INPUT_SIZE, tarack_objects=TRACK_OBJECTS, smooth_poses=SMOOTH_POSES):
        return infer_on_image(self.net, image, height, self.load_cuda, tarack_objects, smooth_poses)


def main(Model):
    model = Model()
    capture = VideoThread().start()
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
    main(CheckPointMobileNet)
