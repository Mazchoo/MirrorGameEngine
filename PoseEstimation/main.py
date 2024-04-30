from time import perf_counter

import cv2
import numpy as np
import torch

from ComputerVision.CameraThread import VideoThread
from PoseEstimation.mobilenet import PoseEstimationWithMobileNet
from PoseEstimation.keypoints import extract_keypoints, group_keypoints
from PoseEstimation.load_state import load_state
from PoseEstimation.pose import Pose

MODEL_PATH = './PoseEstimation/models/checkpoint.pth'
INPUT_SIZE = 256
LOAD_CUDA = True
SMOOTH_POSES = False
TRACK_OBJECTS = False
STRIDE = 8
UPSAMPLE_RATIO = 4
IMAGE_MEAN = (128, 128, 128)
IMAGE_SCALE = 1/256


def infer_fast(net, img, net_input_height_size, upsample_ratio, 
               img_mean, img_mult, cuda):
    height, _, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    tensor_img = torch.from_numpy(scaled_img)
    if cuda:
        tensor_img = tensor_img.cuda()
    tensor_img = tensor_img.permute(2, 0, 1).unsqueeze(0).float()
    tensor_img = (tensor_img - img_mean) * img_mult

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale


def infer_on_image(net, img, height_size, cuda, img_mean, img_mult, stride, upsample_ratio):
    num_keypoints = Pose.num_kpts

    heatmaps, pafs, scale = infer_fast(net, img, height_size, upsample_ratio, 
                                       img_mean, img_mult, cuda)

    total_keypoints_num = 0
    all_keypoints_by_type = []
    for kpt_idx in range(num_keypoints):  # 19th for bg
        total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

    pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
    for kpt_id in range(all_keypoints.shape[0]):
        all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio) / scale
        all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio) / scale

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

    for pose in current_poses:
        pose.draw(img)

    img = cv2.addWeighted(img, 0.6, img, 0.4, 0)
    for pose in current_poses:
        cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                        (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))

    return img


class CheckPointMobileNet:
    def __init__(self, image: np.ndarray, model_path=MODEL_PATH, cuda=LOAD_CUDA):
        self.net = PoseEstimationWithMobileNet()
        self.load_cuda = cuda

        checkpoint = torch.load(model_path, map_location='cuda' if cuda else 'cpu')
        load_state(self.net, checkpoint)
        self.net = self.net.eval()

        self.image_mean = torch.FloatTensor(IMAGE_MEAN)
        self.image_mean = self.image_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        self.image_scale = torch.FloatTensor([IMAGE_SCALE])

        if LOAD_CUDA:
            self.net = self.net.cuda()
            self.image_mean = self.image_mean.cuda()
            self.image_scale = self.image_scale.cuda()

    def __call__(self, image: np.ndarray, input_height=INPUT_SIZE):
        return infer_on_image(self.net, image, input_height, self.load_cuda,
                              self.image_mean, self.image_scale,
                              STRIDE, UPSAMPLE_RATIO)


def main(Model):
    capture = VideoThread().start()
    model = Model(capture.frame)
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
