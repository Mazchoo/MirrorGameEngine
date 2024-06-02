from threading import Thread
import cv2

from PoseEstimation.run import CheckPointMobileNet
from PoseEstimation.model_params import TRT_PATH
from PoseEstimation.tensor_rt import TensorRTModel

class ModelThread:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        self.model = CheckPointMobileNet(TensorRTModel(TRT_PATH), load_cuda=False)
        self.pose_dict = {}

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                self.grabbed, self.frame_raw = self.stream.read()
                self.frame = self.model(self.frame_raw)

                poses = self.model.poses
                if poses:
                    self.pose_dict = poses[0].get_pose_dict()
                else:
                    self.pose_dict = {}

    def stop(self):
        self.stopped = True