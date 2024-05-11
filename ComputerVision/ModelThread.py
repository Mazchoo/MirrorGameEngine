from threading import Thread
import cv2

from PoseEstimation.run import CheckPointMobileNet, PytorchModel, TORCH_PATH

class ModelThread:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        self.model = CheckPointMobileNet(PytorchModel(TORCH_PATH))

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
                

    def stop(self):
        self.stopped = True