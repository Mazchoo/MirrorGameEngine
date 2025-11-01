import onnxruntime as ort
import torch
import numpy as np


class OnnxModel:
    def __init__(self, path: str, **kwargs):
        providers = [
            ("CUDAExecutionProvider", {"device_id": torch.cuda.current_device()})
        ]
        sess_options = ort.SessionOptions()
        self.session = ort.InferenceSession(
            path, sess_options=sess_options, providers=providers
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def __call__(self, inp: np.ndarray):
        outputs = self.session.run(self.output_names, {self.input_name: inp})
        return outputs
