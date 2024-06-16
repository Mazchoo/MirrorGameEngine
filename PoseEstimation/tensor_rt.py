import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit # Does some initialisation

from PoseEstimation.run import main
from PoseEstimation.model_params import TRT_PATH


class TensorRTModel:
    def __init__(self, path: str, **kwargs):
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.trt_logger)
        with open(path, 'rb') as f:
            engine_data = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)

        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()

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
        self.context = None

    def create_context(self):
        self.context = self.engine.create_execution_context() 

    def __call__(self, inp: np.ndarray):
        cuda.Context.push(pycuda.autoinit.context)
        np.copyto(self.inputs[0]['host'], inp)
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.success = self.context.execute_v2(self.bindings)
        cuda.memcpy_dtoh_async(self.outputs[2]['host'], self.outputs[2]['device'], self.stream)
        cuda.memcpy_dtoh_async(self.outputs[3]['host'], self.outputs[3]['device'], self.stream)
        self.stream.synchronize()
        cuda.Context.pop()
        return self.outputs[2]['host'], self.outputs[3]['host']


if __name__ == '__main__':
    main(TensorRTModel(TRT_PATH), load_cuda=False)
