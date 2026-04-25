"""
TensorRT Engine wrapper (torch-based, pycuda-free)

TensorRT 10.x API + torch.cuda 기반. JetPack 6.x / Orin 전용.
"""

import os
import torch
import tensorrt as trt


class TRTEngine:
    def __init__(self, engine_path: str):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.input_names: list[str] = []
        self.output_names: list[str] = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        print(f"[TRT] loaded: {os.path.basename(engine_path)}")
        print(f"      inputs:  {self.input_names}")
        print(f"      outputs: {self.output_names}")

    def infer(self, inputs: dict | torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            inputs: torch.Tensor (CUDA) or dict {name: torch.Tensor}

        Returns:
            dict {name: torch.Tensor (CPU)}
        """
        if isinstance(inputs, torch.Tensor):
            inputs = {self.input_names[0]: inputs}

        input_tensors = {}
        for name, tensor in inputs.items():
            t = tensor.contiguous().cuda().float()
            self.context.set_input_shape(name, list(t.shape))
            self.context.set_tensor_address(name, t.data_ptr())
            input_tensors[name] = t

        output_tensors = {}
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            t = torch.zeros(shape, dtype=torch.float32, device="cuda")
            self.context.set_tensor_address(name, t.data_ptr())
            output_tensors[name] = t

        stream = torch.cuda.current_stream()
        self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        torch.cuda.synchronize()

        return {name: t.cpu() for name, t in output_tensors.items()}
