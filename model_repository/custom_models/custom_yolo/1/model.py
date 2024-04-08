import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import to_dlpack


class TritonPythonModel:
    def initialize(self, args):
        """
        This function initializes pre-trained ResNet50 model,
        depending on the value specified by an `instance_group` parameter
        in `config.pbtxt`.

        Depending on what `instance_group` was specified in
        the config.pbtxt file (KIND_CPU or KIND_GPU), the model instance
        will be initialised on a cpu, a gpu, or both. If `instance_group` was
        not specified in the config file, then models will be loaded onto
        the default device of the framework.
        """
        # Here we set up the device onto which our model will beloaded,
        # based on specified `model_instance_kind` and `model_instance_device_id`
        # fields.
        device = "cuda" if args["model_instance_kind"] == "GPU" else "cpu"
        device_id = args["model_instance_device_id"]
        self.device = f"{device}:{device_id}"
        # This example is configured to work with torch=1.13
        # and torchvision=0.14. Thus, we need to provide a proper tag `0.14.1`
        # to make sure loaded Resnet50 is compatible with
        # installed `torchvision`.
        # Refer to README for installation instructions.
        self.model = (
            torch.hub.load('ultralytics/yolov5', 'yolov5s',trust_repo=True)
            .to(self.device)
            .eval()
        )

    def execute(self, requests):
        """
        This function receives a list of requests (`pb_utils.InferenceRequest`),
        performs inference on every request and appends it to responses.
        """
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
            with torch.no_grad():
                result = self.model(
                    torch.as_tensor(input_tensor.as_numpy(), device=self.device)
                )
            out_tensor = pb_utils.Tensor.from_dlpack("OUTPUT", to_dlpack(result))
            responses.append(pb_utils.InferenceResponse([out_tensor]))
        return responses



