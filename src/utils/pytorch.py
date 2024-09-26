from typing import Optional
from packaging import version
import torch


def flatten_results(results):
    out = []
    for batch in results:
        k = list(batch.keys())[0]
        for i in range(len(batch[k])):
            out.append({k: v[i] for k, v in batch.items()})
    return out


def select_device(device: Optional[str] = None) -> Optional[str]:

    if device is None:
        # Check if CUDA is available
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    if isinstance(device, str):
        if device == "cuda":
            # If the user specifies only CUDA, use the GPU with the most free memory
            free_memory = [
                torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
                for i in range(torch.cuda.device_count())
            ]
            device = "cuda:" + str(torch.argmax(torch.tensor(free_memory)).item())
        elif device.startswith("cuda:"):
            # If the user specifies a specific GPU, make sure it exists
            if int(device[5:]) >= torch.cuda.device_count():
                raise ValueError(f"Invalid device: {device}")

        return device




def torch_int_div(tensor1, tensor2):
    """
    From transformers v4.27.4 (transformers.pytorch_utils)
    A function that performs integer division across different versions of PyTorch.
    """
    parsed_torch_version_base = version.parse(version.parse(torch.__version__).base_version)
    is_torch_less_than_1_8 = parsed_torch_version_base < version.parse("1.8.0")
    if is_torch_less_than_1_8:
        return tensor1 // tensor2
    else:
        return torch.div(tensor1, tensor2, rounding_mode="floor")
