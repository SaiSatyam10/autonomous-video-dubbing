import os
import torch


def resolve_device(requested="auto", gpu_id=0):
    requested = (requested or os.environ.get("MUSETALK_DEVICE") or "auto").lower()

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device(f"cuda:{gpu_id}")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device(f"cuda:{gpu_id}")
        raise RuntimeError("CUDA was requested but is not available.")

    if requested == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS was requested but is not available.")

    if requested == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported device request: {requested}")


def get_inference_dtype(device, use_float16=False):
    if use_float16 and device.type == "cuda":
        return torch.float16
    return torch.float32


def device_str(device):
    return device if isinstance(device, str) else str(device)
