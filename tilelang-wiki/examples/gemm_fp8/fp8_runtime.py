import torch


def has_fp8_runtime_support():
    if not torch.cuda.is_available():
        return False
    sm_major, sm_minor = torch.cuda.get_device_capability()
    return (sm_major, sm_minor) >= (8, 9)
