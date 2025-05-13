import torch


def get_device(device_preference="gpu"):
    """
    Get the appropriate device based on availability and preference.

    Args:
        device_preference: 'gpu' to prefer GPU (CUDA or MPS), 'cpu' to force CPU

    Returns:
        torch.device: The selected device
    """
    if device_preference == "cpu":
        return torch.device("cpu")

    # Check for CUDA (NVIDIA GPUs)
    if torch.cuda.is_available():
        return torch.device("cuda")

    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    # Fall back to CPU
    print(
        "Warning: No GPU acceleration available. Using CPU which will be significantly slower."
    )
    return torch.device("cpu")
