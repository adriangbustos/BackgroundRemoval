import torch
from typing import Tuple, Optional


def get_optimal_device() -> torch.device:
    """
    Returns optimal hardware-accelerated device.
    Supports DirectML (NVIDIA RTX / DirectX 12), CUDA, and CPU.
    """
    try:
        import torch_directml
        device = torch_directml.device()
        print(f"[Device] Hardware acceleration enabled via DirectML (DirectX 12 / RTX GPU)")
        return device
    except Exception:
        pass

    if torch.cuda.is_available():
        try:
            device = torch.device("cuda")
            # Quick sanity test on CUDA
            _ = torch.zeros(1, device=device)
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[Device] Using CUDA GPU: {gpu_name}")
            return device
        except Exception:
            pass

    print("[Device] Using CPU (Warning: 4K video processing is slower on CPU).")
    return torch.device("cpu")


def calculate_downsample_ratio(width: int, height: int, custom_ratio: Optional[float] = None) -> float:
    """
    Determines optimal downsample ratio for the RVM recurrent backbone.
    For 4K (3840x2160): 0.25 scales backbone to 960x540 (fast, low VRAM),
    while the built-in guided filter produces the native 3840x2160 output matte.
    """
    if custom_ratio is not None and 0.0 < custom_ratio <= 1.0:
        return custom_ratio

    max_dim = max(width, height)
    if max_dim >= 3840:  # 4K UHD or Cinema 4K
        return 0.25
    elif max_dim >= 2560:  # 1440p QHD
        return 0.35
    elif max_dim >= 1920:  # 1080p FHD
        return 0.5
    else:
        return 1.0


def parse_color_rgb(color_str: str) -> Tuple[int, int, int]:
    """
    Parses a color string such as 'green', 'blue', 'black', 'white',
    hex code '#00FF00', or comma-separated '0,255,0' into an (R, G, B) tuple.
    """
    color_str = color_str.strip().lower()
    presets = {
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "red": (255, 0, 0),
        "magenta": (255, 0, 255),
    }
    if color_str in presets:
        return presets[color_str]

    if color_str.startswith("#"):
        hex_val = color_str.lstrip("#")
        if len(hex_val) == 6:
            return tuple(int(hex_val[i : i + 2], 16) for i in (0, 2, 4))

    if "," in color_str:
        parts = [int(p.strip()) for p in color_str.split(",")]
        if len(parts) == 3:
            return (parts[0], parts[1], parts[2])

    raise ValueError(f"Unknown color specification: {color_str}")
