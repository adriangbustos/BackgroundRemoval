import os
import cv2
import torch
import numpy as np
import imageio
import time
from PIL import Image
from tqdm import tqdm
from typing import Optional, Tuple, Union, Callable
import threading

from .matting_engine import MattingEngine
from .utils import calculate_downsample_ratio, parse_color_rgb


class VideoProcessor:
    """
    Handles streaming 4K video I/O, background compositing, and export encoding.
    Supports progress callbacks and cancellation flags for UI/Queue processing.
    """

    def __init__(self, engine: MattingEngine):
        self.engine = engine

    def process_video(
        self,
        input_path: str,
        output_path: str,
        bg_mode: str = "green",
        bg_color: Optional[str] = None,
        bg_image_path: Optional[str] = None,
        bg_video_path: Optional[str] = None,
        custom_downsample_ratio: Optional[float] = None,
        output_codec: str = "libx264",
        quality_crf: int = 17,
        progress_callback: Optional[Callable[[int, int, float, str], None]] = None,
        stop_event: Optional[threading.Event] = None,
        matte_threshold: float = 0.0,
    ):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")

        # Open video capture to retrieve metadata
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        ratio = calculate_downsample_ratio(width, height, custom_downsample_ratio)

        # Prepare background asset if needed
        bg_type, bg_asset = self._prepare_background(
            bg_mode=bg_mode,
            bg_color=bg_color,
            bg_image_path=bg_image_path,
            bg_video_path=bg_video_path,
            target_size=(width, height),
        )

        # Initialize video reader & writer
        reader = imageio.get_reader(input_path)
        is_transparent = (bg_mode == "transparent")

        writer_kwargs = {"fps": fps}
        if is_transparent:
            if output_path.lower().endswith(".mov"):
                writer_kwargs.update({
                    "codec": "prores_ks",
                    "pixelformat": "yuva444p10le",
                })
            else:
                if not output_path.lower().endswith(".webm"):
                    output_path = os.path.splitext(output_path)[0] + ".webm"
                writer_kwargs.update({
                    "codec": "libvpx-vp9",
                    "pixelformat": "yuva420p",
                })
        else:
            writer_kwargs.update({
                "codec": output_codec,
                "pixelformat": "yuv420p",
                "ffmpeg_params": ["-crf", str(quality_crf), "-preset", "medium"],
            })

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        writer = imageio.get_writer(output_path, **writer_kwargs)

        self.engine.reset_state()
        device = self.engine.device

        bg_video_cap = None
        if bg_type == "video":
            bg_video_cap = cv2.VideoCapture(bg_video_path)

        start_time = time.time()
        try:
            for frame_idx, frame_rgb in enumerate(reader):
                if stop_event and stop_event.is_set():
                    print("[Process] Cancelled by user.")
                    break

                # Convert HWC uint8 RGB frame to normalized 1CHW float tensor on device
                frame_tensor = (
                    torch.from_numpy(frame_rgb)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .float()
                    .div(255.0)
                    .to(device)
                )

                # Model inference
                fgr, pha = self.engine.process_frame(frame_tensor, downsample_ratio=ratio)

                # Extract foreground and alpha arrays to CPU numpy (H, W, C) & (H, W, 1)
                fgr_np = fgr.squeeze(0).permute(1, 2, 0).cpu().numpy()
                pha_np = pha.squeeze(0).permute(1, 2, 0).cpu().numpy()

                if matte_threshold > 0.0:
                    pha_np = np.clip((pha_np - matte_threshold) / (1.0 - matte_threshold), 0.0, 1.0)

                # Compose output frame
                if is_transparent:
                    rgba = np.dstack((fgr_np, pha_np))
                    out_frame = (np.clip(rgba, 0.0, 1.0) * 255.0).astype(np.uint8)
                else:
                    current_bg = self._get_bg_frame(bg_type, bg_asset, bg_video_cap, width, height)
                    blended = fgr_np * pha_np + current_bg * (1.0 - pha_np)
                    out_frame = (np.clip(blended, 0.0, 1.0) * 255.0).astype(np.uint8)

                writer.append_data(out_frame)

                # Calculate progress and ETA
                if progress_callback:
                    elapsed = time.time() - start_time
                    cur_fps = (frame_idx + 1) / max(elapsed, 0.001)
                    remaining_frames = max(total_frames - (frame_idx + 1), 0)
                    eta_secs = remaining_frames / max(cur_fps, 0.001)
                    eta_str = time.strftime("%M:%S", time.gmtime(eta_secs))
                    progress_callback(frame_idx + 1, total_frames, cur_fps, eta_str)

        finally:
            reader.close()
            writer.close()
            if bg_video_cap is not None:
                bg_video_cap.release()

    def _prepare_background(
        self,
        bg_mode: str,
        bg_color: Optional[str],
        bg_image_path: Optional[str],
        bg_video_path: Optional[str],
        target_size: Tuple[int, int],
    ) -> Tuple[str, Union[np.ndarray, str, None]]:
        width, height = target_size

        if bg_mode == "transparent":
            return ("transparent", None)

        if bg_mode == "image":
            if not bg_image_path or not os.path.exists(bg_image_path):
                raise FileNotFoundError(f"Background image not found: {bg_image_path}")
            img = Image.open(bg_image_path).convert("RGB")
            img = img.resize((width, height), Image.Resampling.LANCZOS)
            bg_arr = np.asarray(img, dtype=np.float32) / 255.0
            return ("image", bg_arr)

        if bg_mode == "video":
            if not bg_video_path or not os.path.exists(bg_video_path):
                raise FileNotFoundError(f"Background video not found: {bg_video_path}")
            return ("video", bg_video_path)

        # Solid color mode
        color_str = bg_color if bg_color else bg_mode
        r, g, b = parse_color_rgb(color_str)
        bg_arr = np.zeros((height, width, 3), dtype=np.float32)
        bg_arr[:, :] = [r / 255.0, g / 255.0, b / 255.0]
        return ("color", bg_arr)

    def _get_bg_frame(
        self,
        bg_type: str,
        bg_asset: Union[np.ndarray, str, None],
        bg_video_cap: Optional[cv2.VideoCapture],
        width: int,
        height: int,
    ) -> np.ndarray:
        if bg_type in ("color", "image"):
            return bg_asset

        if bg_type == "video" and bg_video_cap is not None:
            ret, frame = bg_video_cap.read()
            if not ret:
                bg_video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = bg_video_cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if frame_rgb.shape[1] != width or frame_rgb.shape[0] != height:
                    frame_rgb = cv2.resize(frame_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
                return frame_rgb.astype(np.float32) / 255.0

        fallback = np.zeros((height, width, 3), dtype=np.float32)
        fallback[:, :] = [0.0, 1.0, 0.0]
        return fallback
