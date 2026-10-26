import os
import torch
import torch.nn as nn
from typing import List, Optional, Tuple


class MattingEngine:
    """
    RobustVideoMatting (RVM) matting engine.
    Runs on DirectML / CUDA / CPU with recurrent temporal states.
    """

    def __init__(
        self,
        variant: str = "resnet50",
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cpu")
        self.variant = variant
        self.model = self._load_model(variant)
        self.rec_states: List[Optional[torch.Tensor]] = [None, None, None, None]

    def _load_model(self, variant: str) -> nn.Module:
        print(f"[Model] Loading RobustVideoMatting ({variant})...")
        model = torch.hub.load("PeterL1n/RobustVideoMatting", variant, trust_repo=True)
        model = model.to(self.device).eval()
        return model

    def reset_state(self):
        """Resets the recurrent states for a new video."""
        self.rec_states = [None, None, None, None]

    @torch.no_grad()
    def process_frame(
        self,
        frame_tensor: torch.Tensor,
        downsample_ratio: float = 0.25,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes a single RGB video frame tensor (1, 3, H, W) in [0, 1].
        Returns fgr (1, 3, H, W) and pha (1, 1, H, W).
        """
        fgr, pha, *self.rec_states = self.model(
            frame_tensor,
            *self.rec_states,
            downsample_ratio=downsample_ratio,
        )
        return fgr, pha
