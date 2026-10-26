#!/usr/bin/env python3
"""
4K Video Background Removal Tool (RVM + Guided Filtering)
CLI Interface
"""

import os
import argparse
import sys
import torch

from src.utils import get_optimal_device
from src.matting_engine import MattingEngine
from src.video_processor import VideoProcessor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="4K AI Video Background Removal & Matting Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Path to the input video (e.g. input_4k.mp4)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path for the output video. (Defaults to 'output_<name>.mp4' or '.mov' for transparent)",
    )
    parser.add_argument(
        "--bg",
        type=str,
        default="green",
        choices=["green", "blue", "black", "white", "transparent", "image", "video", "color"],
        help="Background replacement mode.",
    )
    parser.add_argument(
        "--bg-color",
        type=str,
        default=None,
        help="Custom color for --bg color (e.g. '#202020' or '32,32,32')",
    )
    parser.add_argument(
        "--bg-image",
        type=str,
        default=None,
        help="Path to background image (when --bg image)",
    )
    parser.add_argument(
        "--bg-video",
        type=str,
        default=None,
        help="Path to background video (when --bg video)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        choices=["resnet50", "mobilenetv3"],
        help="Model backbone variant: 'resnet50' (best 4K quality) or 'mobilenetv3' (faster).",
    )
    parser.add_argument(
        "--downsample-ratio",
        type=float,
        default=None,
        help="Override internal backbone downsample ratio (default: 0.25 for 4K).",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU execution instead of GPU hardware acceleration.",
    )
    parser.add_argument(
        "--matte-threshold",
        type=float,
        default=0.0,
        help="Choke/clean alpha matte edges. Increase (e.g. 0.1) if you see background halos.",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="libx264",
        help="Video codec for standard MP4 output ('libx264', 'libx265').",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=17,
        help="Constant Rate Factor for output video quality (lower is higher quality, 17 is visually lossless).",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        print(f"[Error] Input video file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Determine default output path if not specified
    if not args.output:
        base, _ = os.path.splitext(input_path)
        ext = ".mov" if args.bg == "transparent" else ".mp4"
        output_path = f"{base}_removed_bg{ext}"
    else:
        output_path = os.path.abspath(args.output)

    print("=" * 60)
    print(" 4K AI Video Background Removal Tool")
    print("=" * 60)
    print(f"Input Video : {input_path}")
    print(f"Output Video: {output_path}")
    print(f"Mode        : {args.bg}")
    print(f"Model       : {args.model}")
    print("-" * 60)

    # Determine device
    device = torch.device("cpu") if args.cpu else get_optimal_device()

    # Initialize model engine & video processor
    engine = MattingEngine(
        variant=args.model,
        device=device,
    )

    processor = VideoProcessor(engine=engine)
    processor.process_video(
        input_path=input_path,
        output_path=output_path,
        bg_mode=args.bg,
        bg_color=args.bg_color,
        bg_image_path=args.bg_image,
        bg_video_path=args.bg_video,
        custom_downsample_ratio=args.downsample_ratio,
        output_codec=args.codec,
        quality_crf=args.crf,
        matte_threshold=args.matte_threshold,
    )


if __name__ == "__main__":
    main()
