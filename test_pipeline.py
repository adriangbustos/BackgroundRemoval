"""
Quick pipeline verification script with synthetic 4K video.
"""
import os
import cv2
import numpy as np
import time

def create_synthetic_4k_test_video(output_path: str = "test_input_4k.mp4", num_frames: int = 15):
    """Creates a lightweight 4K test video with a moving circle to simulate motion."""
    width, height = 3840, 2160
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"[Synthetic Test] Generating {num_frames} frames of 4K video (3840x2160)...")
    for i in range(num_frames):
        # Background: Gradient
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = int(i * 10) % 255
        frame[:, :, 1] = int(100 + i * 5) % 255
        frame[:, :, 2] = 180
        
        # Simulated person (moving circle/oval)
        center_x = int(width / 2 + np.sin(i / 3.0) * 300)
        center_y = int(height / 2 + np.cos(i / 3.0) * 150)
        cv2.circle(frame, (center_x, center_y), 400, (230, 180, 150), -1)  # Face/Head
        cv2.ellipse(frame, (center_x, center_y + 600), (500, 400), 0, 0, 360, (50, 50, 200), -1)  # Body
        
        out.write(frame)
        
    out.release()
    print(f"[Synthetic Test] Saved synthetic 4K video to: {output_path}")

if __name__ == "__main__":
    create_synthetic_4k_test_video()
