#!/usr/bin/env python3
"""
4K AI Video Background Remover - Queue Desktop GUI
Built with CustomTkinter for a modern dark-mode experience.
"""

import os
import sys
import io

class NullWriter:
    def write(self, text):
        pass
    def flush(self):
        pass
    def isatty(self):
        return False

if sys.stdout is None:
    sys.stdout = NullWriter()
if sys.stderr is None:
    sys.stderr = NullWriter()
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser
from typing import List, Dict, Any

import customtkinter as ctk
import cv2

from src.utils import get_optimal_device
from src.matting_engine import MattingEngine
from src.video_processor import VideoProcessor

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")


class QueueApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("4K AI Video Background Remover - Batch Queue")
        self.geometry("980x750")
        self.minsize(850, 600)

        # Set window icon
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_icon.ico")
        if os.path.exists(icon_path):
            try:
                self.iconbitmap(icon_path)
            except Exception:
                pass

        self.queue_items: List[Dict[str, Any]] = []
        self.is_processing = False
        self.stop_event = threading.Event()
        self.engine = None
        self.processor = None

        self._setup_ui()
        self._init_backend_async()

    def _init_backend_async(self):
        def _load():
            self.lbl_device_status.configure(text="Initializing AI Engine on GPU...", text_color="#E0A800")
            try:
                device = get_optimal_device()
                self.engine = MattingEngine(variant="resnet50", device=device)
                self.processor = VideoProcessor(engine=self.engine)
                self.lbl_device_status.configure(
                    text=f"AI Engine Ready (RTX GPU DirectML Accelerated)",
                    text_color="#28A745"
                )
            except Exception as e:
                self.lbl_device_status.configure(text=f"Initialization Error: {e}", text_color="#DC3545")

        threading.Thread(target=_load, daemon=True).start()

    def _setup_ui(self):
        # Top Header Frame
        self.header_frame = ctk.CTkFrame(self, corner_radius=10)
        self.header_frame.pack(fill="x", padx=16, pady=(16, 8))

        self.lbl_title = ctk.CTkLabel(
            self.header_frame,
            text="4K AI Video Background Remover",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        self.lbl_title.pack(side="left", padx=16, pady=10)

        self.lbl_device_status = ctk.CTkLabel(
            self.header_frame,
            text="Checking GPU...",
            font=ctk.CTkFont(size=12),
            text_color="#888888",
        )
        self.lbl_device_status.pack(side="right", padx=16, pady=10)

        # Settings & Source Controls Frame
        self.ctrl_frame = ctk.CTkFrame(self, corner_radius=10)
        self.ctrl_frame.pack(fill="x", padx=16, pady=8)

        # Row 1: Source & Output folders
        row1 = ctk.CTkFrame(self.ctrl_frame, fg_color="transparent")
        row1.pack(fill="x", padx=12, pady=(10, 5))

        self.btn_add_folder = ctk.CTkButton(
            row1, text="📁 Add Folder of Videos", command=self._on_add_folder, width=170
        )
        self.btn_add_folder.pack(side="left", padx=(0, 8))

        self.btn_add_files = ctk.CTkButton(
            row1, text="➕ Add Individual Videos", command=self._on_add_files, width=160, fg_color="#3A3A3A", hover_color="#4A4A4A"
        )
        self.btn_add_files.pack(side="left", padx=(0, 8))

        self.btn_clear_queue = ctk.CTkButton(
            row1, text="🗑️ Clear Queue", command=self._on_clear_queue, width=110, fg_color="#6C1E1E", hover_color="#8C2E2E"
        )
        self.btn_clear_queue.pack(side="left")

        # Output Folder selector
        row2 = ctk.CTkFrame(self.ctrl_frame, fg_color="transparent")
        row2.pack(fill="x", padx=12, pady=5)

        lbl_out = ctk.CTkLabel(row2, text="Output Directory:", font=ctk.CTkFont(size=13, weight="bold"))
        lbl_out.pack(side="left", padx=(0, 8))

        self.entry_output_dir = ctk.CTkEntry(row2, placeholder_text="Same folder as source video (or select custom folder)")
        self.entry_output_dir.pack(side="left", fill="x", expand=True, padx=(0, 8))

        self.btn_browse_output = ctk.CTkButton(row2, text="Browse...", width=90, command=self._on_browse_output)
        self.btn_browse_output.pack(side="left")

        # Row 3: Matting Options & Background Replacement Mode
        row3 = ctk.CTkFrame(self.ctrl_frame, fg_color="transparent")
        row3.pack(fill="x", padx=12, pady=5)

        lbl_mode = ctk.CTkLabel(row3, text="Background Mode:", font=ctk.CTkFont(size=13, weight="bold"))
        lbl_mode.pack(side="left", padx=(0, 8))

        self.combo_bg_mode = ctk.CTkComboBox(
            row3,
            values=[
                "Transparent (.mov ProRes 4444)",
                "Green Screen (.mp4)",
                "Solid Color (.mp4)",
                "Custom Image (.mp4)",
                "Custom Video (.mp4)",
            ],
            command=self._on_bg_mode_change,
            width=260,
        )
        self.combo_bg_mode.set("Transparent (.mov ProRes 4444)")
        self.combo_bg_mode.pack(side="left", padx=(0, 12))

        self.btn_pick_asset = ctk.CTkButton(
            row3,
            text="Pick Color...",
            width=130,
            command=self._on_pick_custom_asset,
        )
        self.custom_bg_value = "#00FF00"

        # Row 4: Fine-Tuning (Downsample Ratio & Matte Threshold)
        row4 = ctk.CTkFrame(self.ctrl_frame, fg_color="transparent")
        row4.pack(fill="x", padx=12, pady=(5, 10))

        lbl_tune = ctk.CTkLabel(row4, text="Fine-Tuning:", font=ctk.CTkFont(size=13, weight="bold"))
        lbl_tune.pack(side="left", padx=(0, 8))

        def _update_ratio_lbl(val):
            self.lbl_ratio_val.configure(text=f"{val:.2f}")

        def _update_thresh_lbl(val):
            self.lbl_thresh_val.configure(text=f"{val:.2f}")

        lbl_ratio = ctk.CTkLabel(row4, text="AI Quality:", font=ctk.CTkFont(size=12))
        lbl_ratio.pack(side="left", padx=(0, 4))
        
        self.slider_ratio = ctk.CTkSlider(row4, from_=0.15, to=1.0, width=120, command=_update_ratio_lbl)
        self.slider_ratio.set(0.25)
        self.slider_ratio.pack(side="left", padx=(0, 4))

        self.lbl_ratio_val = ctk.CTkLabel(row4, text="0.25", font=ctk.CTkFont(size=12, weight="bold"), width=30)
        self.lbl_ratio_val.pack(side="left", padx=(0, 12))

        lbl_thresh = ctk.CTkLabel(row4, text="Edge Cleanup:", font=ctk.CTkFont(size=12))
        lbl_thresh.pack(side="left", padx=(0, 4))

        self.slider_thresh = ctk.CTkSlider(row4, from_=0.0, to=0.5, width=120, command=_update_thresh_lbl)
        self.slider_thresh.set(0.0)
        self.slider_thresh.pack(side="left", padx=(0, 4))

        self.lbl_thresh_val = ctk.CTkLabel(row4, text="0.00", font=ctk.CTkFont(size=12, weight="bold"), width=30)
        self.lbl_thresh_val.pack(side="left", padx=(0, 12))

        # Queue List Container
        self.list_frame = ctk.CTkFrame(self, corner_radius=10)
        self.list_frame.pack(fill="both", expand=True, padx=16, pady=8)

        # Header for queue
        list_header = ctk.CTkFrame(self.list_frame, height=30, fg_color="#2B2B2B")
        list_header.pack(fill="x", padx=8, pady=(8, 4))

        ctk.CTkLabel(list_header, text="Video Name", font=ctk.CTkFont(size=12, weight="bold"), width=320, anchor="w").pack(side="left", padx=8)
        ctk.CTkLabel(list_header, text="Resolution", font=ctk.CTkFont(size=12, weight="bold"), width=120, anchor="w").pack(side="left", padx=4)
        ctk.CTkLabel(list_header, text="Frames / Duration", font=ctk.CTkFont(size=12, weight="bold"), width=140, anchor="w").pack(side="left", padx=4)
        ctk.CTkLabel(list_header, text="Status", font=ctk.CTkFont(size=12, weight="bold"), width=180, anchor="w").pack(side="left", padx=8)

        # Scrollable container for queue rows
        self.queue_scroll = ctk.CTkScrollableFrame(self.list_frame, fg_color="transparent")
        self.queue_scroll.pack(fill="both", expand=True, padx=4, pady=(0, 8))

        # Bottom Progress & Action Frame
        self.bottom_frame = ctk.CTkFrame(self, corner_radius=10)
        self.bottom_frame.pack(fill="x", padx=16, pady=(8, 16))

        # Progress bars
        p_row = ctk.CTkFrame(self.bottom_frame, fg_color="transparent")
        p_row.pack(fill="x", padx=16, pady=(12, 6))

        self.lbl_cur_progress = ctk.CTkLabel(p_row, text="Queue Idle", font=ctk.CTkFont(size=13))
        self.lbl_cur_progress.pack(side="left")

        self.lbl_cur_stats = ctk.CTkLabel(p_row, text="", font=ctk.CTkFont(size=12), text_color="#AAAAAA")
        self.lbl_cur_stats.pack(side="right")

        self.progress_bar = ctk.CTkProgressBar(self.bottom_frame)
        self.progress_bar.pack(fill="x", padx=16, pady=(0, 10))
        self.progress_bar.set(0)

        # Action Buttons Row
        act_row = ctk.CTkFrame(self.bottom_frame, fg_color="transparent")
        act_row.pack(fill="x", padx=16, pady=(0, 12))

        self.lbl_queue_count = ctk.CTkLabel(act_row, text="0 videos in queue", font=ctk.CTkFont(size=12), text_color="#888888")
        self.lbl_queue_count.pack(side="left")

        self.btn_stop = ctk.CTkButton(
            act_row,
            text="⏹️ Cancel",
            command=self._on_stop_queue,
            width=100,
            fg_color="#6C1E1E",
            hover_color="#8C2E2E",
            state="disabled",
        )
        self.btn_stop.pack(side="right", padx=(8, 0))

        self.btn_start = ctk.CTkButton(
            act_row,
            text="🚀 Start Processing Queue",
            command=self._on_start_queue,
            width=220,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#007ACC",
            hover_color="#005999",
        )
        self.btn_start.pack(side="right")

    def _on_bg_mode_change(self, mode_str: str):
        if "Solid Color" in mode_str:
            self.btn_pick_asset.pack(side="left", padx=(0, 8))
            self.btn_pick_asset.configure(text="Pick Color...", fg_color="#3A3A3A")
        elif "Custom Image" in mode_str:
            self.btn_pick_asset.pack(side="left", padx=(0, 8))
            self.btn_pick_asset.configure(text="Browse Image...", fg_color="#3A3A3A")
        elif "Custom Video" in mode_str:
            self.btn_pick_asset.pack(side="left", padx=(0, 8))
            self.btn_pick_asset.configure(text="Browse Video...", fg_color="#3A3A3A")
        else:
            self.btn_pick_asset.pack_forget()

    def _on_pick_custom_asset(self):
        mode = self.combo_bg_mode.get()
        if "Solid Color" in mode:
            color = colorchooser.askcolor(title="Choose Background Color")
            if color and color[1]:
                self.custom_bg_value = color[1]
                self.btn_pick_asset.configure(text=f"Color: {color[1]}")
        elif "Custom Image" in mode:
            path = filedialog.askopenfilename(
                title="Select Background Image",
                filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.webp;*.bmp")]
            )
            if path:
                self.custom_bg_value = path
                self.btn_pick_asset.configure(text=os.path.basename(path)[:15])
        elif "Custom Video" in mode:
            path = filedialog.askopenfilename(
                title="Select Background Video",
                filetypes=[("Video Files", "*.mp4;*.mov;*.mkv;*.avi")]
            )
            if path:
                self.custom_bg_value = path
                self.btn_pick_asset.configure(text=os.path.basename(path)[:15])

    def _on_add_folder(self):
        folder = filedialog.askdirectory(title="Select Folder Containing Videos")
        if not folder:
            return
        found_files = []
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(VIDEO_EXTENSIONS) and not file.lower().endswith(("_transparent.mov", "_removed_bg.mp4")):
                    found_files.append(os.path.join(root, file))
        if not found_files:
            messagebox.showinfo("No Videos Found", "No supported video files were found in that folder.")
            return
        self._add_files_to_queue(found_files)

    def _on_add_files(self):
        files = filedialog.askopenfilenames(
            title="Select Video Files",
            filetypes=[("Video Files", "*.mp4;*.mov;*.mkv;*.avi;*.webm;*.m4v")]
        )
        if files:
            self._add_files_to_queue(list(files))

    def _add_files_to_queue(self, paths: List[str]):
        existing_paths = {item["path"] for item in self.queue_items}
        for p in paths:
            if p in existing_paths:
                continue
            
            # Quick metadata read
            cap = cv2.VideoCapture(p)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            dur_str = time.strftime("%M:%S", time.gmtime(frames / max(fps, 1)))

            item = {
                "path": p,
                "name": os.path.basename(p),
                "width": w,
                "height": h,
                "fps": fps,
                "frames": frames,
                "duration_str": dur_str,
                "status": "Pending",
                "ui_row": None,
                "lbl_status": None,
            }
            self.queue_items.append(item)

        self._refresh_queue_ui()

    def _on_clear_queue(self):
        if self.is_processing:
            messagebox.showwarning("Queue Busy", "Please stop the queue before clearing.")
            return
        self.queue_items.clear()
        self._refresh_queue_ui()

    def _on_browse_output(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.entry_output_dir.delete(0, "end")
            self.entry_output_dir.insert(0, folder)

    def _refresh_queue_ui(self):
        for widget in self.queue_scroll.winfo_children():
            widget.destroy()

        for idx, item in enumerate(self.queue_items):
            row = ctk.CTkFrame(self.queue_scroll, fg_color="#1E1E1E" if idx % 2 == 0 else "#252525", corner_radius=6)
            row.pack(fill="x", pady=2)

            name_lbl = ctk.CTkLabel(row, text=item["name"], font=ctk.CTkFont(size=12), width=320, anchor="w")
            name_lbl.pack(side="left", padx=8, pady=6)

            res_text = f"{item['width']}x{item['height']}" + (" (4K)" if item['width'] >= 3840 else "")
            res_lbl = ctk.CTkLabel(row, text=res_text, font=ctk.CTkFont(size=12), width=120, anchor="w")
            res_lbl.pack(side="left", padx=4)

            dur_lbl = ctk.CTkLabel(row, text=f"{item['frames']}f ({item['duration_str']})", font=ctk.CTkFont(size=12), width=140, anchor="w")
            dur_lbl.pack(side="left", padx=4)

            lbl_status = ctk.CTkLabel(row, text=item["status"], font=ctk.CTkFont(size=12, weight="bold"), width=180, anchor="w", text_color="#AAAAAA")
            lbl_status.pack(side="left", padx=8)

            item["ui_row"] = row
            item["lbl_status"] = lbl_status

        count = len(self.queue_items)
        self.lbl_queue_count.configure(text=f"{count} video{'s' if count != 1 else ''} in queue")

    def _on_start_queue(self):
        if not self.queue_items:
            messagebox.showinfo("Queue Empty", "Please add some videos to the queue first.")
            return

        if not self.processor:
            messagebox.showwarning("Engine Loading", "AI Engine is still initializing on GPU. Please wait a moment.")
            return

        self.is_processing = True
        self.stop_event.clear()
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.btn_add_folder.configure(state="disabled")
        self.btn_add_files.configure(state="disabled")
        self.btn_clear_queue.configure(state="disabled")

        threading.Thread(target=self._run_queue_worker, daemon=True).start()

    def _on_stop_queue(self):
        if self.is_processing:
            self.stop_event.set()
            self.lbl_cur_progress.configure(text="Cancelling after current step...")

    def _run_queue_worker(self):
        mode_str = self.combo_bg_mode.get()
        custom_out_dir = self.entry_output_dir.get().strip()

        # Determine bg_mode parameter
        if "Transparent" in mode_str:
            bg_mode = "transparent"
        elif "Green" in mode_str:
            bg_mode = "green"
        elif "Solid Color" in mode_str:
            bg_mode = "color"
        elif "Custom Image" in mode_str:
            bg_mode = "image"
        elif "Custom Video" in mode_str:
            bg_mode = "video"
        else:
            bg_mode = "green"

        total_videos = len(self.queue_items)

        for idx, item in enumerate(self.queue_items):
            if self.stop_event.is_set():
                item["status"] = "Cancelled"
                self._update_item_status(item, "Cancelled", "#E0A800")
                continue

            if item["status"] == "Completed":
                continue

            input_path = item["path"]
            in_dir = os.path.dirname(input_path)
            base_name, _ = os.path.splitext(item["name"])

            out_dir = custom_out_dir if custom_out_dir else in_dir
            ext = ".mov" if bg_mode == "transparent" else ".mp4"
            output_path = os.path.join(out_dir, f"{base_name}_nobg{ext}")

            self._update_item_status(item, "Processing...", "#007ACC")
            self.lbl_cur_progress.configure(text=f"Processing [{idx + 1}/{total_videos}]: {item['name']}")

            def _progress_cb(cur_frame, tot_frames, fps, eta):
                pct = cur_frame / max(tot_frames, 1)
                self.progress_bar.set(pct)
                self.lbl_cur_stats.configure(
                    text=f"Frame {cur_frame}/{tot_frames} ({int(pct*100)}%)  |  {fps:.1f} FPS  |  ETA: {eta}"
                )
                self._update_item_status(item, f"Processing ({int(pct*100)}%)", "#007ACC")

            try:
                self.processor.process_video(
                    input_path=input_path,
                    output_path=output_path,
                    bg_mode=bg_mode,
                    bg_color=self.custom_bg_value if bg_mode == "color" else None,
                    bg_image_path=self.custom_bg_value if bg_mode == "image" else None,
                    bg_video_path=self.custom_bg_value if bg_mode == "video" else None,
                    progress_callback=_progress_cb,
                    stop_event=self.stop_event,
                    custom_downsample_ratio=self.slider_ratio.get(),
                    matte_threshold=self.slider_thresh.get(),
                )

                if self.stop_event.is_set():
                    item["status"] = "Cancelled"
                    self._update_item_status(item, "Cancelled", "#E0A800")
                else:
                    item["status"] = "Completed"
                    self._update_item_status(item, "Completed ✅", "#28A745")

            except Exception as e:
                item["status"] = "Failed"
                self._update_item_status(item, f"Failed: {str(e)[:25]}", "#DC3545")

        self.is_processing = False
        self.progress_bar.set(1.0 if not self.stop_event.is_set() else 0.0)
        self.lbl_cur_progress.configure(text="Queue Finished" if not self.stop_event.is_set() else "Queue Stopped")
        self.lbl_cur_stats.configure(text="")
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.btn_add_folder.configure(state="normal")
        self.btn_add_files.configure(state="normal")
        self.btn_clear_queue.configure(state="normal")

    def _update_item_status(self, item: Dict[str, Any], text: str, color: str):
        if item.get("lbl_status"):
            self.after(0, lambda: item["lbl_status"].configure(text=text, text_color=color))


if __name__ == "__main__":
    app = QueueApp()
    app.mainloop()
