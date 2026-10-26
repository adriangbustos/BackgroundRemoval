# BackgroundRemoval: 4K AI Video Matting & Background Removal

An AI-powered video matting and background removal tool optimized for **4K UHD (3840x2160)** footage using **RobustVideoMatting (RVM)** and **DirectML GPU acceleration** (compatible with modern NVIDIA RTX Blackwell/Ada/Ampere GPUs as well as AMD and Intel GPUs).

---

## Features

- **Native 4K UHD Processing**: Uses temporal recurrent states and guided filter matting to produce clean, flicker-free alpha mattes at full resolution without running out of VRAM.
- **DirectML Hardware Acceleration**: Runs accelerated on modern GPUs via DirectX 12, avoiding CUDA architecture mismatch errors.
- **Desktop GUI with Batch Queue**: Drag-and-drop or select entire folders of videos to process in queue with live progress bars, FPS counter, and ETA.
- **Fine-Tuning Controls**:
  - **AI Quality (Resolution Slider)**: Adjust internal inference downsample ratio (e.g. `0.25` for maximum speed, `0.50` or higher for crisp edge detail).
  - **Edge Cleanup (Matte Choke)**: Clean up semi-transparent background halos around hair and dark clothing.
- **Multiple Export Modes**:
  - `Transparent (.mov ProRes 4444)`: Lossless alpha channel for DaVinci Resolve, Adobe Premiere Pro, and Final Cut Pro.
  - `Green Screen (.mp4)`: Traditional chroma key background.
  - `Solid Color (.mp4)`: Custom color background with color picker.
  - `Custom Image (.mp4)`: Static background replacement.
  - `Custom Video (.mp4)`: Dynamic video background replacement.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/adriangbustos/BackgroundRemoval.git
   cd BackgroundRemoval
   ```

2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Desktop Graphical Interface (GUI)

Launch the desktop queue manager:

```bash
python app_gui.py
```
*Or double-click `start_ui.bat`.*

To compile as a standalone Windows executable (`.exe`), run:
```bash
build_exe.bat
```

### 2. Command-Line Interface (CLI)

Process videos individually via terminal:

```bash
# Export lossless transparent 4K ProRes 4444
python main.py -i "input.mp4" --bg transparent -o "output.mov"

# Export with custom AI quality and edge cleanup
python main.py -i "input.mp4" --bg transparent --downsample-ratio 0.5 --matte-threshold 0.10 -o "output.mov"

# Green Screen replacement
python main.py -i "input.mp4" --bg green -o "output.mp4"

# Custom image background
python main.py -i "input.mp4" --bg image --bg-image "office.jpg" -o "output.mp4"
```

---

## Architecture Overview

```
BackgroundRemoval/
├── app_gui.py             # Modern CustomTkinter desktop GUI with batch queue & fine-tuning
├── main.py                # Command-line interface
├── start_ui.bat           # Quick launcher for the desktop application
├── build_exe.bat          # PyInstaller build script for standalone Windows .exe
├── app_icon.ico           # Application icon
├── requirements.txt       # Dependencies (PyTorch, Torch-DirectML, OpenCV, ImageIO, CustomTkinter)
└── src/
    ├── matting_engine.py  # RobustVideoMatting wrapper with temporal recurrent states
    ├── video_processor.py # Streaming 4K frame reader, compositor, and ProRes/H.264 encoder
    └── utils.py           # Device detection (DirectML/CUDA/CPU) and downsample calculation
```

---

## License

MIT License.
