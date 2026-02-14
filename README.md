# Speech Recognition App

A cross-platform system tray speech-to-text app using Whisper.

## Features

- 🎤 Double-tap to start/stop recording:
  - **macOS**: Command (⌘)
  - **Windows/Linux**: Control (Ctrl)
- 📝 Transcription typed directly at your cursor (clipboard is not touched)
- ⏱️ Auto-stops after 30 seconds to prevent accidental recordings
- 🎙️ Select your preferred input device from the tray menu
- 🔄 Choose between different Whisper models
- ⚡ Optimized for each platform:
  - **Mac (Apple Silicon)**: MLX-accelerated inference via Metal GPU
  - **Windows/Linux**: GPU (CUDA) or CPU with faster-whisper

## Quick Start

```bash
./run.sh        # macOS/Linux
run.bat         # Windows
```

## Setup

### macOS (Apple Silicon)

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (fast Python package manager)
2. Run the app:
   ```bash
   ./run.sh
   ```
   The script auto-creates the virtual environment on first run.

Grant permissions: **System Settings → Privacy & Security → Accessibility** (add your terminal app)

### Windows

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Run the app:
   ```batch
   run.bat
   ```
   The script auto-creates the virtual environment on first run.

#### GPU Acceleration (NVIDIA)

GPU support is **automatically set up** when you install the dependencies via `run.bat`. The required CUDA libraries (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`) are included in `requirements.txt`.

**Prerequisites:**
- An NVIDIA GPU with Compute Capability ≥ 3.0 (most GPUs from 2012 onwards)
- **Latest NVIDIA GPU drivers** — download from [nvidia.com/drivers](https://www.nvidia.com/Download/index.aspx)

**That's it!** The app will auto-detect your GPU on startup. You should see output like:
```
🔍 Checking CUDA availability...
  ✅ ctranslate2 has CUDA support
🚀 Loading faster-whisper with CUDA: large-v3...
✅ faster-whisper loaded with CUDA (GPU accelerated)
```

#### Troubleshooting GPU / CUDA Issues

If the app falls back to CPU mode despite having an NVIDIA GPU:

1. **Update GPU drivers** — This is the #1 fix. Get the latest from [nvidia.com/drivers](https://www.nvidia.com/Download/index.aspx)

2. **Reinstall dependencies** — Delete the `.venv` folder and run `run.bat` again:
   ```batch
   rmdir /s /q .venv
   run.bat
   ```

3. **Manually install CUDA libraries** — If the automatic setup didn't work:
   ```batch
   .venv\Scripts\pip install --force-reinstall nvidia-cublas-cu12 nvidia-cudnn-cu12 ctranslate2
   ```

4. **Check the console output** — The app prints detailed diagnostics on startup. Look for `❌` messages that explain exactly what's missing.

5. **Verify CUDA is working** — You can test in the venv Python:
   ```batch
   .venv\Scripts\python -c "import ctranslate2; print(ctranslate2.get_supported_compute_types('cuda'))"
   ```
   If this prints compute types (e.g., `{'float16', 'float32', 'int8'}`), CUDA is working.

### Linux

1. Install PortAudio (required for pyaudio):
   ```bash
   sudo apt-get install portaudio19-dev  # Debian/Ubuntu
   sudo dnf install portaudio-devel      # Fedora
   ```
2. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
3. Run the app:
   ```bash
   ./run.sh
   ```

For GPU acceleration, ensure you have the latest NVIDIA drivers installed. The CUDA libraries are installed automatically via `requirements.txt`.

## Usage

1. Run the app — a 🎤 icon appears in your system tray
2. **Double-tap trigger key** (⌘ for Mac, Ctrl for Win) to start recording (icon turns red)
3. Speak
4. **Double-tap trigger key** again to stop — transcription is typed at your cursor
5. Recording auto-stops after 30 seconds if you forget
6. Right-click the icon to:
   - Change Whisper model
   - Select input device (or refresh the device list after plugging in a mic)
   - Toggle "Press Enter after paste"
   - Quit

## Models

| Model | Size | Quality | Notes |
|-------|------|---------|-------|
| `tiny` | ~75MB | Fast, basic | Good for quick notes |
| `base` | ~150MB | Better accuracy | Balanced |
| `small` | ~500MB | Good quality | Recommended |
| `medium` | ~1.5GB | High quality | Slower |
| `large-v3` | ~3GB | Best quality | Requires more RAM |

Change models via the tray icon menu.

## Platform Backends

| Platform | Backend | Acceleration |
|----------|---------|--------------|
| macOS (Apple Silicon) | mlx-whisper | Metal GPU |
| Windows | faster-whisper | CUDA GPU or CPU |
| Linux | faster-whisper | CUDA GPU or CPU |

## Files

- `main.py` - Main system tray application
- `whisper_backend.py` - Cross-platform Whisper abstraction
- `run.sh` - macOS/Linux launcher
- `run.bat` - Windows launcher
