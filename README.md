# Speech Recognition App

A cross-platform system tray speech-to-text app using Whisper.

## Features

- 🎤 Double-tap to start/stop recording:
  - **macOS**: Command (⌘)
  - **Windows/Linux**: Control (Ctrl)
- 📝 Transcription auto-pastes at your cursor
- 🔄 Choose between different Whisper models
- ⚡ Optimized for each platform:
  - **Mac (Apple Silicon)**: MLX-accelerated inference
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

**For GPU acceleration** (optional): Install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads). The app will automatically detect CUDA if available, otherwise it falls back to CPU.

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

## Usage

1. Run the app - a 🎤 icon appears in your system tray
2. **Double-tap trigger key** (⌘ for Mac, Ctrl for Win) to start recording (icon turns red)
3. Speak
4. **Double-tap trigger key** again to stop - transcription pastes at cursor
5. Right-click the icon to change models or quit

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
