"""
Whisper Backend Abstraction Layer
==================================
Provides a unified interface for speech-to-text transcription that
automatically selects the optimal backend based on the platform:

- macOS (Apple Silicon): Uses mlx-whisper for Metal-accelerated inference
- Windows/Linux: Uses faster-whisper with CUDA GPU or CPU fallback
"""

import platform
import os
import sys
from typing import Optional, Dict, Any

# Detect platform once at module load
IS_MACOS = platform.system() == "Darwin"
IS_APPLE_SILICON = IS_MACOS and platform.machine() == "arm64"


# Model mappings for each backend
MLX_MODELS = {
    "tiny": "mlx-community/whisper-tiny",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3-4bit": "mlx-community/whisper-large-v3-mlx-4bit",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
}

FASTER_WHISPER_MODELS = {
    "tiny": "tiny",
    "base": "base",
    "small": "small",
    "medium": "medium",
    # No 4-bit quantized version for faster-whisper, use int8
    "large-v3-4bit": "large-v3",
    "large-v3": "large-v3",
}

# Default model to use
DEFAULT_MODEL = "large-v3"


class WhisperBackend:
    """
    Unified Whisper transcription backend that auto-selects the optimal
    implementation based on the current platform.
    """
    
    def __init__(self, model_key: str = DEFAULT_MODEL):
        """
        Initialize the Whisper backend.
        
        Args:
            model_key: Key for the model to use (e.g., "tiny", "large-v3")
        """
        self.model_key = model_key
        self._backend_name = "mlx" if IS_APPLE_SILICON else "faster-whisper"
        self._model = None
        
        print(f"🖥️  Platform: {platform.system()} ({platform.machine()})")
        print(f"⚙️  Backend: {self._backend_name}")
        
        self._init_backend()
    
    def _init_backend(self):
        """Initialize the appropriate backend based on platform."""
        if IS_APPLE_SILICON:
            self._init_mlx()
        else:
            self._init_faster_whisper()
    
    def _init_mlx(self):
        """Initialize MLX-Whisper backend for Apple Silicon."""
        try:
            import mlx_whisper
            self._mlx_whisper = mlx_whisper
            print(f"✅ MLX-Whisper loaded")
        except ImportError as e:
            print(f"❌ Failed to import mlx-whisper: {e}")
            print("   Install with: pip install mlx-whisper")
            sys.exit(1)
    
    def _check_cuda_available(self):
        """Check if CUDA is available and print diagnostic information."""
        print("🔍 Checking CUDA availability...")
        
        # Check 1: Is ctranslate2 built with CUDA support?
        try:
            import ctranslate2
            cuda_supported = "cuda" in ctranslate2.get_supported_compute_types("cuda")
            if cuda_supported:
                print("  ✅ ctranslate2 has CUDA support")
            else:
                print("  ❌ ctranslate2 was installed WITHOUT CUDA support")
                print("     Fix: pip install --force-reinstall ctranslate2")
                return False
        except Exception as e:
            print(f"  ❌ ctranslate2 CUDA check failed: {e}")
            print("     This usually means CUDA libraries are not installed.")
            print("     Fix: pip install nvidia-cublas-cu12 nvidia-cudnn-cu12")
            return False
        
        # Check 2: Can we detect the GPU via CUDA runtime?
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                cuda_version = torch.version.cuda
                print(f"  ✅ GPU detected: {gpu_name}")
                print(f"  ✅ CUDA version: {cuda_version}")
            else:
                print("  ⚠️  PyTorch reports CUDA not available (torch is optional, continuing...)")
        except ImportError:
            print("  ℹ️  PyTorch not installed (optional — skipping GPU name detection)")
        except Exception as e:
            print(f"  ⚠️  GPU detection via PyTorch failed: {e}")
        
        return True
    
    def _init_faster_whisper(self):
        """Initialize faster-whisper backend for Windows/Linux."""
        try:
            from faster_whisper import WhisperModel
            
            model_name = FASTER_WHISPER_MODELS.get(self.model_key, "base")
            
            # Check CUDA availability with diagnostics
            cuda_available = self._check_cuda_available()
            
            if cuda_available:
                try:
                    print(f"🚀 Loading faster-whisper with CUDA: {model_name}...")
                    self._model = WhisperModel(model_name, device="cuda", compute_type="float16")
                    print(f"✅ faster-whisper loaded with CUDA (GPU accelerated)")
                    return
                except Exception as e:
                    print(f"⚠️  CUDA model loading failed: {e}")
                    print("")
                    print("   Common fixes:")
                    print("   1. Install CUDA libraries: pip install nvidia-cublas-cu12 nvidia-cudnn-cu12")
                    print("   2. Update GPU drivers: https://www.nvidia.com/Download/index.aspx")
                    print("   3. Reinstall ctranslate2: pip install --force-reinstall ctranslate2")
                    print("")
                    print("   Falling back to CPU...")
            else:
                print("")
                print("   ⚠️  To enable GPU acceleration, install CUDA dependencies:")
                print("   pip install nvidia-cublas-cu12 nvidia-cudnn-cu12")
                print("   Then restart the app.")
                print("")
            
            # Fallback to CPU
            print(f"💻 Loading model on CPU: {model_name}...")
            self._model = WhisperModel(model_name, device="cpu", compute_type="int8")
            print(f"✅ faster-whisper loaded (CPU mode — will be slower)")
            
        except ImportError as e:
            print(f"❌ Failed to import faster-whisper: {e}")
            print("   Install with: pip install faster-whisper")
            sys.exit(1)
    
    @property
    def backend_name(self) -> str:
        """Return the name of the active backend."""
        return self._backend_name
    
    @property
    def available_models(self) -> Dict[str, str]:
        """Return available models for the current backend."""
        if IS_APPLE_SILICON:
            return MLX_MODELS
        return FASTER_WHISPER_MODELS
    
    def get_model_path(self) -> str:
        """Get the current model path/name for the active backend."""
        if IS_APPLE_SILICON:
            return MLX_MODELS.get(self.model_key, MLX_MODELS[DEFAULT_MODEL])
        return FASTER_WHISPER_MODELS.get(self.model_key, FASTER_WHISPER_MODELS[DEFAULT_MODEL])
    
    def set_model(self, model_key: str):
        """
        Switch to a different model.
        
        Args:
            model_key: Key for the model to use
        """
        if model_key not in self.available_models:
            print(f"⚠️  Unknown model: {model_key}")
            return
        
        self.model_key = model_key
        print(f"🔄 Switching to model: {model_key}")
        
        # Reinitialize for faster-whisper (needs to reload model)
        if not IS_APPLE_SILICON:
            self._init_faster_whisper()
    
    def preload(self):
        """
        Preload the model to avoid delay on first transcription.
        For MLX, this does a dummy transcription. For faster-whisper,
        the model is already loaded during init.
        """
        if IS_APPLE_SILICON:
            import tempfile
            import wave
            import struct
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
            
            try:
                # Create a minimal valid WAV file
                with wave.open(temp_path, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(struct.pack('<' + 'h' * 1600, *([0] * 1600)))
                
                # Run dummy transcription to load model
                self._mlx_whisper.transcribe(temp_path, path_or_hf_repo=self.get_model_path())
            finally:
                try:
                    os.unlink(temp_path)
                except:
                    pass
        # For faster-whisper, model is already loaded in _init_faster_whisper
    
    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe an audio file to text.
        
        Args:
            audio_path: Path to the audio file (WAV format, 16kHz recommended)
            
        Returns:
            Transcribed text string
        """
        if IS_APPLE_SILICON:
            return self._transcribe_mlx(audio_path)
        else:
            return self._transcribe_faster_whisper(audio_path)
    
    def _transcribe_mlx(self, audio_path: str) -> str:
        """Transcribe using MLX-Whisper."""
        result = self._mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=self.get_model_path(),
        )
        return result.get("text", "").strip()
    
    def _transcribe_faster_whisper(self, audio_path: str) -> str:
        """Transcribe using faster-whisper."""
        segments, info = self._model.transcribe(audio_path, beam_size=5)
        
        # Combine all segments into a single string
        text = " ".join(segment.text for segment in segments)
        return text.strip()


# Convenience function for simple usage
def transcribe(audio_path: str, model_key: str = DEFAULT_MODEL) -> str:
    """
    Simple transcription function that creates a backend and transcribes.
    For repeated use, create a WhisperBackend instance instead.
    """
    backend = WhisperBackend(model_key)
    return backend.transcribe(audio_path)
