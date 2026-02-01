#!/usr/bin/env python3
"""
System Tray App
========================
A system tray speech recognition app.

Double-tap triggers recording:
- macOS: Command (⌘)
- Windows/Linux: Control (Ctrl)

Transcription is typed at the cursor position.
"""

import json
import os
import platform
import sys
import wave
import tempfile
import threading
import time

import pyaudio
import pyautogui
from pynput import keyboard

from whisper_backend import WhisperBackend
from PIL import Image, ImageDraw

# Try to import pystray (cross-platform system tray)
try:
    import pystray
    from pystray import MenuItem as Item, Menu
except ImportError:
    print("Error: pystray not installed. Run: pip install pystray Pillow")
    sys.exit(1)


# Audio recording settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Whisper expects 16kHz audio

# Default model
DEFAULT_MODEL = "large-v3"

# Preferences file (in same directory as script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PREFS_FILE = os.path.join(SCRIPT_DIR, "preferences.json")

# Default preferences
DEFAULT_PREFS = {
    "press_enter_after_paste": False,
    "model": DEFAULT_MODEL,
}


def create_icon(color="white", recording=False):
    """Create a simple microphone icon."""
    size = 64
    image = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    
    # Microphone body color
    if recording:
        mic_color = (255, 59, 48)  # Red when recording
    else:
        mic_color = (255, 255, 255) if color == "white" else (0, 0, 0)
    
    # Draw microphone body (rounded rectangle)
    draw.rounded_rectangle([20, 8, 44, 40], radius=8, fill=mic_color)
    
    # Draw microphone stand
    draw.arc([16, 24, 48, 52], start=0, end=180, fill=mic_color, width=3)
    draw.line([32, 52, 32, 58], fill=mic_color, width=3)
    draw.line([22, 58, 42, 58], fill=mic_color, width=3)
    
    return image


class SpeechRecognizerTray:
    """System tray speech recognizer using Whisper."""
    
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.is_recording = False
        self.running = True
        self.lock = threading.Lock()
        
        # Load preferences
        self.prefs = self._load_prefs()
        
        # Initialize Whisper backend (auto-detects platform)
        print(f"Initializing Whisper backend...")
        model = self.prefs.get("model", DEFAULT_MODEL)
        self.backend = WhisperBackend(model)
        
        # Double-tap detection
        self.last_press_time = 0
        self.double_tap_threshold = 0.4  # seconds
        
        # Determine trigger keys based on platform
        if platform.system() == "Darwin":
            self.trigger_keys = {keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r}
            self.trigger_key_name = "⌘"
        else:
            self.trigger_keys = {keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r}
            self.trigger_key_name = "Ctrl"
        
        # System tray icon
        self.icon = None
        self.keyboard_listener = None
        
        # Preload model
        print(f"Preloading model...")
        self._preload_model()
        print("Model loaded!")
    
    def _load_prefs(self):
        """Load preferences from JSON file."""
        try:
            if os.path.exists(PREFS_FILE):
                with open(PREFS_FILE, 'r') as f:
                    prefs = json.load(f)
                    # Merge with defaults for any missing keys
                    return {**DEFAULT_PREFS, **prefs}
        except Exception as e:
            print(f"⚠️  Could not load preferences: {e}")
        return DEFAULT_PREFS.copy()
    
    def _save_prefs(self):
        """Save preferences to JSON file."""
        try:
            with open(PREFS_FILE, 'w') as f:
                json.dump(self.prefs, f, indent=2)
        except Exception as e:
            print(f"⚠️  Could not save preferences: {e}")
    
    def _preload_model(self):
        """Preload the Whisper model to avoid delay on first transcription."""
        self.backend.preload()
    
    def _update_icon(self):
        """Update the tray icon based on recording state."""
        if self.icon:
            self.icon.icon = create_icon(recording=self.is_recording)
    
    def start_recording(self):
        """Start recording audio from microphone."""
        with self.lock:
            if self.is_recording:
                return
            
            self.is_recording = True
            self.frames = []
            
            self.stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
            
            print("🔴 Recording...")
            self._update_icon()
            
            # Start recording thread
            self.record_thread = threading.Thread(target=self._record_audio)
            self.record_thread.start()
    
    def _record_audio(self):
        """Record audio in a separate thread."""
        while self.is_recording and self.running:
            try:
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                self.frames.append(data)
            except Exception as e:
                print(f"Recording error: {e}")
                break
    
    def stop_recording(self):
        """Stop recording and transcribe the audio."""
        with self.lock:
            if not self.is_recording:
                return
            
            self.is_recording = False
        
        self._update_icon()
        
        # Wait for recording thread to finish
        if hasattr(self, 'record_thread'):
            self.record_thread.join(timeout=1.0)
        
        # Stop and close the stream
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        if not self.frames:
            print("⚠️  No audio recorded")
            return
        
        print("⏹️  Transcribing...")
        
        # Save to temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        
        try:
            self._save_wav(temp_path)
            self._transcribe(temp_path)
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
    
    def _save_wav(self, filepath: str):
        """Save recorded frames to WAV file."""
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(self.frames))
    
    def _transcribe(self, audio_path: str):
        """Transcribe audio file using Whisper and type at cursor."""
        try:
            text = self.backend.transcribe(audio_path)
            
            if text:
                print(f"📝 Inserting: {text}")
                # Use clipboard to paste text (more reliable on macOS)
                self._insert_text(text)
            else:
                print("⚠️  No speech detected")
            
        except Exception as e:
            print(f"❌ Transcription error: {e}")
    
    def _insert_text(self, text: str):
        """Insert text at cursor position using clipboard paste."""
        import subprocess
        import platform
        
        if platform.system() == "Darwin":  # macOS
            # Copy text to clipboard
            process = subprocess.Popen(['pbcopy'], stdin=subprocess.PIPE)
            process.communicate(text.encode('utf-8'))
            print(f"📋 Copied to clipboard: {text[:50]}...")
            
            # Use Quartz to simulate Cmd+V
            try:
                from Quartz import (
                    CGEventCreateKeyboardEvent,
                    CGEventPost,
                    CGEventSetFlags,
                    kCGEventFlagMaskCommand,
                    kCGHIDEventTap
                )
                
                # Key code for 'v' is 9
                v_keycode = 9
                
                # Longer delay to ensure the app regains focus
                print("⏳ Waiting for focus to return...")
                time.sleep(0.3)
                
                # Key down with Cmd
                event = CGEventCreateKeyboardEvent(None, v_keycode, True)
                CGEventSetFlags(event, kCGEventFlagMaskCommand)
                CGEventPost(kCGHIDEventTap, event)
                
                time.sleep(0.05)
                
                # Key up
                event = CGEventCreateKeyboardEvent(None, v_keycode, False)
                CGEventSetFlags(event, kCGEventFlagMaskCommand)
                CGEventPost(kCGHIDEventTap, event)
                
                time.sleep(0.05)
                
                # Optionally press Enter/Return (keycode 36)
                # Optionally press Enter/Return (keycode 36)
                if self.prefs.get("press_enter_after_paste", False):
                    # Small delay to ensure paste completes and app handles it
                    time.sleep(0.1) 
                    return_keycode = 36
                    event = CGEventCreateKeyboardEvent(None, return_keycode, True)
                    CGEventPost(kCGHIDEventTap, event)
                    time.sleep(0.01)
                    event = CGEventCreateKeyboardEvent(None, return_keycode, False)
                    CGEventPost(kCGHIDEventTap, event)
                    print("✅ Text pasted with Enter!")
                else:
                    print("✅ Text pasted!")
                
            except Exception as e:
                print(f"⚠️  Quartz paste failed: {e}")
                print("� Text is in clipboard - press ⌘V to paste manually")
        else:
            # Windows/Linux: use pynput
            self._clipboard_paste_pynput(text)
    
    def _clipboard_paste_pynput(self, text: str):
        """Fallback: paste text using pynput."""
        import subprocess
        
        # Copy to clipboard (Windows)
        subprocess.run(['clip'], input=text.encode(), shell=True)
        
        # Simulate Ctrl+V
        from pynput.keyboard import Controller, Key
        kb = Controller()
        time.sleep(0.05)
        kb.press(Key.ctrl)
        kb.press('v')
        kb.release('v')
        kb.release(Key.ctrl)
        
        # Optionally press Enter
        if self.prefs.get("press_enter_after_paste", False):
            time.sleep(0.1)
            kb.press(Key.enter)
            kb.release(Key.enter)
            print("✅ Text pasted with Enter!")
        else:
            print("✅ Text pasted!")
    
    def _is_trigger_key(self, key):
        """Check if the key is one of the trigger keys (Cmd or Ctrl)."""
        return key in self.trigger_keys
    
    def on_key_release(self, key):
        """Handle key release events - detect double-tap of trigger key."""
        if self._is_trigger_key(key):
            current_time = time.time()
            time_since_last = current_time - self.last_press_time
            
            if time_since_last < self.double_tap_threshold:
                # Double-tap detected! Toggle recording
                if self.is_recording:
                    self.stop_recording()
                else:
                    self.start_recording()
                self.last_press_time = 0
            else:
                self.last_press_time = current_time
    
    def set_model(self, model_key):
        """Change the current model."""
        available = self.backend.available_models
        if model_key in available:
            self.backend.set_model(model_key)
            self.prefs["model"] = model_key
            self._save_prefs()
            print(f"Switched to model: {model_key}")
            # Preload the new model
            threading.Thread(target=self._preload_model, daemon=True).start()
    
    def toggle_enter_after_paste(self):
        """Toggle the 'press enter after paste' setting."""
        self.prefs["press_enter_after_paste"] = not self.prefs.get("press_enter_after_paste", False)
        self._save_prefs()
        status = "enabled" if self.prefs["press_enter_after_paste"] else "disabled"
        print(f"Press Enter after paste: {status}")
    
    def create_menu(self):
        """Create the system tray menu."""
        def make_model_action(key):
            return lambda: self.set_model(key)
        
        def is_model_selected(key):
            return lambda item: self.backend.model_key == key
        
        available_models = self.backend.available_models
        model_items = [
            Item(
                f"{'●' if self.backend.model_key == key else '○'} {key}",
                make_model_action(key),
                checked=is_model_selected(key)
            )
            for key in available_models.keys()
        ]
        
        return Menu(
            Item(f"🎤 Double-tap {self.trigger_key_name} to record", lambda: None, enabled=False),
            Menu.SEPARATOR,
            Item("Model", Menu(*model_items)),
            Item(
                "Press Enter after paste",
                self.toggle_enter_after_paste,
                checked=lambda item: self.prefs.get("press_enter_after_paste", False)
            ),
            Menu.SEPARATOR,
            Item("Quit", self.quit_app),
        )
    
    def quit_app(self):
        """Quit the application."""
        print("👋 Exiting...")
        self.running = False
        if self.keyboard_listener:
            self.keyboard_listener.stop()
        if self.icon:
            self.icon.stop()
        self.audio.terminate()
    
    def run(self):
        """Run the system tray application."""
        # Start keyboard listener in a separate thread
        self.keyboard_listener = keyboard.Listener(on_release=self.on_key_release)
        self.keyboard_listener.start()
        
        # Create and run the system tray icon
        self.icon = pystray.Icon(
            "speech_recognition",
            create_icon(),
            "Speech Recognition",
            menu=self.create_menu()
        )
        
        print(f"System tray app running. Double-tap {self.trigger_key_name} to record.")
        self.icon.run()


def main():
    """Main entry point."""
    try:
        app = SpeechRecognizerTray()
        app.run()
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
