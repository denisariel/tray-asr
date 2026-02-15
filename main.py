#!/usr/bin/env python3
"""
System Tray App
========================
A system tray speech recognition app.

Double-tap triggers recording:
- macOS: Command (⌘)
- Windows/Linux: Control (Ctrl)

Transcription is typed directly at the cursor position.
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
MAX_RECORDING_TIME = 30  # Maximum recording duration in seconds

# Default model
DEFAULT_MODEL = "large-v3"

# Preferences file (in same directory as script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PREFS_FILE = os.path.join(SCRIPT_DIR, "preferences.json")

# Default preferences
DEFAULT_PREFS = {
    "press_enter_after_paste": False,
    "model": DEFAULT_MODEL,
    "input_device_index": None,  # None = system default
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
        
        # Scan for input devices
        self.input_devices = self._get_input_devices()
        self.selected_device_index = self.prefs.get("input_device_index", None)
        
        # Validate the saved device still exists
        if self.selected_device_index is not None:
            valid_indices = [d["index"] for d in self.input_devices]
            if self.selected_device_index not in valid_indices:
                print(f"⚠️  Saved input device (index {self.selected_device_index}) no longer available. Resetting to default.")
                self.selected_device_index = None
                self.prefs["input_device_index"] = None
                self._save_prefs()
        
        if self.input_devices:
            if self.selected_device_index is not None:
                name = next((d["name"] for d in self.input_devices if d["index"] == self.selected_device_index), "Unknown")
                print(f"🎙️  Input device: {name} (index {self.selected_device_index})")
            else:
                print(f"🎙️  Input device: System default")
        else:
            print("⚠️  No input devices found! Recording will be disabled.")
        
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
    
    def _get_input_devices(self):
        """Enumerate available audio input devices."""
        devices = []
        for i in range(self.audio.get_device_count()):
            try:
                info = self.audio.get_device_info_by_index(i)
                if info.get("maxInputChannels", 0) > 0:
                    devices.append({
                        "index": i,
                        "name": info["name"],
                        "channels": info["maxInputChannels"],
                        "sample_rate": int(info["defaultSampleRate"]),
                    })
            except Exception:
                pass
        
        if devices:
            print(f"🎙️  Found {len(devices)} input device(s):")
            for d in devices:
                print(f"    [{d['index']}] {d['name']}")
        
        return devices
    
    def set_input_device(self, device_index):
        """Set the input device to use for recording."""
        self.selected_device_index = device_index
        self.prefs["input_device_index"] = device_index
        self._save_prefs()
        
        if device_index is None:
            print("🎙️  Input device: System default")
        else:
            name = next((d["name"] for d in self.input_devices if d["index"] == device_index), "Unknown")
            print(f"🎙️  Input device: {name} (index {device_index})")
        
        # Rebuild menu to update checkmarks
        if self.icon:
            self.icon.menu = self.create_menu()
    
    def refresh_input_devices(self):
        """Rescan audio input devices (reinitializes PyAudio to detect new hardware)."""
        # PyAudio caches devices at init time — must recreate to detect changes
        self.audio.terminate()
        self.audio = pyaudio.PyAudio()
        self.input_devices = self._get_input_devices()
        
        # Validate current selection still exists
        if self.selected_device_index is not None:
            valid_indices = [d["index"] for d in self.input_devices]
            if self.selected_device_index not in valid_indices:
                print(f"⚠️  Selected device no longer available. Resetting to default.")
                self.selected_device_index = None
                self.prefs["input_device_index"] = None
                self._save_prefs()
        
        # Rebuild menu
        if self.icon:
            self.icon.menu = self.create_menu()
    
    def start_recording(self):
        """Start recording audio from microphone."""
        with self.lock:
            if self.is_recording:
                return
            
            # Check if any input devices are available
            if not self.input_devices:
                print("⚠️  No input devices available — cannot record.")
                return
            
            self.is_recording = True
            self.frames = []
            self.recording_start_time = time.time()  # Track when recording started
            
            # Build open() kwargs, adding device index if user selected one
            open_kwargs = dict(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
            )
            if self.selected_device_index is not None:
                open_kwargs["input_device_index"] = self.selected_device_index
            
            try:
                self.stream = self.audio.open(**open_kwargs)
            except OSError as e:
                print(f"❌ Failed to open audio device: {e}")
                print("   Try selecting a different input device from the tray menu.")
                self.is_recording = False
                return
            
            print("🔴 Recording...")
            self._update_icon()
            
            # Start recording thread
            self.record_thread = threading.Thread(target=self._record_audio)
            self.record_thread.start()
    
    def _record_audio(self):
        """Record audio in a separate thread with automatic timeout."""
        while self.is_recording and self.running:
            try:
                # Check if we've exceeded the maximum recording time
                elapsed_time = time.time() - self.recording_start_time
                if elapsed_time >= MAX_RECORDING_TIME:
                    print(f"⏱️  Maximum recording time ({MAX_RECORDING_TIME}s) reached. Stopping...")
                    # Stop recording from within the thread
                    threading.Thread(target=self.stop_recording, daemon=True).start()
                    break
                
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
        """Insert text at cursor position by typing directly (no clipboard)."""
        import platform
        
        if platform.system() == "Darwin":  # macOS
            # Type text using CGEventKeyboardSetUnicodeString (fast, chunked)
            try:
                from Quartz import (
                    CGEventCreateKeyboardEvent,
                    CGEventKeyboardSetUnicodeString,
                    CGEventPost,
                    kCGHIDEventTap
                )
                
                # Wait for focus to return to the target application
                print("⏳ Waiting for focus to return...")
                time.sleep(0.3)
                
                print(f"⌨️  Typing: {text[:50]}...")
                
                # Send text in chunks of 20 chars using Unicode keyboard events
                chunk_size = 20
                for i in range(0, len(text), chunk_size):
                    chunk = text[i:i + chunk_size]
                    # Key down – attach the Unicode string to the event
                    event_down = CGEventCreateKeyboardEvent(None, 0, True)
                    CGEventKeyboardSetUnicodeString(event_down, len(chunk), chunk)
                    CGEventPost(kCGHIDEventTap, event_down)
                    # Key up
                    event_up = CGEventCreateKeyboardEvent(None, 0, False)
                    CGEventPost(kCGHIDEventTap, event_up)
                    time.sleep(0.01)  # Small delay between chunks for reliability
                
                # Optionally press Enter/Return
                if self.prefs.get("press_enter_after_paste", False):
                    time.sleep(0.05)
                    return_keycode = 36
                    event = CGEventCreateKeyboardEvent(None, return_keycode, True)
                    CGEventPost(kCGHIDEventTap, event)
                    time.sleep(0.01)
                    event = CGEventCreateKeyboardEvent(None, return_keycode, False)
                    CGEventPost(kCGHIDEventTap, event)
                    print("✅ Text typed with Enter!")
                else:
                    print("✅ Text typed!")
                
            except Exception as e:
                print(f"⚠️  Quartz typing failed: {e}, falling back to pyautogui...")
                try:
                    time.sleep(0.3)
                    pyautogui.write(text, interval=0.02)
                    if self.prefs.get("press_enter_after_paste", False):
                        pyautogui.press('enter')
                        print("✅ Text typed with Enter!")
                    else:
                        print("✅ Text typed!")
                except Exception as e2:
                    print(f"❌ Fallback typing also failed: {e2}")
        else:
            # Windows/Linux: use pyautogui directly
            self._type_text_direct(text)
    
    def _type_text_direct(self, text: str):
        """Type text at cursor position (Windows/Linux)."""
        try:
            print(f"⌨️  Typing: {text[:50]}...")
            time.sleep(0.3)  # Wait for focus

            if platform.system() == "Windows":
                self._type_unicode_win32(text)
            else:
                # Linux fallback
                pyautogui.write(text, interval=0.02)

            # Optionally press Enter
            if self.prefs.get("press_enter_after_paste", False):
                time.sleep(0.05)
                pyautogui.press('enter')
                print("✅ Text typed with Enter!")
            else:
                print("✅ Text typed!")
        except Exception as e:
            print(f"❌ Typing failed: {e}")

    def _type_unicode_win32(self, text: str):
        """Type Unicode text on Windows using SendInput with KEYEVENTF_UNICODE."""
        import ctypes
        from ctypes import wintypes

        ULONG_PTR = ctypes.POINTER(ctypes.c_ulong)

        INPUT_KEYBOARD = 1
        KEYEVENTF_UNICODE = 0x0004
        KEYEVENTF_KEYUP = 0x0002

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [
                ("dx", wintypes.LONG),
                ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ULONG_PTR),
            ]

        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [
                ("wVk", wintypes.WORD),
                ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ULONG_PTR),
            ]

        class HARDWAREINPUT(ctypes.Structure):
            _fields_ = [
                ("uMsg", wintypes.DWORD),
                ("wParamL", wintypes.WORD),
                ("wParamH", wintypes.WORD),
            ]

        class INPUT(ctypes.Structure):
            class _INPUT(ctypes.Union):
                _fields_ = [
                    ("mi", MOUSEINPUT),
                    ("ki", KEYBDINPUT),
                    ("hi", HARDWAREINPUT),
                ]
            _fields_ = [
                ("type", wintypes.DWORD),
                ("_input", _INPUT),
            ]

        SendInput = ctypes.windll.user32.SendInput
        SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
        SendInput.restype = wintypes.UINT

        for char in text:
            # Key down
            inp_down = INPUT()
            inp_down.type = INPUT_KEYBOARD
            inp_down._input.ki.wVk = 0
            inp_down._input.ki.wScan = ord(char)
            inp_down._input.ki.dwFlags = KEYEVENTF_UNICODE
            inp_down._input.ki.time = 0
            inp_down._input.ki.dwExtraInfo = None

            # Key up
            inp_up = INPUT()
            inp_up.type = INPUT_KEYBOARD
            inp_up._input.ki.wVk = 0
            inp_up._input.ki.wScan = ord(char)
            inp_up._input.ki.dwFlags = KEYEVENTF_UNICODE | KEYEVENTF_KEYUP
            inp_up._input.ki.time = 0
            inp_up._input.ki.dwExtraInfo = None

            SendInput(2, (INPUT * 2)(inp_down, inp_up), ctypes.sizeof(INPUT))
            time.sleep(0.005)  # Small delay for reliability
    
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
        
        # Input device menu items
        def make_device_action(idx):
            return lambda: self.set_input_device(idx)
        
        def is_device_selected(idx):
            return lambda item: self.selected_device_index == idx
        
        device_items = [
            Item(
                "System Default",
                make_device_action(None),
                checked=lambda item: self.selected_device_index is None
            )
        ]
        for dev in self.input_devices:
            device_items.append(
                Item(
                    dev["name"],
                    make_device_action(dev["index"]),
                    checked=is_device_selected(dev["index"])
                )
            )
        device_items.append(Menu.SEPARATOR)
        device_items.append(Item("🔄 Refresh Devices", lambda: self.refresh_input_devices()))
        
        return Menu(
            Item(f"🎤 Double-tap {self.trigger_key_name} to record", lambda: None, enabled=False),
            Menu.SEPARATOR,
            Item("Model", Menu(*model_items)),
            Item("Input Device", Menu(*device_items)),
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
