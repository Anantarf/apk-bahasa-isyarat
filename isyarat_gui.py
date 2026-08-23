"""
Sign Language Recognition GUI Application
Clean OOP wrapper around existing SignLanguageRecognizer class
"""
import warnings
# Suppress protobuf deprecation warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

import customtkinter as ctk
from PIL import Image
import cv2
import threading
import tkinter as tk
from typing import Optional
from datetime import datetime
from isyarat import SignLanguageRecognizer
from app_logging import get_logger

logger = get_logger(__name__)


class SignLanguageGUI:
    """Main GUI application for sign language recognition."""
    
    # Application version
    VERSION = "1.0"
    
    # UI Constants
    MARGIN = 10
    SMALL_PADDING = 4
    MEDIUM_PADDING = 8
    LARGE_PADDING = 10
    HEADER_PADDING = (10, 8)
    BUTTON_HEIGHT = 50
    CORNER_RADIUS = 12
    UI_ELEMENTS_HEIGHT = 300  # Extra space for UI elements beyond camera
    
    # Color schemes
    COLOR_BUTTON_HOVER = ("#1f538d", "#144870")
    COLOR_EXIT_BG = ("#8B0000", "#660000")
    COLOR_EXIT_HOVER = ("#A52A2A", "#8B0000")
    COLOR_RESET_BG = ("#555555", "#444444")
    COLOR_RESET_HOVER = ("#666666", "#555555")
    COLOR_TEXT_GRAY = ("gray60", "gray40")
    
    # Help text constant
    HELP_TEXT = """How to Use:

1. Click START or press Space to begin.
2. Show your hand gesture to the camera.
3. Hold the gesture steady for 2.5 seconds.
4. Letters accumulate on screen (max 10).
5. Press R to reset accumulated text.

Keyboard Shortcuts:
- Space: Start/stop recognition
- R: Reset accumulated text
- Esc: Exit application

Supported Gestures:
- All 26 SIBI alphabet gestures (A-Z)
- Static gestures: A-I, K-Y
- Dynamic gestures: J (hook motion), Z (zigzag)

Tips for Best Results:
- Ensure good lighting conditions.
- Keep your hand clearly visible in frame.
- Avoid busy or distracting backgrounds."""
    
    def __init__(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.camera_width, self.camera_height = self._detect_camera_resolution()
        self.window = ctk.CTk()
        self._setup_window()
        self._bind_shortcuts()
        self._init_state()
        self._setup_ui()

    def _setup_window(self):
        """Configure the root window."""
        self.window.title(f"SIBI Sign Language Recognition v{self.VERSION}")
        try:
            self.window.iconbitmap("icon.ico")
        except (OSError, tk.TclError):
            pass

        window_width = self.camera_width + (self.MARGIN * 2)
        window_height = self.camera_height + self.UI_ELEMENTS_HEIGHT
        self.window.geometry(f"{window_width}x{window_height}")
        self.window.resizable(True, True)
        self.window.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _bind_shortcuts(self):
        """Bind application keyboard shortcuts."""
        self.window.bind("<space>", lambda e: self._toggle_recognition())
        self.window.bind("<Key-r>", lambda e: self._reset_text())
        self.window.bind("<Key-R>", lambda e: self._reset_text())
        self.window.bind("<Escape>", lambda e: self._on_closing())

    def _init_state(self):
        """Initialize mutable runtime state."""
        self._is_running = False
        self._video_thread: Optional[threading.Thread] = None
        self._recognizer: Optional[SignLanguageRecognizer] = None
        self._last_logged_text = ""
        self._detection_count = 0
    def _probe_camera(self, index: int) -> Optional[dict]:
        """Return camera info when an index opens and captures a frame."""
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        try:
            if not cap.isOpened():
                return None
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if width <= 0 or height <= 0:
                return None
            ret, frame = cap.read()
            if not ret or frame is None:
                return None
            return {"index": index, "width": width, "height": height}
        finally:
            cap.release()
    def _detect_camera_resolution(self) -> tuple:
        """Detect camera native resolution with smart camera selection."""
        detected_cameras = [cam for index in range(3) if (cam := self._probe_camera(index))]
        if not detected_cameras:
            self._camera_index = 0
            return 640, 480

        selected_cam = next((cam for cam in detected_cameras if cam["index"] == 1), detected_cameras[0])
        self._camera_index = selected_cam["index"]
        return selected_cam["width"], selected_cam["height"]
    def _setup_ui(self):
        """Initialize all UI components with clean layout."""
        self._setup_header()
        self._setup_video_section()
        self._setup_controls()
        self._setup_shortcuts_and_footer()
    
    def _setup_header(self):
        """Setup header section with title and help button."""
        header_frame = ctk.CTkFrame(self.window, corner_radius=self.CORNER_RADIUS)
        header_frame.pack(side="top", fill="x", padx=self.MARGIN, pady=self.HEADER_PADDING)
        
        title_container = ctk.CTkFrame(header_frame, fg_color="transparent")
        title_container.pack(pady=self.LARGE_PADDING, fill="x", padx=self.MARGIN)
        
        ctk.CTkLabel(
            title_container,
            text="SIBI Sign Language Recognition",
            font=ctk.CTkFont(size=26, weight="bold")
        ).pack(side="left", expand=True)
        
        ctk.CTkButton(
            title_container,
            text="?",
            width=35,
            height=35,
            corner_radius=17,
            command=self._show_help,
            font=ctk.CTkFont(size=18, weight="bold"),
            fg_color=("#3a7ebf", "#1f538d"),
            hover_color=("#325a8c", "#144870")
        ).pack(side="right")
    
    def _setup_video_section(self):
        """Setup video canvas with gray frame container."""
        video_frame = ctk.CTkFrame(
            self.window,
            fg_color=("#d0d0d0", "#2b2b2b"),
            corner_radius=self.CORNER_RADIUS
        )
        video_frame.pack(side="top", padx=self.MARGIN, pady=(self.MARGIN * 2))
        
        self.canvas = ctk.CTkLabel(
            video_frame,
            text="Click START to begin recognition.\n\n[camera]",
            fg_color=("#2b2b2b", "#2b2b2b"),
            corner_radius=self.CORNER_RADIUS,
            font=ctk.CTkFont(size=16),
            width=self.camera_width,
            height=self.camera_height,
            anchor="center"
        )
        self.canvas.pack(padx=self.MARGIN, pady=self.MARGIN)
    
    def _create_control_button(self, parent, text: str, command, **kwargs):
        """Create a consistently styled control button."""
        options = {
            "text": text,
            "command": command,
            "height": self.BUTTON_HEIGHT,
            "font": ctk.CTkFont(size=15, weight="bold"),
            "corner_radius": self.CORNER_RADIUS,
        }
        options.update(kwargs)
        return ctk.CTkButton(parent, **options)
    def _setup_controls(self):
        """Setup control buttons (START, RESET, and EXIT)."""
        control_frame = ctk.CTkFrame(self.window, fg_color="transparent")
        control_frame.pack(side="top", padx=self.MARGIN, pady=self.LARGE_PADDING)

        button_container = ctk.CTkFrame(control_frame, fg_color="transparent")
        button_container.pack()

        self.start_btn = self._create_control_button(
            button_container,
            "START",
            self._start_recognition,
            hover_color=self.COLOR_BUTTON_HOVER,
        )
        self.start_btn.pack(side="left", fill="x", expand=True, padx=(0, self.SMALL_PADDING))

        self.reset_btn = self._create_control_button(
            button_container,
            "RESET (R)",
            self._reset_text,
            fg_color=self.COLOR_RESET_BG,
            hover_color=self.COLOR_RESET_HOVER,
            state="disabled",
        )
        self.reset_btn.pack(side="left", fill="x", expand=True, padx=self.SMALL_PADDING)

        self.exit_btn = self._create_control_button(
            button_container,
            "EXIT",
            self._on_closing,
            fg_color=self.COLOR_EXIT_BG,
            hover_color=self.COLOR_EXIT_HOVER,
        )
        self.exit_btn.pack(side="left", fill="x", expand=True, padx=(self.SMALL_PADDING, 0))
    def _setup_shortcuts_and_footer(self):
        """Setup keyboard shortcuts hint and footer."""
        shortcut_frame = ctk.CTkFrame(self.window, fg_color="transparent")
        shortcut_frame.pack(side="top", padx=self.MARGIN, pady=(self.MEDIUM_PADDING, self.SMALL_PADDING))
        
        ctk.CTkLabel(
            shortcut_frame,
            text="Shortcuts: Space (start/stop) | R (reset) | Esc (exit)",
            font=ctk.CTkFont(size=11),
            text_color=self.COLOR_TEXT_GRAY
        ).pack()
        
        footer_frame = ctk.CTkFrame(self.window, fg_color="transparent")
        footer_frame.pack(side="top", padx=self.MARGIN, pady=(self.SMALL_PADDING, self.LARGE_PADDING))
        
        ctk.CTkLabel(
            footer_frame,
            text="(c) 2026 Ananta Raihan | Tugas Akhir",
            font=ctk.CTkFont(size=11),
            text_color=("gray70", "gray50")
        ).pack(pady=self.MEDIUM_PADDING)
        
    def _show_loading_state(self):
        """Show camera initialization state."""
        self.canvas.configure(text="Initializing camera, please wait...", font=ctk.CTkFont(size=14))
        self.window.update_idletasks()
        self.start_btn.configure(text="STOP", command=self._stop_recognition, state="disabled")

    def _start_session_log(self):
        """Reset and print detection session header."""
        self._last_logged_text = ""
        self._detection_count = 0
        logger.info("%s", "=" * 60)
        logger.info("DETECTION SESSION STARTED")
        logger.info("%s", "=" * 60)
        logger.info("#\tTime\t\t\tLetter")
        logger.info("%s", "-" * 60)

    def _reset_idle_canvas(self):
        """Restore idle canvas and controls."""
        self.start_btn.configure(text="START", command=self._start_recognition, state="normal")
        self.reset_btn.configure(state="disabled")
        self.canvas.configure(
            image=None,
            fg_color="#2b2b2b",
            text="Click START to begin recognition.\n\n[camera]",
            font=ctk.CTkFont(size=16),
        )

    def _log_session_summary(self):
        """Print detection session summary when there were detections."""
        if self._detection_count <= 0:
            return
        logger.info("%s", "-" * 60)
        logger.info("Session ended: %s letter(s) detected", self._detection_count)
        logger.info("%s", "=" * 60)

    def _cleanup_recognizer(self):
        """Release recognizer resources if initialized."""
        if self._recognizer:
            self._recognizer.cleanup()
            self._recognizer = None
    def _toggle_recognition(self):
        """Toggle recognition on/off (for Space key)."""
        if self._is_running:
            self._stop_recognition()
        else:
            self._start_recognition()
        
    def _start_recognition(self):
        """Start video processing in separate thread."""
        if self._is_running:
            return

        self._show_loading_state()
        self._is_running = True
        self._start_session_log()

        self._video_thread = threading.Thread(target=self._init_recognizer_and_loop, daemon=True)
        self._video_thread.start()

    def _init_recognizer_and_loop(self):
        """Initialize recognizer off the UI thread, then run video loop."""
        try:
            self._recognizer = SignLanguageRecognizer()
            self.window.after(0, self._enable_controls)
        except (RuntimeError, OSError, ValueError) as e:
            self.window.after(0, self._show_init_error, str(e))
            return

        self._video_loop()
    def _enable_controls(self):
        """Re-enable buttons after successful init."""
        self.start_btn.configure(state="normal")
        self.reset_btn.configure(state="normal")
        
    def _stop_recognition(self):
        """Stop video processing gracefully."""
        if not self._is_running:
            return

        self._is_running = False
        self._fade_canvas()
        self._wait_for_video_thread()
        self._cleanup_recognizer()
        self._log_session_summary()
        self._reset_idle_canvas()

    def _wait_for_video_thread(self):
        """Wait briefly for the video thread to exit."""
        if not self._video_thread or not self._video_thread.is_alive():
            return
        self._video_thread.join(timeout=2.0)
        if self._video_thread.is_alive():
            logger.warning("Video thread did not stop gracefully")
    def _reset_text(self):
        """Reset accumulated text (R key or button)."""
        if not self._is_running or not self._recognizer:
            return
        self._recognizer.reset_text()
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        logger.info("---\t%s\tRESET", timestamp)
        self._last_logged_text = ""

    def _process_video_frame(self, frame) -> bool:
        """Process a captured frame and schedule UI updates."""
        recognizer = self._recognizer
        if not recognizer:
            return False

        processed_frame, current_text = recognizer.process_frame(frame)
        self.window.after(0, self._log_detection, current_text)
        self.window.after(0, self._update_frame, processed_frame)
        return True
    def _video_loop(self):
        """Main video processing loop (runs in separate thread)."""
        cap = cv2.VideoCapture(self._camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self.window.after(0, self._show_camera_error)
            return

        try:
            while self._is_running:
                try:
                    ret, frame = cap.read()
                    if not ret:
                        self.window.after(0, self._show_camera_error)
                        break
                    if not self._process_video_frame(frame):
                        break
                except (cv2.error, RuntimeError, ValueError, TypeError) as e:
                    self.window.after(0, self._show_processing_error, str(e))
                    break
        finally:
            cap.release()
            if self._is_running:
                self.window.after(0, self._stop_recognition)
    def _log_detection(self, current_text: str):
        """Log each detected letter individually to terminal."""
        if not self._is_running or current_text == self._last_logged_text:
            return
        
        # Handle text reset (text shortened or cleared)
        if len(current_text) < len(self._last_logged_text):
            self._last_logged_text = current_text
            return
        
        # Log each new letter individually
        if len(current_text) > len(self._last_logged_text):
            new_letters = current_text[len(self._last_logged_text):]
            for letter in new_letters:
                self._detection_count += 1
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                logger.info("%s\t%s\t%s", self._detection_count, timestamp, letter)
        
        self._last_logged_text = current_text
    
    def _update_frame(self, frame):
        """Update canvas with new frame (called from main thread)."""
        if not self._is_running:
            return
            
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image and CTkImage
        pil_img = Image.fromarray(rgb_frame)
        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(self.camera_width, self.camera_height))
        
        # Update canvas
        self.canvas.configure(image=ctk_img, text="", compound="center")
        self.canvas.image = ctk_img  # Keep reference
        
    def _on_closing(self):
        """Clean shutdown on window close."""
        self._stop_recognition()
        self.window.destroy()
    
    def _fade_canvas(self):
        """Smooth fade transition effect."""
        try:
            for alpha in [0.6, 0.3, 0.0]:
                color = f"#{int(43 * alpha):02x}" * 3
                self.canvas.configure(fg_color=color)
                self.window.update()
        except (RuntimeError, tk.TclError):
            pass  # Window closing, skip gracefully
    
    def _build_help_window(self):
        """Create and configure the help dialog window."""
        help_window = ctk.CTkToplevel(self.window)
        help_window.title("How to Use")
        help_window.geometry("460x520")
        help_window.resizable(False, False)
        help_window.transient(self.window)
        help_window.grab_set()
        help_window.bind("<Escape>", lambda e: help_window.destroy())
        help_window.bind("<Return>", lambda e: help_window.destroy())
        return help_window

    def _add_help_content(self, help_window) -> None:
        """Populate help dialog content."""
        content_frame = ctk.CTkFrame(help_window, fg_color="transparent")
        content_frame.pack(fill="both", expand=True, padx=20, pady=20)
        ctk.CTkLabel(
            content_frame,
            text="SIBI Sign Language Recognition",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(pady=(0, 12))
        ctk.CTkLabel(content_frame, text=self.HELP_TEXT, justify="left", font=ctk.CTkFont(size=13)).pack(pady=(0, 12))
        ctk.CTkLabel(
            content_frame,
            text=f"Version {self.VERSION} | (c) 2026 Ananta Raihan",
            font=ctk.CTkFont(size=11),
            text_color=("gray60", "gray40"),
        ).pack(pady=(8, 0))
        ctk.CTkButton(
            content_frame,
            text="Got it!",
            command=help_window.destroy,
            width=120,
            height=35,
            corner_radius=10,
        ).pack(pady=(10, 0))
    def _show_help(self):
        """Display help/about dialog."""
        help_window = self._build_help_window()
        self._add_help_content(help_window)
    def _show_error(self, error_type: str, message: str):
        """Display error message with retry option."""
        self._is_running = False
        # Cleanup recognizer to avoid resource leak
        if self._recognizer:
            self._recognizer.cleanup()
            self._recognizer = None
        self.canvas.configure(
            text=f"{error_type}\n\n{message}\n\nClick START to retry.",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#2b2b2b"
        )
        self.start_btn.configure(
            text="START",
            command=self._start_recognition,
            state="normal"
        )
        self.reset_btn.configure(state="disabled")
    
    def _show_camera_error(self):
        """Display camera error."""
        self._show_error("Camera Error", "Unable to access camera.\nPlease check your camera connection and try again.")
    
    def _show_processing_error(self, error_msg: str):
        """Display processing error."""
        self._show_error("Processing Error", error_msg)
    
    def _show_init_error(self, error_msg: str):
        """Display initialization error."""
        self._show_error("Initialization Error", f"{error_msg}\n\nPlease check model file and try again.")
        
    def run(self):
        """Start the GUI application."""
        self.window.mainloop()


def main():
    """Entry point for GUI application."""
    app = SignLanguageGUI()
    app.run()


if __name__ == "__main__":
    main()










