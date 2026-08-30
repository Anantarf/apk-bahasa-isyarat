"""
Sign Language Recognition GUI Application
Clean OOP wrapper around existing SignLanguageRecognizer class.
Features a modern, academic, clean 2-column Dashboard Layout.
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
from src.config import Config
from src.isyarat import SignLanguageRecognizer
from src.app_logging import get_logger

logger = get_logger(__name__)


class SignLanguageGUI:
    """Main GUI application for sign language recognition with Modern Dashboard Layout."""
    
    # Application version
    VERSION = "1.0"
    
    # UI Layout Constants
    MARGIN = 12
    PADDING_SMALL = 6
    PADDING_MEDIUM = 12
    PADDING_LARGE = 16
    CORNER_RADIUS = 10
    
    # Color Schemes (Clean Academic Dark Theme - Slate Palette)
    COLOR_BG_DARK = "#181825"
    COLOR_CARD_BG = "#1E1E2E"
    COLOR_CARD_BORDER = "#313244"
    
    COLOR_PRIMARY = "#2563EB"        # Blue Start
    COLOR_PRIMARY_HOVER = "#1D4ED8"
    COLOR_DANGER = "#DC2626"         # Red Stop
    COLOR_DANGER_HOVER = "#B91C1C"
    COLOR_NEUTRAL = "#374151"        # Gray Controls
    COLOR_NEUTRAL_HOVER = "#4B5563"
    COLOR_EXIT_HOVER = "#9CA3AF"
    
    COLOR_TEXT_WHITE = "#F9FAFB"
    COLOR_TEXT_MUTED = "#9CA3AF"
    COLOR_TEXT_ACCENT = "#60A5FA"
    
    # Help text constant
    HELP_TEXT = """Petunjuk Penggunaan:

1. Klik START atau tekan Space untuk memulai kamera.
2. Tunjukkan gestur tangan bahasa isyarat SIBI ke arah kamera.
3. Tahan gestur dengan stabil selama 2.5 detik.
4. Huruf yang terdeteksi akan terakumulasi pada kotak Hasil Terjemahan (maksimal 10 karakter).
5. Klik SALIN TEKS untuk menyalin hasil ke clipboard, atau tekan R / tombol RESET untuk mengosongkan teks.

Pintasan Keyboard (Shortcuts):
- Space: Mulai / Hentikan rekognisi
- R: Reset akumulasi teks
- Esc: Keluar dari aplikasi

Gestur Terdukung:
- 26 Abjad SIBI (A - Z)
- Gestur statis: A-I, K-Y
- Gestur dinamis: J (gerakan kail), Z (gerakan ziz-zag)

Tips Hasil Terbaik:
- Pastikan pencahayaan ruangan cukup terang.
- Posisikan tangan secara jelas di tengah bingkai kamera.
- Hindari latar belakang yang terlalu ramai atau bergerak."""
    
    def __init__(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.config = Config()
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

        # Calculate optimal width & height for 2-column layout
        window_width = max(self.camera_width + 440, 1080)
        window_height = max(self.camera_height + 170, 680)
        
        self.window.geometry(f"{window_width}x{window_height}")
        self.window.minsize(1020, 650)
        self.window.configure(fg_color=self.COLOR_BG_DARK)
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
        self._is_closing = False

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

    def _get_camera_names(self) -> dict[int, str]:
        """Auto-detect camera index to device hardware name mapping on Windows."""
        cam_map = {}
        try:
            from pygrabber.dshow_graph import FilterGraph
            graph = FilterGraph()
            devices = graph.get_input_devices()
            for idx, name in enumerate(devices):
                cam_map[idx] = name
        except Exception as e:
            logger.debug("Could not query camera device names: %s", e)
        return cam_map

    def _detect_camera_resolution(self) -> tuple:
        """Detect available cameras (0-5) with clean friendly hardware names."""
        self.camera_map = self._get_camera_names()
        self.detected_cameras = []  # List of tuples: (index, display_label)
        self.camera_label_to_index = {}  # Map display_label -> camera index
        
        preferred_index = int(getattr(self.config, "CAMERA_INDEX", 2))
        scan_order = [preferred_index] + [i for i in range(6) if i != preferred_index]
        
        detected_cams = []
        for index in scan_order:
            cam = self._probe_camera(index)
            if cam:
                detected_cams.append(cam)
                device_name = self.camera_map.get(index, f"Kamera {index}")
                display_label = device_name
                if display_label in self.camera_label_to_index:
                    display_label = f"{device_name} ({index})"
                    
                self.camera_label_to_index[display_label] = index
                if (index, display_label) not in self.detected_cameras:
                    self.detected_cameras.append((index, display_label))
        
        self.detected_cameras.sort(key=lambda x: x[0])
        if not detected_cams:
            self._camera_index = preferred_index
            self.detected_cameras = [(preferred_index, f"Kamera {preferred_index}")]
            self.camera_label_to_index[f"Kamera {preferred_index}"] = preferred_index
            return 640, 480

        preferred_cam = next((cam for cam in detected_cams if cam["index"] == preferred_index), detected_cams[0])
        self._camera_index = preferred_cam["index"]
        return preferred_cam["width"], preferred_cam["height"]

    def _setup_ui(self):
        """Initialize all UI components with a clean 2-column Dashboard layout."""
        self._setup_header()
        
        # Main split container (Grid: Left = Camera Feed, Right = Dashboard Panel)
        self.main_container = ctk.CTkFrame(self.window, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True, padx=self.MARGIN, pady=(0, self.PADDING_SMALL))
        self.main_container.grid_columnconfigure(0, weight=0)  # Camera column fixed width
        self.main_container.grid_columnconfigure(1, weight=1)  # Dashboard panel expands
        self.main_container.grid_rowconfigure(0, weight=1)

        self._setup_video_column()
        self._setup_dashboard_column()
        self._setup_footer()

    def _on_camera_selected(self, selected_option: str):
        """Handle camera selection change from dropdown."""
        try:
            new_index = self.camera_label_to_index.get(selected_option)
            if new_index is None:
                if selected_option.startswith("["):
                    new_index = int(selected_option.split("]")[0].replace("[", "").strip())
                else:
                    new_index = int(selected_option.replace("Kamera ", "").strip())
                
            if new_index is not None and new_index != self._camera_index:
                logger.info("Switching camera to index %s (%s)", new_index, selected_option)
                self._camera_index = new_index
                if self._is_running:
                    self._stop_recognition()
                    self.window.after(300, self._start_recognition)
        except ValueError:
            pass

    def _setup_header(self):
        """Setup header bar with title, camera selector, status pill, and help button."""
        header_frame = ctk.CTkFrame(
            self.window,
            fg_color=self.COLOR_CARD_BG,
            border_color=self.COLOR_CARD_BORDER,
            border_width=1,
            corner_radius=self.CORNER_RADIUS
        )
        header_frame.pack(side="top", fill="x", padx=self.MARGIN, pady=self.MARGIN)

        header_container = ctk.CTkFrame(header_frame, fg_color="transparent")
        header_container.pack(fill="x", padx=self.PADDING_MEDIUM, pady=10)

        # Title & Subtitle Info
        title_box = ctk.CTkFrame(header_container, fg_color="transparent")
        title_box.pack(side="left")

        ctk.CTkLabel(
            title_box,
            text="Sistem Pengenalan Bahasa Isyarat (SIBI)",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=self.COLOR_TEXT_WHITE
        ).pack(anchor="w")

        ctk.CTkLabel(
            title_box,
            text="Sistem Real-Time Penerjemah Abjad SIBI Berbasis Computer Vision",
            font=ctk.CTkFont(size=11),
            text_color=self.COLOR_TEXT_MUTED
        ).pack(anchor="w")

        # Right-side Action Box (Camera Selector, Status Badge & Help Button)
        action_box = ctk.CTkFrame(header_container, fg_color="transparent")
        action_box.pack(side="right")

        # Camera Selection Dropdown with actual hardware device names
        camera_options = [label for _, label in self.detected_cameras] if hasattr(self, "detected_cameras") and self.detected_cameras else ["Kamera 0"]
        initial_label = next((label for idx, label in self.detected_cameras if idx == self._camera_index), camera_options[0])

        self.camera_dropdown = ctk.CTkOptionMenu(
            action_box,
            values=camera_options,
            command=self._on_camera_selected,
            width=240,
            height=32,
            corner_radius=8,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#272738",
            button_color="#374151",
            button_hover_color="#4B5563"
        )
        self.camera_dropdown.set(initial_label)
        self.camera_dropdown.pack(side="left", padx=(0, 10))

        self.status_badge = ctk.CTkLabel(
            action_box,
            text="⚪ STANDBY",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#272738",
            text_color=self.COLOR_TEXT_MUTED,
            corner_radius=12
        )
        self.status_badge.pack(side="left", padx=(0, 10), pady=4)

        ctk.CTkButton(
            action_box,
            text="?",
            width=32,
            height=32,
            corner_radius=16,
            command=self._show_help,
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color=self.COLOR_NEUTRAL,
            hover_color=self.COLOR_NEUTRAL_HOVER
        ).pack(side="right")

    def _setup_video_column(self):
        """Setup video feed column (Left Column)."""
        left_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, self.PADDING_MEDIUM))

        # Camera Display Frame Container
        video_card = ctk.CTkFrame(
            left_frame,
            fg_color=self.COLOR_CARD_BG,
            border_color=self.COLOR_CARD_BORDER,
            border_width=1,
            corner_radius=self.CORNER_RADIUS
        )
        video_card.pack(side="top", fill="both", expand=True)

        self.canvas = ctk.CTkLabel(
            video_card,
            text="Kamera Belum Aktif\n\nKlik tombol ▶ MULAI DETEKSI di bawah untuk memulai analisis gestur SIBI.",
            fg_color="#11111B",
            corner_radius=self.CORNER_RADIUS - 2,
            font=ctk.CTkFont(size=13),
            width=self.camera_width,
            height=self.camera_height,
            anchor="center",
            text_color=self.COLOR_TEXT_MUTED
        )
        self.canvas.pack(padx=10, pady=10, fill="both", expand=True)

        # Control Bar Below Video
        control_frame = ctk.CTkFrame(left_frame, fg_color="transparent")
        control_frame.pack(side="top", fill="x", pady=(10, 0))

        self.start_btn = ctk.CTkButton(
            control_frame,
            text="▶ MULAI DETEKSI",
            command=self._start_recognition,
            height=44,
            font=ctk.CTkFont(size=14, weight="bold"),
            corner_radius=self.CORNER_RADIUS,
            fg_color=self.COLOR_PRIMARY,
            hover_color=self.COLOR_PRIMARY_HOVER
        )
        self.start_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))

        self.reset_btn = ctk.CTkButton(
            control_frame,
            text="🔄 RESET (R)",
            command=self._reset_text,
            height=44,
            font=ctk.CTkFont(size=13, weight="bold"),
            corner_radius=self.CORNER_RADIUS,
            fg_color=self.COLOR_NEUTRAL,
            hover_color=self.COLOR_NEUTRAL_HOVER,
            state="disabled"
        )
        self.reset_btn.pack(side="left", fill="x", expand=True, padx=4)

        self.exit_btn = ctk.CTkButton(
            control_frame,
            text="✕ EXIT",
            command=self._on_closing,
            height=44,
            width=80,
            font=ctk.CTkFont(size=13, weight="bold"),
            corner_radius=self.CORNER_RADIUS,
            fg_color="#2A2A3C",
            hover_color="#3F3F56",
            text_color=self.COLOR_TEXT_MUTED
        )
        self.exit_btn.pack(side="right", padx=(4, 0))

    def _setup_dashboard_column(self):
        """Setup right dashboard column containing results, stats, and shortcuts."""
        right_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        right_frame.grid(row=0, column=1, sticky="nsew")

        # -------------------------------------------------------------
        # CARD 1: Hasil Terjemahan (Accumulated Text Result)
        # -------------------------------------------------------------
        result_card = ctk.CTkFrame(
            right_frame,
            fg_color=self.COLOR_CARD_BG,
            border_color=self.COLOR_CARD_BORDER,
            border_width=1,
            corner_radius=self.CORNER_RADIUS
        )
        result_card.pack(fill="x", pady=(0, 10))

        result_header = ctk.CTkFrame(result_card, fg_color="transparent")
        result_header.pack(fill="x", padx=self.PADDING_MEDIUM, pady=(10, 4))

        ctk.CTkLabel(
            result_header,
            text="HASIL REKOGNISI TEKS SIBI",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=self.COLOR_TEXT_ACCENT
        ).pack(side="left")

        self.char_count_label = ctk.CTkLabel(
            result_header,
            text="0 / 10 Karakter",
            font=ctk.CTkFont(size=11),
            text_color=self.COLOR_TEXT_MUTED
        )
        self.char_count_label.pack(side="right")

        # Big Text Output Box Area
        text_container = ctk.CTkFrame(
            result_card,
            fg_color="#141421",
            corner_radius=8,
            border_color="#2C2C40",
            border_width=1
        )
        text_container.pack(fill="x", padx=self.PADDING_MEDIUM, pady=(0, 10))

        self.text_display = ctk.CTkLabel(
            text_container,
            text="[ Belum Ada Gestur Terdeteksi ]",
            font=ctk.CTkFont(size=17, weight="bold"),
            text_color=self.COLOR_TEXT_MUTED,
            anchor="w"
        )
        self.text_display.pack(fill="x", padx=12, pady=14)

        # Card 1 Action Toolbar
        action_toolbar = ctk.CTkFrame(result_card, fg_color="transparent")
        action_toolbar.pack(fill="x", padx=self.PADDING_MEDIUM, pady=(0, 10))

        self.copy_btn = ctk.CTkButton(
            action_toolbar,
            text="📋 Salin Teks",
            command=self._copy_text,
            height=32,
            font=ctk.CTkFont(size=12, weight="bold"),
            corner_radius=6,
            fg_color=self.COLOR_NEUTRAL,
            hover_color=self.COLOR_NEUTRAL_HOVER,
            state="disabled"
        )
        self.copy_btn.pack(side="left")

        # -------------------------------------------------------------
        # CARD 2: Huruf Terakhir & Status Deteksi
        # -------------------------------------------------------------
        status_card = ctk.CTkFrame(
            right_frame,
            fg_color=self.COLOR_CARD_BG,
            border_color=self.COLOR_CARD_BORDER,
            border_width=1,
            corner_radius=self.CORNER_RADIUS
        )
        status_card.pack(fill="x", pady=(0, 10))

        status_card_inner = ctk.CTkFrame(status_card, fg_color="transparent")
        status_card_inner.pack(fill="x", padx=self.PADDING_MEDIUM, pady=10)
        status_card_inner.grid_columnconfigure(0, weight=1)
        status_card_inner.grid_columnconfigure(1, weight=1)

        # Sub-box 1: Huruf Terakhir
        letter_box = ctk.CTkFrame(status_card_inner, fg_color="#141421", corner_radius=8)
        letter_box.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=4)

        ctk.CTkLabel(
            letter_box,
            text="HURUF TERAKHIR",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=self.COLOR_TEXT_MUTED
        ).pack(pady=(6, 0))

        self.latest_letter_label = ctk.CTkLabel(
            letter_box,
            text="-",
            font=ctk.CTkFont(size=32, weight="bold"),
            text_color=self.COLOR_TEXT_WHITE
        )
        self.latest_letter_label.pack(pady=(2, 6))

        # Sub-box 2: Durasi Hold Gesture
        hold_box = ctk.CTkFrame(status_card_inner, fg_color="#141421", corner_radius=8)
        hold_box.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=4)

        ctk.CTkLabel(
            hold_box,
            text="DURASI TAHAN GESTUR",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=self.COLOR_TEXT_MUTED
        ).pack(pady=(6, 0))

        self.gesture_status_label = ctk.CTkLabel(
            hold_box,
            text="2.5 Detik / Karakter",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=self.COLOR_TEXT_ACCENT
        )
        self.gesture_status_label.pack(pady=(8, 6))

        # -------------------------------------------------------------
        # CARD 3: Pintasan Keyboard & Tips Deteksi
        # -------------------------------------------------------------
        guide_card = ctk.CTkFrame(
            right_frame,
            fg_color=self.COLOR_CARD_BG,
            border_color=self.COLOR_CARD_BORDER,
            border_width=1,
            corner_radius=self.CORNER_RADIUS
        )
        guide_card.pack(fill="both", expand=True)

        guide_inner = ctk.CTkFrame(guide_card, fg_color="transparent")
        guide_inner.pack(fill="both", expand=True, padx=self.PADDING_MEDIUM, pady=(12, 16))

        ctk.CTkLabel(
            guide_inner,
            text="PINTASAN KEYBOARD",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=self.COLOR_TEXT_ACCENT
        ).pack(anchor="w", pady=(0, 6))

        shortcuts_list = [
            ("Space", "Mulai / Hentikan kamera"),
            ("R", "Reset teks terakumulasi"),
            ("Esc", "Keluar dari aplikasi")
        ]

        for key, desc in shortcuts_list:
            row = ctk.CTkFrame(guide_inner, fg_color="transparent")
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(
                row,
                text=f" [{key}] ",
                font=ctk.CTkFont(size=11, weight="bold"),
                fg_color="#2A2A3D",
                corner_radius=4,
                text_color=self.COLOR_TEXT_WHITE
            ).pack(side="left")
            ctk.CTkLabel(
                row,
                text=f"  {desc}",
                font=ctk.CTkFont(size=11),
                text_color=self.COLOR_TEXT_MUTED
            ).pack(side="left")

        ctk.CTkLabel(
            guide_inner,
            text="PANDUAN DETEKSI OPTIMAL",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=self.COLOR_TEXT_ACCENT
        ).pack(anchor="w", pady=(12, 4))

        tips_text = "• Pastikan pencahayaan ruangan cukup terang & merata.\n• Posisikan telapak tangan secara jelas di tengah bingkai kamera.\n• Tahan posisi gestur secara stabil selama 2.5 detik."
        ctk.CTkLabel(
            guide_inner,
            text=tips_text,
            font=ctk.CTkFont(size=11),
            text_color=self.COLOR_TEXT_MUTED,
            justify="left"
        ).pack(anchor="w")

    def _setup_footer(self):
        """Setup minimal clean footer."""
        footer = ctk.CTkFrame(self.window, fg_color="transparent")
        footer.pack(side="bottom", fill="x", padx=self.MARGIN, pady=(0, 6))

        ctk.CTkLabel(
            footer,
            text="© 2026 Ananta Raihan | Penulisan Ilmiah - Sistem Pengenalan Bahasa Isyarat SIBI",
            font=ctk.CTkFont(size=11),
            text_color="#6B7280"
        ).pack(anchor="center")

    def _update_status_badge(self, status_type: str, custom_msg: Optional[str] = None):
        """Update status pill badge with clean color scheme."""
        if status_type == "OFFLINE":
            self.status_badge.configure(
                text="⚪ STANDBY",
                fg_color="#272738",
                text_color=self.COLOR_TEXT_MUTED
            )
        elif status_type == "INITIALIZING":
            self.status_badge.configure(
                text="🟡 MENGHUBUNGKAN KAMERA...",
                fg_color="#3D3216",
                text_color="#FBBF24"
            )
        elif status_type == "LIVE":
            self.status_badge.configure(
                text="🟢 DETEKSI AKTIF",
                fg_color="#064E3B",
                text_color="#34D399"
            )
        elif status_type == "ERROR":
            msg = custom_msg if custom_msg else "🔴 TERJADI KENDALA"
            self.status_badge.configure(
                text=msg,
                fg_color="#451A1A",
                text_color="#F87171"
            )

    def _copy_text(self):
        """Copy accumulated text to system clipboard."""
        text_to_copy = self._last_logged_text
        if text_to_copy:
            self.window.clipboard_clear()
            self.window.clipboard_append(text_to_copy)
            self.copy_btn.configure(text="✓ Tersalin!", fg_color="#059669")
            self.window.after(1500, lambda: self.copy_btn.configure(text="📋 Salin Teks", fg_color=self.COLOR_NEUTRAL))

    def _show_loading_state(self):
        """Show camera initialization state."""
        self._update_status_badge("INITIALIZING")
        self.canvas.configure(
            text="Connecting to camera sensor, please wait...",
            font=ctk.CTkFont(size=13),
            text_color=self.COLOR_TEXT_MUTED
        )
        self.window.update_idletasks()
        self.start_btn.configure(
            text="⏹ HENTIKAN",
            command=self._stop_recognition,
            fg_color=self.COLOR_DANGER,
            hover_color=self.COLOR_DANGER_HOVER,
            state="disabled"
        )

    def _start_session_log(self):
        """Reset and print detection session header."""
        self._last_logged_text = ""
        self._detection_count = 0
        self._is_closing = False
        logger.info("%s", "=" * 60)
        logger.info("DETECTION SESSION STARTED")
        logger.info("%s", "=" * 60)
        logger.info("#\tTime\t\t\tLetter")
        logger.info("%s", "-" * 60)

    def _reset_idle_canvas(self):
        """Restore idle canvas and controls."""
        self._update_status_badge("OFFLINE")
        self.start_btn.configure(
            text="▶ MULAI DETEKSI",
            command=self._start_recognition,
            fg_color=self.COLOR_PRIMARY,
            hover_color=self.COLOR_PRIMARY_HOVER,
            state="normal"
        )
        self.reset_btn.configure(state="disabled")
        self.canvas.configure(
            image=None,
            fg_color="#11111B",
            text="Kamera Belum Aktif\n\nKlik tombol ▶ MULAI DETEKSI di bawah untuk memulai analisis gestur SIBI.",
            font=ctk.CTkFont(size=13),
            text_color=self.COLOR_TEXT_MUTED
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

    def _schedule_ui(self, callback, *args):
        if self._is_closing:
            return
        try:
            self.window.after(0, callback, *args)
        except tk.TclError:
            pass

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
            self._recognizer = SignLanguageRecognizer(self.config)
            self._schedule_ui(self._enable_controls)
        except (RuntimeError, OSError, ValueError) as e:
            self._schedule_ui(self._show_init_error, str(e))
            return

        self._video_loop()

    def _enable_controls(self):
        """Re-enable buttons after successful init."""
        self._update_status_badge("LIVE")
        self.start_btn.configure(state="normal")
        self.reset_btn.configure(state="normal")

    def _stop_recognition(self):
        """Stop video processing gracefully."""
        if not self._is_running:
            return

        self._is_running = False
        self._fade_canvas()
        self._finish_stop_when_thread_exited()

    def _finish_stop_when_thread_exited(self):
        """Finish cleanup after the video thread has actually exited."""
        if self._video_thread and self._video_thread.is_alive():
            self._video_thread.join(timeout=0.2)
            if self._video_thread.is_alive():
                logger.warning("Waiting for video thread to stop")
                self.window.after(200, self._finish_stop_when_thread_exited)
                return

        self._cleanup_recognizer()
        self._video_thread = None
        self._log_session_summary()
        self._reset_idle_canvas()

    def _reset_text(self):
        """Reset accumulated text (R key or button)."""
        if not self._is_running or not self._recognizer:
            return
        self._recognizer.reset_text()
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        logger.info("---\t%s\tRESET", timestamp)
        self._last_logged_text = ""
        self.text_display.configure(text="- Belum Ada Teks -", text_color=self.COLOR_TEXT_MUTED)
        self.char_count_label.configure(text="0 / 10 Karakter")
        self.latest_letter_label.configure(text="-", text_color=self.COLOR_TEXT_WHITE)
        self.copy_btn.configure(state="disabled")

    def _process_video_frame(self, frame) -> bool:
        """Process a captured frame and schedule UI updates."""
        recognizer = self._recognizer
        if not recognizer:
            return False

        processed_frame, current_text = recognizer.process_frame(frame)
        self._schedule_ui(self._log_detection, current_text)
        self._schedule_ui(self._update_frame, processed_frame)
        return True

    def _video_loop(self):
        """Main video processing loop (runs in separate thread)."""
        cap = cv2.VideoCapture(self._camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self._schedule_ui(self._show_camera_error)
            return

        try:
            while self._is_running:
                try:
                    ret, frame = cap.read()
                    if not ret:
                        self._schedule_ui(self._show_camera_error)
                        break
                    if not self._process_video_frame(frame):
                        break
                except (cv2.error, RuntimeError, ValueError, TypeError) as e:
                    self._schedule_ui(self._show_processing_error, str(e))
                    break
        finally:
            cap.release()
            if self._is_running:
                self._schedule_ui(self._stop_recognition)

    def _log_detection(self, current_text: str):
        """Log each detected letter individually to terminal and update GUI labels."""
        if not self._is_running:
            return

        # Update text display box
        if current_text:
            self.text_display.configure(text=current_text, text_color=self.COLOR_TEXT_WHITE)
            self.char_count_label.configure(text=f"{len(current_text)} / 10 Karakter")
            self.latest_letter_label.configure(text=current_text[-1], text_color=self.COLOR_TEXT_ACCENT)
            self.copy_btn.configure(state="normal")
        else:
            self.text_display.configure(text="[ Belum Ada Gestur Terdeteksi ]", text_color=self.COLOR_TEXT_MUTED)
            self.char_count_label.configure(text="0 / 10 Karakter")
            self.latest_letter_label.configure(text="-", text_color=self.COLOR_TEXT_WHITE)
            self.copy_btn.configure(state="disabled")

        if current_text == self._last_logged_text:
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
        self._current_ctk_img = ctk_img  # Keep reference to prevent GC

    def _on_closing(self):
        """Clean shutdown on window close."""
        if self._is_closing:
            return
        self._is_closing = True
        self._is_running = False
        self.start_btn.configure(state="disabled")
        self.reset_btn.configure(state="disabled")
        self.exit_btn.configure(state="disabled")
        self._destroy_when_thread_exited()

    def _destroy_when_thread_exited(self):
        """Destroy the window only after the video thread has stopped."""
        if self._video_thread and self._video_thread.is_alive():
            self._video_thread.join(timeout=0.2)
            if self._video_thread.is_alive():
                try:
                    self.window.after(200, self._destroy_when_thread_exited)
                except tk.TclError:
                    pass
                return

        self._cleanup_recognizer()
        try:
            self.window.destroy()
        except tk.TclError:
            pass

    def _fade_canvas(self):
        """Smooth fade transition effect."""
        try:
            for alpha in [0.6, 0.3, 0.0]:
                color = f"#{int(17 * alpha):02x}" * 3
                self.canvas.configure(fg_color=color)
                self.window.update()
        except (RuntimeError, tk.TclError):
            pass  # Window closing, skip gracefully

    def _build_help_window(self):
        """Create and configure the help dialog window."""
        help_window = ctk.CTkToplevel(self.window)
        help_window.title("Petunjuk Penggunaan")
        help_window.geometry("480x540")
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
            text_color=self.COLOR_TEXT_WHITE
        ).pack(pady=(0, 12))
        ctk.CTkLabel(
            content_frame,
            text=self.HELP_TEXT,
            justify="left",
            font=ctk.CTkFont(size=12),
            text_color=self.COLOR_TEXT_MUTED
        ).pack(pady=(0, 12))
        ctk.CTkLabel(
            content_frame,
            text=f"Version {self.VERSION} | © 2026 Ananta Raihan",
            font=ctk.CTkFont(size=11),
            text_color="#6B7280"
        ).pack(pady=(8, 0))
        ctk.CTkButton(
            content_frame,
            text="Mengerti",
            command=help_window.destroy,
            width=120,
            height=36,
            corner_radius=8,
            fg_color=self.COLOR_PRIMARY,
            hover_color=self.COLOR_PRIMARY_HOVER
        ).pack(pady=(12, 0))

    def _show_help(self):
        """Display help/about dialog."""
        help_window = self._build_help_window()
        self._add_help_content(help_window)

    def _show_error(self, error_type: str, message: str):
        """Display error message with retry option."""
        self._is_running = False
        self._update_status_badge("ERROR")
        if self._recognizer:
            self._recognizer.cleanup()
            self._recognizer = None
        self.canvas.configure(
            text=f"{error_type}\n\n{message}\n\nKlik START untuk mencoba kembali.",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#11111B",
            text_color="#F87171"
        )
        self.start_btn.configure(
            text="▶ START",
            command=self._start_recognition,
            fg_color=self.COLOR_PRIMARY,
            hover_color=self.COLOR_PRIMARY_HOVER,
            state="normal"
        )
        self.reset_btn.configure(state="disabled")

    def _show_camera_error(self):
        """Display camera error."""
        self._show_error("Camera Error", "Gagal menguji atau membuka modul kamera.\nHarap periksa koneksi webcam Anda.")

    def _show_processing_error(self, error_msg: str):
        """Display processing error."""
        self._show_error("Processing Error", error_msg)

    def _show_init_error(self, error_msg: str):
        """Display initialization error."""
        self._show_error("Initialization Error", f"{error_msg}\n\nHarap periksa ketersediaan file model AI.")

    def run(self):
        """Start the GUI application."""
        self.window.mainloop()


def main():
    """Entry point for GUI application."""
    app = SignLanguageGUI()
    app.run()


if __name__ == "__main__":
    main()
