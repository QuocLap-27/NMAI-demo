"""Giao diện demo trực quan (Tkinter) cho hệ thống TouchAssist.

Tích hợp trực tiếp bố cục bàn phím (keyboard_layout.py), mô hình ngôn ngữ
tiếng Việt N-gram + Telex (ngram_model.py) và bộ giải mã Bayes + Viterbi
(viterbi_decoder.py).
"""

from __future__ import annotations

import math
import random
import time
import tkinter as tk
from tkinter import ttk
from typing import List, Optional, Tuple

from keyboard_layout import KeyboardLayout
from ngram_model import TelexConverter, VietnameseLanguageModel
from viterbi_decoder import TouchEvent, ViterbiDecoder


class TouchAssistApp:
    """Ứng dụng Tkinter mô phỏng và trực quan hóa hệ thống TouchAssist."""

    SAMPLE_SENTENCES: List[Tuple[str, str]] = [
        ("Mẫu 1: Xin chào bác sĩ...", "xin chào bác sĩ tôi cần uống thuốc"),
        ("Mẫu 2: Cảm ơn bạn đã...", "cảm ơn bạn đã giúp đỡ tôi"),
        ("Mẫu 3: Hôm nay thời tiết...", "hôm nay thời tiết rất đẹp"),
    ]

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(
            "TouchAssist - Hỗ trợ gõ tiếng Việt cho người bệnh Parkinson (Bayes + Viterbi)"
        )
        self.root.geometry("1240x760")
        self.root.minsize(1220, 710)
        self.root.configure(bg="#f1f5f9")

        # Khởi tạo các thành phần lõi AI
        self.layout = KeyboardLayout(canvas_width=860, canvas_height=340)
        self.lm = VietnameseLanguageModel()
        self.decoder = ViterbiDecoder(layout=self.layout, language_model=self.lm)

        # Trạng thái phiên gõ hiện tại
        self.touch_buffer: List[TouchEvent] = []
        self.output_words: List[str] = []
        self.prev_words: List[str] = []
        self._last_actions_summary: str = "-"
        self._last_elapsed_ms: Optional[float] = None
        self._last_top_candidates: List[Tuple[str, str, float]] = []
        self._demo_after_ids: List[str] = []
        self._demo_running: bool = False
        self._rng = random.Random()

        # Biến điều khiển giao diện
        self.mode_var = tk.StringVar(value="ai")
        self.tremor_var = tk.BooleanVar(value=True)
        self.sigma_var = tk.DoubleVar(value=14.0)
        self.sigma_label_var = tk.StringVar(value="14.0 px")
        self.status_var = tk.StringVar(value="Sẵn sàng — Chạm phím để bắt đầu gõ.")
        self.buffer_info_var = tk.StringVar(value="Buffer trống (0 cú chạm)")
        self.raw_info_var = tk.StringVar(value="Ký tự thô gần nhất: -")
        self.latency_var = tk.StringVar(value="Thời gian xử lý: -")
        self.action_trace_var = tk.StringVar(value="Chuỗi hành động Viterbi: -")

        self._configure_styles()
        self._build_ui()
        self.layout.draw_on_canvas(self.canvas, show_centers=True)
        self._refresh_debug_panel()

    # ------------------------------------------------------------------
    # Xây dựng giao diện
    # ------------------------------------------------------------------

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TFrame", background="#f1f5f9")
        style.configure("Card.TFrame", background="#ffffff", relief="solid")
        style.configure(
            "Title.TLabel",
            background="#ffffff",
            foreground="#0f172a",
            font=("Segoe UI", 11, "bold"),
        )
        style.configure(
            "Sub.TLabel",
            background="#ffffff",
            foreground="#475569",
            font=("Segoe UI", 9),
        )
        style.configure(
            "Treeview",
            font=("Segoe UI", 9),
            rowheight=22,
        )
        style.configure(
            "Treeview.Heading",
            font=("Segoe UI", 9, "bold"),
        )

    def _build_ui(self) -> None:
        main_container = tk.Frame(self.root, bg="#f1f5f9", padx=12, pady=10)
        main_container.pack(fill="both", expand=True)

        # Pack cột phải trước để luôn giữ đủ chiều rộng bảng điều khiển (325px)
        right_col = tk.Frame(
            main_container,
            bg="#ffffff",
            highlightbackground="#cbd5e1",
            highlightthickness=1,
            width=325,
        )
        right_col.pack(side="right", fill="y", padx=(12, 0))
        right_col.pack_propagate(False)

        # Cột trái: Output + Debug + Bàn phím ảo
        left_col = tk.Frame(main_container, bg="#f1f5f9")
        left_col.pack(side="left", fill="both", expand=True)

        self._build_output_frame(left_col)
        self._build_debug_frame(left_col)
        self._build_keyboard_frame(left_col)
        self._build_control_panel(right_col)

    def _build_output_frame(self, parent: tk.Widget) -> None:
        frame = tk.Frame(
            parent,
            bg="#ffffff",
            highlightbackground="#cbd5e1",
            highlightthickness=1,
            padx=12,
            pady=8,
        )
        frame.pack(fill="x", pady=(0, 8))

        header = tk.Frame(frame, bg="#ffffff")
        header.pack(fill="x", pady=(0, 6))

        tk.Label(
            header,
            text="Văn bản tiếng Việt hoàn thiện (Output Text)",
            bg="#ffffff",
            fg="#0f172a",
            font=("Segoe UI", 11, "bold"),
        ).pack(side="left")

        btn_bar = tk.Frame(header, bg="#ffffff")
        btn_bar.pack(side="right")

        tk.Button(
            btn_bar,
            text="Copy",
            command=self.copy_output,
            bg="#2563eb",
            fg="#ffffff",
            activebackground="#1d4ed8",
            activeforeground="#ffffff",
            relief="flat",
            font=("Segoe UI", 9, "bold"),
            padx=12,
            pady=3,
            cursor="hand2",
        ).pack(side="left", padx=(0, 6))

        tk.Button(
            btn_bar,
            text="Xóa tất cả",
            command=self.clear_all,
            bg="#dc2626",
            fg="#ffffff",
            activebackground="#b91c1c",
            activeforeground="#ffffff",
            relief="flat",
            font=("Segoe UI", 9, "bold"),
            padx=12,
            pady=3,
            cursor="hand2",
        ).pack(side="left")

        self.output_text = tk.Text(
            frame,
            width=30,
            height=2,
            font=("Segoe UI", 18, "bold"),
            bg="#f8fafc",
            fg="#0f172a",
            relief="flat",
            highlightbackground="#cbd5e1",
            highlightthickness=1,
            wrap="word",
            padx=10,
            pady=8,
        )
        self.output_text.pack(fill="x")
        self.output_text.configure(state="disabled")

        tk.Label(
            frame,
            textvariable=self.status_var,
            bg="#ffffff",
            fg="#475569",
            font=("Segoe UI", 9, "italic"),
            anchor="w",
        ).pack(fill="x", pady=(4, 0))

    def _build_debug_frame(self, parent: tk.Widget) -> None:
        frame = tk.Frame(
            parent,
            bg="#ffffff",
            highlightbackground="#cbd5e1",
            highlightthickness=1,
            padx=12,
            pady=8,
        )
        frame.pack(fill="both", expand=True, pady=(0, 8))

        top_row = tk.Frame(frame, bg="#ffffff")
        top_row.pack(fill="x", pady=(0, 6))

        tk.Label(
            top_row,
            text="Debug & Giải thích AI (Bayes + Viterbi + N-gram)",
            bg="#ffffff",
            fg="#0f172a",
            font=("Segoe UI", 11, "bold"),
        ).pack(side="left")

        self.latency_label = tk.Label(
            top_row,
            textvariable=self.latency_var,
            bg="#ecfdf5",
            fg="#047857",
            font=("Segoe UI", 9, "bold"),
            padx=8,
            pady=2,
        )
        self.latency_label.pack(side="right")

        content = tk.Frame(frame, bg="#ffffff")
        content.pack(fill="both", expand=True)

        # Nửa trái Debug: Buffer + Ký tự thô
        left_dbg = tk.Frame(content, bg="#ffffff")
        left_dbg.pack(side="left", fill="both", expand=True, padx=(0, 10))

        tk.Label(
            left_dbg,
            textvariable=self.buffer_info_var,
            bg="#ffffff",
            fg="#1e293b",
            font=("Segoe UI", 9, "bold"),
            anchor="w",
        ).pack(fill="x")

        tk.Label(
            left_dbg,
            textvariable=self.raw_info_var,
            bg="#ffffff",
            fg="#b91c1c",
            font=("Consolas", 10, "bold"),
            anchor="w",
            justify="left",
            wraplength=470,
        ).pack(fill="x", pady=(2, 4))

        self.buffer_listbox = tk.Listbox(
            left_dbg,
            height=4,
            font=("Consolas", 9),
            bg="#f8fafc",
            fg="#334155",
            relief="flat",
            highlightbackground="#cbd5e1",
            highlightthickness=1,
        )
        self.buffer_listbox.pack(fill="both", expand=True)

        tk.Label(
            left_dbg,
            textvariable=self.action_trace_var,
            bg="#ffffff",
            fg="#0369a1",
            font=("Consolas", 9),
            anchor="w",
            justify="left",
            wraplength=470,
        ).pack(fill="x", pady=(4, 0))

        # Nửa phải Debug: Bảng Top-3 ứng viên
        right_dbg = tk.Frame(content, bg="#ffffff", width=360)
        right_dbg.pack(side="right", fill="y")
        right_dbg.pack_propagate(False)

        tk.Label(
            right_dbg,
            text="Top-3 từ ứng viên Viterbi",
            bg="#ffffff",
            fg="#1e293b",
            font=("Segoe UI", 9, "bold"),
            anchor="w",
        ).pack(fill="x", pady=(0, 4))

        columns = ("rank", "vn", "telex", "score")
        self.candidates_tree = ttk.Treeview(
            right_dbg,
            columns=columns,
            show="headings",
            height=3,
        )
        self.candidates_tree.heading("rank", text="#")
        self.candidates_tree.heading("vn", text="Tiếng Việt")
        self.candidates_tree.heading("telex", text="Telex")
        self.candidates_tree.heading("score", text="Log-Score")

        self.candidates_tree.column("rank", width=35, anchor="center")
        self.candidates_tree.column("vn", width=120, anchor="w")
        self.candidates_tree.column("telex", width=110, anchor="w")
        self.candidates_tree.column("score", width=85, anchor="e")
        self.candidates_tree.pack(fill="both", expand=True)

    def _build_keyboard_frame(self, parent: tk.Widget) -> None:
        frame = tk.Frame(
            parent,
            bg="#ffffff",
            highlightbackground="#cbd5e1",
            highlightthickness=1,
            padx=8,
            pady=8,
        )
        frame.pack(fill="x")

        legend = tk.Frame(frame, bg="#ffffff")
        legend.pack(fill="x", pady=(0, 4))

        tk.Label(
            legend,
            text="Bàn phím ảo TouchAssist (Click chuột để chạm)",
            bg="#ffffff",
            fg="#0f172a",
            font=("Segoe UI", 10, "bold"),
        ).pack(side="left")

        tk.Label(
            legend,
            text="● Đỏ: Tọa độ chạm thực tế (có nhiễu)    ● Xanh lá: Tâm phím Viterbi chọn",
            bg="#ffffff",
            fg="#475569",
            font=("Segoe UI", 9),
        ).pack(side="right")

        self.canvas = tk.Canvas(
            frame,
            width=self.layout.canvas_width,
            height=self.layout.canvas_height,
            bg="#e2e8f0",
            highlightthickness=1,
            highlightbackground="#94a3b8",
            cursor="crosshair",
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._on_canvas_click)

    def _build_control_panel(self, parent: tk.Widget) -> None:
        inner = tk.Frame(parent, bg="#ffffff", padx=14, pady=12)
        inner.pack(fill="both", expand=True)

        tk.Label(
            inner,
            text="Bảng điều khiển Demo",
            bg="#ffffff",
            fg="#0f172a",
            font=("Segoe UI", 12, "bold"),
        ).pack(anchor="w", pady=(0, 8))

        # 1. Chế độ hoạt động
        mode_box = tk.LabelFrame(
            inner,
            text="Chế độ hoạt động",
            bg="#ffffff",
            fg="#1e293b",
            font=("Segoe UI", 9, "bold"),
            padx=10,
            pady=8,
        )
        mode_box.pack(fill="x", pady=(0, 10))

        tk.Radiobutton(
            mode_box,
            text="TouchAssist AI\n(Bayes + Viterbi + N-gram)",
            variable=self.mode_var,
            value="ai",
            justify="left",
            bg="#ffffff",
            fg="#0f172a",
            activebackground="#ffffff",
            font=("Segoe UI", 9),
            command=self._refresh_debug_panel,
        ).pack(anchor="w", pady=(0, 4))

        tk.Radiobutton(
            mode_box,
            text="Baseline\n(Bàn phím thường + Lọc < 500ms)",
            variable=self.mode_var,
            value="baseline",
            justify="left",
            bg="#ffffff",
            fg="#0f172a",
            activebackground="#ffffff",
            font=("Segoe UI", 9),
            command=self._refresh_debug_panel,
        ).pack(anchor="w")

        # 2. Giả lập run tay Parkinson & Slider sigma
        tremor_box = tk.LabelFrame(
            inner,
            text="Mô phỏng run tay Parkinson",
            bg="#ffffff",
            fg="#1e293b",
            font=("Segoe UI", 9, "bold"),
            padx=10,
            pady=8,
        )
        tremor_box.pack(fill="x", pady=(0, 10))

        tk.Checkbutton(
            tremor_box,
            text="Giả lập run tay Parkinson\n(Tremor Simulation)",
            variable=self.tremor_var,
            justify="left",
            bg="#ffffff",
            fg="#0f172a",
            activebackground="#ffffff",
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w")

        tk.Label(
            tremor_box,
            text="• Nhiễu vị trí Gaussian 2D (σ px)\n• 30% chạm lặp: 50–180ms, < 2.5mm",
            bg="#ffffff",
            fg="#475569",
            font=("Segoe UI", 8),
            justify="left",
        ).pack(anchor="w", pady=(2, 6))

        sigma_header = tk.Frame(tremor_box, bg="#ffffff")
        sigma_header.pack(fill="x")
        tk.Label(
            sigma_header,
            text="Độ lệch chuẩn nhiễu σ:",
            bg="#ffffff",
            fg="#1e293b",
            font=("Segoe UI", 9),
        ).pack(side="left")
        tk.Label(
            sigma_header,
            textvariable=self.sigma_label_var,
            bg="#ffffff",
            fg="#dc2626",
            font=("Segoe UI", 9, "bold"),
        ).pack(side="right")

        tk.Scale(
            tremor_box,
            from_=5,
            to=25,
            resolution=0.5,
            orient="horizontal",
            variable=self.sigma_var,
            showvalue=False,
            bg="#ffffff",
            highlightthickness=0,
            command=self._on_sigma_changed,
        ).pack(fill="x", pady=(2, 4))

        tk.Button(
            tremor_box,
            text="Xóa dấu vết trên Canvas",
            command=self.clear_traces,
            bg="#475569",
            fg="#ffffff",
            activebackground="#334155",
            activeforeground="#ffffff",
            relief="flat",
            font=("Segoe UI", 9),
            pady=4,
            cursor="hand2",
        ).pack(fill="x", pady=(6, 0))

        # 3. Các nút gõ thử câu mẫu
        sample_box = tk.LabelFrame(
            inner,
            text="Gõ thử câu mẫu (Tự động)",
            bg="#ffffff",
            fg="#1e293b",
            font=("Segoe UI", 9, "bold"),
            padx=10,
            pady=8,
        )
        sample_box.pack(fill="x", pady=(0, 10))

        for label_text, sentence in self.SAMPLE_SENTENCES:
            tk.Button(
                sample_box,
                text=label_text,
                command=lambda s=sentence: self.run_sample_sentence(s),
                bg="#0284c7",
                fg="#ffffff",
                activebackground="#0369a1",
                activeforeground="#ffffff",
                relief="flat",
                font=("Segoe UI", 9, "bold"),
                anchor="w",
                padx=10,
                pady=5,
                cursor="hand2",
            ).pack(fill="x", pady=(0, 6))

        tk.Button(
            sample_box,
            text="Dừng mô phỏng câu mẫu",
            command=self.stop_sample_demo,
            bg="#e2e8f0",
            fg="#0f172a",
            activebackground="#cbd5e1",
            relief="flat",
            font=("Segoe UI", 9),
            pady=4,
            cursor="hand2",
        ).pack(fill="x")

        # Thống kê mô hình ngôn ngữ
        stats = self.lm.get_stats()
        stats_text = (
            f"Corpus nạp sẵn: {stats['sentences']} câu | "
            f"{stats['unique_vn_words']} từ đơn\n"
            f"Ngưỡng lọc run tay AI: < {self.decoder.tremor_time_ms:.0f}ms "
            f"& < {self.decoder.tremor_dist_mm:.1f}mm"
        )
        tk.Label(
            inner,
            text=stats_text,
            bg="#f8fafc",
            fg="#475569",
            font=("Segoe UI", 8),
            justify="left",
            padx=8,
            pady=6,
            highlightbackground="#cbd5e1",
            highlightthickness=1,
        ).pack(fill="x", side="bottom")

    # ------------------------------------------------------------------
    # Xử lý sự kiện chạm & logic giải mã
    # ------------------------------------------------------------------

    def _on_sigma_changed(self, value: str) -> None:
        self.sigma_label_var.set(f"{float(value):.1f} px")

    def _on_canvas_click(self, event: tk.Event) -> None:
        """Bắt sự kiện click chuột trên bàn phím ảo."""
        if self._demo_running:
            self.stop_sample_demo()
        timestamp_ms = time.time() * 1000.0
        self.process_touch(
            float(event.x),
            float(event.y),
            timestamp_ms,
            apply_tremor=self.tremor_var.get(),
        )

    def _clamp_to_canvas(self, x: float, y: float) -> Tuple[float, float]:
        cx = max(3.0, min(float(self.layout.canvas_width) - 3.0, x))
        cy = max(3.0, min(float(self.layout.canvas_height) - 3.0, y))
        return cx, cy

    def _append_letter_touch(
        self,
        x: float,
        y: float,
        timestamp_ms: float,
        apply_tremor: bool,
    ) -> None:
        """Thêm cú chạm chữ cái vào Buffer (có thể kèm nhiễu và chạm lặp run tay)."""
        if not self.touch_buffer:
            self.clear_traces()

        if apply_tremor:
            sigma = float(self.sigma_var.get())
            noisy_x, noisy_y = self._clamp_to_canvas(
                x + self._rng.gauss(0.0, sigma),
                y + self._rng.gauss(0.0, sigma),
            )
            primary = TouchEvent(x=noisy_x, y=noisy_y, timestamp_ms=timestamp_ms)
            self.touch_buffer.append(primary)
            self._draw_touch_dot(primary.x, primary.y)

            # Xác suất 30% sinh thêm cú chạm lặp do run tay (50-180ms, < 2.5mm)
            if self._rng.random() < 0.30:
                dup_dt = self._rng.uniform(50.0, 180.0)
                angle = self._rng.uniform(0.0, 2.0 * math.pi)
                dist_px = self._rng.uniform(0.2, 2.49) * self.layout.px_per_mm
                dup_x, dup_y = self._clamp_to_canvas(
                    primary.x + dist_px * math.cos(angle),
                    primary.y + dist_px * math.sin(angle),
                )
                duplicate = TouchEvent(
                    x=dup_x,
                    y=dup_y,
                    timestamp_ms=timestamp_ms + dup_dt,
                )
                self.touch_buffer.append(duplicate)
                self._draw_touch_dot(duplicate.x, duplicate.y)
        else:
            event = TouchEvent(x=x, y=y, timestamp_ms=timestamp_ms)
            self.touch_buffer.append(event)
            self._draw_touch_dot(event.x, event.y)

    def process_touch(
        self,
        x: float,
        y: float,
        timestamp_ms: float,
        apply_tremor: Optional[bool] = None,
    ) -> None:
        """Xử lý một sự kiện chạm tại tọa độ (x, y) và thời điểm timestamp_ms."""
        space_key = self.layout.get_key(KeyboardLayout.SPACE_KEY)
        backspace_key = self.layout.get_key(KeyboardLayout.BACKSPACE_KEY)

        if space_key.contains(x, y):
            self._handle_space_press()
            return

        if backspace_key.contains(x, y):
            self._handle_backspace_press()
            return

        use_tremor = self.tremor_var.get() if apply_tremor is None else apply_tremor
        self._append_letter_touch(x, y, timestamp_ms, apply_tremor=use_tremor)
        self._refresh_debug_panel()

    def _decode_baseline(
        self, touches: List[TouchEvent]
    ) -> Tuple[str, str, List[str], str]:
        """Giải mã Baseline: lấy phím gần nhất và lọc bỏ chạm lặp < 500ms."""
        filtered_chars: List[str] = []
        action_parts: List[str] = []
        last_accepted_ts: Optional[float] = None

        for touch in touches:
            nearest = self.layout.nearest_key(touch.x, touch.y, letters_only=True)
            if last_accepted_ts is not None and (
                0.0 <= (touch.timestamp_ms - last_accepted_ts) < 500.0
            ):
                action_parts.append(f"DROP<500ms({nearest})")
                continue
            filtered_chars.append(nearest)
            action_parts.append(f"RAW({nearest})")
            last_accepted_ts = touch.timestamp_ms

        raw_telex = "".join(filtered_chars)
        vn_word = TelexConverter.telex_to_vietnamese(raw_telex)
        return raw_telex, vn_word, filtered_chars, " ".join(action_parts)

    def _handle_space_press(self) -> None:
        """Giải mã Buffer hiện tại khi nhấn Space và nối vào Output Text."""
        if not self.touch_buffer:
            self.status_var.set("Buffer đang trống — không có từ nào để giải mã.")
            return

        mode = self.mode_var.get()
        start_time = time.perf_counter()

        if mode == "ai":
            result = self.decoder.decode(
                self.touch_buffer,
                prev_words=self.prev_words,
                top_k=3,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            chosen_word = result.vietnamese_text
            chosen_chars = [
                act.char
                for act in result.actions
                if act.action == "EMIT" and act.char is not None
            ]
            self._last_top_candidates = list(result.top_candidates)
            self._last_actions_summary = " ".join(
                f"{a.action}({a.char or '-'})" for a in result.actions
            )
            self.status_var.set(
                f"[AI] Đã giải mã '{chosen_word}' (Telex: '{result.telex_text}') "
                f"trong {elapsed_ms:.2f} ms."
            )
        else:
            raw_telex, chosen_word, chosen_chars, actions_str = self._decode_baseline(
                self.touch_buffer
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            self._last_top_candidates = (
                [(raw_telex, chosen_word, 0.0)] if raw_telex else []
            )
            self._last_actions_summary = actions_str or "-"
            self.status_var.set(
                f"[Baseline] Kết quả thô '{chosen_word}' (Telex: '{raw_telex}') "
                f"trong {elapsed_ms:.2f} ms."
            )

        self._last_elapsed_ms = elapsed_ms
        if chosen_word:
            self.output_words.append(chosen_word)
            self.prev_words.append(chosen_word.lower())
            self._render_output_text()

        self._draw_decoded_path(chosen_chars)
        self.touch_buffer.clear()
        self._refresh_debug_panel()

    def _handle_backspace_press(self) -> None:
        """Xóa 1 cú chạm cuối trong Buffer hoặc 1 từ ở Output Text nếu Buffer rỗng."""
        if self.touch_buffer:
            self.touch_buffer.pop()
            self.canvas.delete("touch_trace")
            for touch in self.touch_buffer:
                self._draw_touch_dot(touch.x, touch.y)
            self.status_var.set("Đã xóa 1 cú chạm cuối trong Buffer.")
            self._refresh_debug_panel()
            return

        if self.output_words:
            removed = self.output_words.pop()
            if self.prev_words:
                self.prev_words.pop()
            self._render_output_text()
            self.clear_traces()
            self.status_var.set(f"Buffer rỗng — đã xóa từ '{removed}' khỏi Output.")
        else:
            self.status_var.set("Output và Buffer đều đang trống.")

    # ------------------------------------------------------------------
    # Trực quan hóa Canvas & cập nhật giao diện
    # ------------------------------------------------------------------

    def _draw_touch_dot(self, x: float, y: float) -> None:
        """Vẽ chấm tròn màu đỏ tại tọa độ chạm thực tế."""
        r = 4.5
        self.canvas.create_oval(
            x - r,
            y - r,
            x + r,
            y + r,
            fill="#ef4444",
            outline="#7f1d1d",
            width=1,
            tags=("touch_trace",),
        )
        self.canvas.tag_raise("touch_trace")

    def _draw_decoded_path(self, chars: List[str]) -> None:
        """Vẽ các chấm tròn xanh lá nối nhau tại tâm các phím đã được chọn."""
        self.canvas.delete("decode_trace")
        if not chars:
            return

        points: List[Tuple[float, float]] = []
        for ch in chars:
            if ch in self.layout.keys:
                spec = self.layout.get_key(ch)
                points.append((spec.x, spec.y))

        if len(points) >= 2:
            flat_coords: List[float] = []
            for px, py in points:
                flat_coords.extend((px, py))
            self.canvas.create_line(
                *flat_coords,
                fill="#16a34a",
                width=3,
                smooth=False,
                tags=("decode_trace",),
            )

        r = 6.5
        for idx, (px, py) in enumerate(points):
            self.canvas.create_oval(
                px - r,
                py - r,
                px + r,
                py + r,
                fill="#22c55e",
                outline="#14532d",
                width=2,
                tags=("decode_trace",),
            )
            self.canvas.create_text(
                px,
                py - 14,
                text=str(idx + 1),
                fill="#15803d",
                font=("Segoe UI", 8, "bold"),
                tags=("decode_trace",),
            )

        self.canvas.tag_raise("touch_trace")
        self.canvas.tag_raise("decode_trace")

    def clear_traces(self) -> None:
        """Xóa toàn bộ chấm đỏ và đường nối xanh lá trên Canvas."""
        self.canvas.delete("touch_trace")
        self.canvas.delete("decode_trace")

    def _render_output_text(self) -> None:
        text_value = " ".join(self.output_words)
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", "end")
        self.output_text.insert("1.0", text_value)
        self.output_text.configure(state="disabled")

    def _refresh_debug_panel(self) -> None:
        count = len(self.touch_buffer)
        self.buffer_info_var.set(f"Bộ đệm hiện tại (Buffer): {count} cú chạm")

        self.buffer_listbox.delete(0, "end")
        raw_chars: List[str] = []
        for idx, touch in enumerate(self.touch_buffer):
            nearest = self.layout.nearest_key(touch.x, touch.y, letters_only=True)
            raw_chars.append(nearest)
            if idx == 0:
                dt_str = "đầu từ"
                tremor_flag = ""
            else:
                prev_t = self.touch_buffer[idx - 1]
                dt = touch.timestamp_ms - prev_t.timestamp_ms
                dist_mm = self.layout.px_to_mm(
                    self.layout.euclidean_distance(prev_t.x, prev_t.y, touch.x, touch.y)
                )
                dt_str = f"Δt={dt:5.0f}ms, d={dist_mm:4.1f}mm"
                tremor_flag = (
                    " [RUN TAY]"
                    if self.decoder.is_tremor_duplicate(prev_t, touch)
                    else ""
                )
            self.buffer_listbox.insert(
                "end",
                f"#{idx + 1}: phím≈'{nearest}' ({touch.x:5.1f}, {touch.y:5.1f}) | {dt_str}{tremor_flag}",
            )

        if raw_chars:
            raw_str = "".join(raw_chars)
            raw_vn = TelexConverter.telex_to_vietnamese(raw_str)
            self.raw_info_var.set(f"Không AI (gần nhất): '{raw_str}' -> '{raw_vn}'")
        else:
            self.raw_info_var.set("Không AI (gần nhất): -")

        self.action_trace_var.set(
            f"Hành động từ vừa giải mã: {self._last_actions_summary}"
        )

        if self._last_elapsed_ms is None:
            self.latency_var.set("Thời gian xử lý: -")
            self.latency_label.configure(bg="#f1f5f9", fg="#475569")
        elif self._last_elapsed_ms < 50.0:
            self.latency_var.set(
                f"Thời gian xử lý: {self._last_elapsed_ms:.2f} ms (< 50ms ✓)"
            )
            self.latency_label.configure(bg="#ecfdf5", fg="#047857")
        else:
            self.latency_var.set(f"Thời gian xử lý: {self._last_elapsed_ms:.2f} ms")
            self.latency_label.configure(bg="#fef2f2", fg="#b91c1c")

        for item_id in self.candidates_tree.get_children():
            self.candidates_tree.delete(item_id)
        for rank, (telex_c, vn_c, score) in enumerate(
            self._last_top_candidates[:3], start=1
        ):
            self.candidates_tree.insert(
                "",
                "end",
                values=(rank, vn_c, telex_c, f"{score:.2f}"),
            )

    # ------------------------------------------------------------------
    # Hành động tiện ích & chạy câu mẫu tự động
    # ------------------------------------------------------------------

    def copy_output(self) -> None:
        text_value = " ".join(self.output_words)
        self.root.clipboard_clear()
        self.root.clipboard_append(text_value)
        self.status_var.set(
            f"Đã copy {len(text_value)} ký tự vào clipboard!"
            if text_value
            else "Output đang trống."
        )

    def clear_all(self) -> None:
        self.stop_sample_demo()
        self.touch_buffer.clear()
        self.output_words.clear()
        self.prev_words.clear()
        self._last_top_candidates = []
        self._last_actions_summary = "-"
        self._last_elapsed_ms = None
        self._render_output_text()
        self.clear_traces()
        self._refresh_debug_panel()
        self.status_var.set("Đã xóa toàn bộ nội dung và bộ đệm.")

    def stop_sample_demo(self) -> None:
        for after_id in self._demo_after_ids:
            try:
                self.root.after_cancel(after_id)
            except tk.TclError:
                pass
        self._demo_after_ids.clear()
        if self._demo_running:
            self._demo_running = False
            self.status_var.set("Đã dừng mô phỏng gõ câu mẫu.")

    def run_sample_sentence(self, sentence: str) -> None:
        """Tự động mô phỏng gõ từng phím kèm nhiễu cho câu mẫu tiếng Việt."""
        self.clear_all()
        self._demo_running = True
        self.status_var.set(f"Đang mô phỏng gõ câu mẫu: '{sentence}'...")

        words = sentence.split()
        space_key = self.layout.get_key(KeyboardLayout.SPACE_KEY)

        # Lên lịch sự kiện (delay_ms, key_char, simulated_ts_ms)
        schedule: List[Tuple[int, str, float]] = []
        ui_delay_ms = 120
        sim_ts_ms = time.time() * 1000.0

        for word in words:
            telex_word = TelexConverter.word_to_telex(word)
            for ch in telex_word:
                schedule.append((ui_delay_ms, ch, sim_ts_ms))
                ui_delay_ms += 170
                sim_ts_ms += self._rng.uniform(250.0, 360.0)
            schedule.append((ui_delay_ms, KeyboardLayout.SPACE_KEY, sim_ts_ms))
            ui_delay_ms += 460
            sim_ts_ms += 520.0

        for idx, (delay_ms, key_char, event_ts_ms) in enumerate(schedule):
            is_last = idx == len(schedule) - 1

            def _step(
                kc: str = key_char,
                ts: float = event_ts_ms,
                finish: bool = is_last,
            ) -> None:
                if not self._demo_running:
                    return
                spec = space_key if kc == KeyboardLayout.SPACE_KEY else self.layout.get_key(kc)
                self.process_touch(
                    spec.x,
                    spec.y,
                    ts,
                    apply_tremor=self.tremor_var.get(),
                )
                if finish:
                    self._demo_running = False
                    self._demo_after_ids.clear()
                    self.status_var.set(
                        f"Hoàn tất mô phỏng câu mẫu ({self.mode_var.get().upper()}): "
                        f"'{' '.join(self.output_words)}'"
                    )

            after_id = self.root.after(delay_ms, _step)
            self._demo_after_ids.append(after_id)


def main() -> None:
    """Điểm khởi chạy ứng dụng TouchAssist GUI."""
    root = tk.Tk()
    TouchAssistApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
