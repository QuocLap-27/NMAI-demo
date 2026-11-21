"""Module định nghĩa bố cục bàn phím ảo QWERTY và mô hình xác suất không gian 2D Gaussian.

Dùng cho hệ thống TouchAssist hỗ trợ người bệnh Parkinson.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class KeySpec:
    """Thông số hình học của một phím trên Canvas."""

    char: str
    label: str
    x: float
    y: float
    width: float
    height: float
    row: int
    col: int

    @property
    def x1(self) -> float:
        """Tọa độ cạnh trái của phím."""
        return self.x - self.width / 2.0

    @property
    def y1(self) -> float:
        """Tọa độ cạnh trên của phím."""
        return self.y - self.height / 2.0

    @property
    def x2(self) -> float:
        """Tọa độ cạnh phải của phím."""
        return self.x + self.width / 2.0

    @property
    def y2(self) -> float:
        """Tọa độ cạnh dưới của phím."""
        return self.y + self.height / 2.0

    def contains(self, touch_x: float, touch_y: float) -> bool:
        """Kiểm tra điểm chạm (touch_x, touch_y) có nằm trong khung phím hay không."""
        return self.x1 <= touch_x <= self.x2 and self.y1 <= touch_y <= self.y2


class KeyboardLayout:
    """Quản lý layout bàn phím QWERTY chuẩn và phân phối không gian 2D Gaussian."""

    BACKSPACE_KEY = "BACKSPACE"
    SPACE_KEY = " "

    def __init__(
        self,
        canvas_width: int = 860,
        canvas_height: int = 340,
        key_width: float = 74.0,
        key_height: float = 66.0,
        key_gap: float = 8.0,
        sigma: Optional[float] = None,
        dpi: float = 96.0,
    ) -> None:
        if canvas_width <= 0 or canvas_height <= 0:
            raise ValueError("Kích thước canvas phải lớn hơn 0.")
        if key_width <= 0 or key_height <= 0:
            raise ValueError("Kích thước phím phải lớn hơn 0.")
        if dpi <= 0:
            raise ValueError("DPI phải lớn hơn 0.")

        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.key_width = float(key_width)
        self.key_height = float(key_height)
        self.key_gap = float(key_gap)
        self.sigma = float(sigma) if sigma is not None else 0.55 * self.key_width
        if self.sigma <= 0:
            raise ValueError("Độ lệch chuẩn sigma phải lớn hơn 0.")

        self.dpi = float(dpi)
        self.px_per_mm = self.dpi / 25.4

        self.keys: Dict[str, KeySpec] = {}
        self.char_keys: List[str] = [chr(ord("a") + i) for i in range(26)]
        self._build_qwerty_layout()

    def _build_qwerty_layout(self) -> None:
        """Khởi tạo tọa độ tâm và kích thước cho các phím QWERTY + Backspace + Space."""
        rows_spec: List[List[Tuple[str, str, float]]] = [
            [(ch, ch.upper(), self.key_width) for ch in "qwertyuiop"],
            [(ch, ch.upper(), self.key_width) for ch in "asdfghjkl"],
            [(ch, ch.upper(), self.key_width) for ch in "zxcvbnm"]
            + [(self.BACKSPACE_KEY, "⌫", 1.5 * self.key_width)],
            [(self.SPACE_KEY, "SPACE", 5.5 * self.key_width)],
        ]

        total_rows = len(rows_spec)
        total_height = total_rows * self.key_height + (total_rows - 1) * self.key_gap
        start_y = (self.canvas_height - total_height) / 2.0 + self.key_height / 2.0

        for row_idx, row_items in enumerate(rows_spec):
            row_width = (
                sum(width for _, _, width in row_items)
                + (len(row_items) - 1) * self.key_gap
            )
            cursor_x = (self.canvas_width - row_width) / 2.0
            center_y = start_y + row_idx * (self.key_height + self.key_gap)

            for col_idx, (char, label, width) in enumerate(row_items):
                center_x = cursor_x + width / 2.0
                self.keys[char] = KeySpec(
                    char=char,
                    label=label,
                    x=center_x,
                    y=center_y,
                    width=width,
                    height=self.key_height,
                    row=row_idx,
                    col=col_idx,
                )
                cursor_x += width + self.key_gap

    def mm_to_px(self, mm: float) -> float:
        """Quy đổi khoảng cách từ milimét sang pixel trên Canvas."""
        return mm * self.px_per_mm

    def px_to_mm(self, px: float) -> float:
        """Quy đổi khoảng cách từ pixel trên Canvas sang milimét."""
        return px / self.px_per_mm

    @staticmethod
    def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
        """Tính khoảng cách Euclidean giữa hai điểm (x1, y1) và (x2, y2)."""
        return math.hypot(x1 - x2, y1 - y2)

    def _normalize_key_name(self, key_char: str) -> str:
        """Chuẩn hóa định danh phím về khóa trong bảng self.keys."""
        if key_char in self.keys:
            return key_char
        lowered = key_char.lower()
        if lowered in self.keys:
            return lowered
        uppered = key_char.upper()
        if uppered in self.keys:
            return uppered
        raise KeyError(f"Phím không hợp lệ trong layout: {key_char!r}")

    def get_key(self, key_char: str) -> KeySpec:
        """Lấy thông tin hình học KeySpec của một phím."""
        normalized = self._normalize_key_name(key_char)
        return self.keys[normalized]

    def spatial_probability(
        self,
        touch_x: float,
        touch_y: float,
        key_char: str,
        sigma: Optional[float] = None,
    ) -> float:
        """Tính xác suất không gian P_s(touch | key_char) theo Gaussian 2D.

        Công thức:
            P_s(x, y | k) = (1 / (2 * pi * sigma^2)) * exp(-d^2 / (2 * sigma^2))
        với d là khoảng cách Euclidean từ (touch_x, touch_y) tới tâm phím (x_k, y_k).
        """
        s = float(sigma) if sigma is not None else self.sigma
        if s <= 0:
            raise ValueError("Độ lệch chuẩn sigma phải lớn hơn 0.")

        key = self.get_key(key_char)
        dist = self.euclidean_distance(touch_x, touch_y, key.x, key.y)
        variance = s * s
        norm_const = 1.0 / (2.0 * math.pi * variance)
        return norm_const * math.exp(-(dist * dist) / (2.0 * variance))

    def log_spatial_probability(
        self,
        touch_x: float,
        touch_y: float,
        key_char: str,
        sigma: Optional[float] = None,
    ) -> float:
        """Tính log xác suất không gian ln(P_s) để tránh tràn số dưới."""
        s = float(sigma) if sigma is not None else self.sigma
        if s <= 0:
            raise ValueError("Độ lệch chuẩn sigma phải lớn hơn 0.")

        key = self.get_key(key_char)
        dist = self.euclidean_distance(touch_x, touch_y, key.x, key.y)
        variance = s * s
        return -math.log(2.0 * math.pi * variance) - (dist * dist) / (2.0 * variance)

    def normalized_spatial_probabilities(
        self,
        touch_x: float,
        touch_y: float,
        sigma: Optional[float] = None,
    ) -> Dict[str, float]:
        """Chuẩn hóa phân phối không gian trên 26 phím chữ cái A-Z bằng log-sum-exp."""
        log_probs = {
            ch: self.log_spatial_probability(touch_x, touch_y, ch, sigma=sigma)
            for ch in self.char_keys
        }
        max_log = max(log_probs.values())
        exp_scores = {ch: math.exp(lp - max_log) for ch, lp in log_probs.items()}
        total = sum(exp_scores.values())
        return {ch: val / total for ch, val in exp_scores.items()}

    def nearest_key(
        self,
        touch_x: float,
        touch_y: float,
        letters_only: bool = True,
    ) -> str:
        """Tìm phím có tâm gần điểm chạm (touch_x, touch_y) nhất."""
        candidates = self.char_keys if letters_only else list(self.keys.keys())
        return min(
            candidates,
            key=lambda ch: self.euclidean_distance(
                touch_x, touch_y, self.keys[ch].x, self.keys[ch].y
            ),
        )

    def top_k_keys(
        self,
        touch_x: float,
        touch_y: float,
        k: int = 5,
        sigma: Optional[float] = None,
    ) -> List[Tuple[str, float]]:
        """Trả về k phím chữ cái có xác suất không gian chuẩn hóa cao nhất."""
        probs = self.normalized_spatial_probabilities(touch_x, touch_y, sigma=sigma)
        ranked = sorted(probs.items(), key=lambda item: item[1], reverse=True)
        return ranked[: max(1, k)]

    def draw_on_canvas(
        self,
        canvas: object,
        highlight_key: Optional[str] = None,
        show_centers: bool = False,
    ) -> None:
        """Vẽ toàn bộ bàn phím lên đối tượng tkinter.Canvas."""
        canvas.delete("keyboard_key")
        norm_highlight = (
            self._normalize_key_name(highlight_key)
            if highlight_key is not None
            else None
        )

        for char, key in self.keys.items():
            is_highlighted = char == norm_highlight
            if is_highlighted:
                fill = "#fde68a"
                outline = "#d97706"
            elif char in (self.SPACE_KEY, self.BACKSPACE_KEY):
                fill = "#e2e8f0"
                outline = "#94a3b8"
            else:
                fill = "#f8fafc"
                outline = "#94a3b8"

            canvas.create_rectangle(
                key.x1,
                key.y1,
                key.x2,
                key.y2,
                fill=fill,
                outline=outline,
                width=2,
                tags=("keyboard_key", f"key_{char}"),
            )
            canvas.create_text(
                key.x,
                key.y,
                text=key.label,
                font=("Segoe UI", 14, "bold"),
                fill="#0f172a",
                tags=("keyboard_key", f"label_{char}"),
            )

            if show_centers:
                r = 2.5
                canvas.create_oval(
                    key.x - r,
                    key.y - r,
                    key.x + r,
                    key.y + r,
                    fill="#ef4444",
                    outline="",
                    tags=("keyboard_key", "key_center"),
                )


if __name__ == "__main__":
    layout = KeyboardLayout()
    print(
        f"Đã khởi tạo {len(layout.keys)} phím "
        f"(1mm = {layout.px_per_mm:.2f}px, sigma = {layout.sigma:.2f}px)."
    )

    sample_keys = ["q", "a", "s", "d", "f", " ", "BACKSPACE"]
    for k in sample_keys:
        spec = layout.get_key(k)
        print(
            f"  Phím {spec.label:>5s}: tâm=({spec.x:6.1f}, {spec.y:6.1f}), "
            f"kích thước=({spec.width:.0f}x{spec.height:.0f})"
        )

    # Giả lập chạm lệch giữa phím 'a' và phím 's'
    key_a = layout.get_key("a")
    key_s = layout.get_key("s")
    touch_x = 0.65 * key_a.x + 0.35 * key_s.x
    touch_y = key_a.y + 8.0
    print(f"\nĐiểm chạm giả lập gần 'a'/'s': ({touch_x:.1f}, {touch_y:.1f})")
    print(
        f"  P_s thô ('a') = {layout.spatial_probability(touch_x, touch_y, 'a'):.6e}"
    )
    print(
        f"  P_s thô ('s') = {layout.spatial_probability(touch_x, touch_y, 's'):.6e}"
    )
    print("  Top 5 phím xác suất chuẩn hóa:", layout.top_k_keys(touch_x, touch_y, k=5))
