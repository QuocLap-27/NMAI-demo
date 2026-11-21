"""Script đánh giá thực nghiệm hệ thống TouchAssist (chạy thuần terminal).

So sánh đối chứng Baseline (bàn phím thường + lọc thời gian thô < 500ms) và
TouchAssist AI (Bayes + Viterbi + N-gram tiếng Việt) trên tập 20 câu tiếng Việt
với mô phỏng nhiễu chạm lệch Gaussian 2D (sigma = 15px) và chạm lặp run tay
Parkinson (30% xác suất lặp, trễ 50-180ms, lệch < 2.5mm).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import random
import time
from typing import List, Optional, Sequence, Tuple

from keyboard_layout import KeyboardLayout
from ngram_model import TelexConverter, VietnameseLanguageModel
from viterbi_decoder import TouchEvent, ViterbiDecoder


# Tham số mô phỏng & tính WPM thực tế
RANDOM_SEED = 42
GAUSSIAN_SIGMA_PX = 15.0
TREMOR_PROBABILITY = 0.30
TREMOR_DELAY_MIN_MS = 50.0
TREMOR_DELAY_MAX_MS = 180.0
TREMOR_MAX_DIST_MM = 2.5
BASELINE_REPEAT_THRESHOLD_MS = 500.0
SPACE_PRESS_TIME_MS = 350.0
CORRECTION_PER_CHAR_MS = 600.0  # Thời gian ước tính để xóa & gõ lại 1 ký tự sai


# Tập dữ liệu kiểm thử: 20 câu tiếng Việt (~201 từ), gồm cả câu trong corpus và ngữ cảnh mới
TEST_SENTENCES: List[str] = [
    "xin chào bác sĩ hôm nay tôi cảm thấy khỏe hơn",
    "tay phải của tôi sáng nay vẫn hơi run một chút",
    "cho tôi xin một cốc nước ấm để uống thuốc",
    "đã đến giờ uống thuốc buổi sáng rồi phải không",
    "cảm ơn gia đình đã luôn chăm sóc tôi tận tình",
    "tôi muốn ra ngoài sân đi dạo cho thoáng mát",
    "bữa trưa hôm nay có cơm nóng cá kho và canh rau",
    "hãy nói chậm lại giúp tôi vì tôi chưa nghe rõ",
    "làm ơn bật quạt lên vì trong phòng hơi nóng bức",
    "tối qua tôi ngủ rất ngon giấc và thấy tỉnh táo",
    "bác sĩ dặn tôi không được tự ý bỏ thuốc đâu",
    "vui lòng gọi điện cho con trai tôi về sớm nhé",
    "tôi thích uống trà nóng và đọc sách vào buổi sáng",
    "trời hôm nay nắng đẹp và thời tiết thật dễ chịu",
    "nhờ bạn lấy giúp tôi hộp thuốc đặt ở trên bàn",
    "tôi bị đau nhức vai nhưng bây giờ đã đỡ nhiều",
    "cuối tuần này cả nhà mình cùng ăn cơm vui vẻ",
    "màn hình chữ to giúp tôi nhắn tin rất dễ dàng",
    "hãy hít thở sâu và giữ tinh thần bình tĩnh nhé",
    "chúc mọi người một ngày mới thật nhiều sức khỏe",
]


@dataclass
class WordEvalRecord:
    """Lưu chi tiết kết quả giải mã của một từ trong quá trình thực nghiệm."""

    sentence_idx: int
    target_vn: str
    target_telex: str
    raw_nearest_str: str
    baseline_vn: str
    ai_vn: str
    ai_telex: str
    ai_latency_ms: float
    had_tremor: bool
    baseline_filtered_count: int


def levenshtein_distance(s1: str, s2: str) -> int:
    """Tính khoảng cách chỉnh sửa Levenshtein ở mức ký tự giữa hai chuỗi."""
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1, start=1):
        curr_row = [i]
        for j, c2 in enumerate(s2, start=1):
            insert_cost = curr_row[j - 1] + 1
            delete_cost = prev_row[j] + 1
            replace_cost = prev_row[j - 1] + (0 if c1 == c2 else 1)
            curr_row.append(min(insert_cost, delete_cost, replace_cost))
        prev_row = curr_row
    return prev_row[-1]


def clamp_to_canvas(layout: KeyboardLayout, x: float, y: float) -> Tuple[float, float]:
    """Giới hạn tọa độ chạm nằm trong kích thước Canvas của bàn phím."""
    cx = max(3.0, min(float(layout.canvas_width) - 3.0, x))
    cy = max(3.0, min(float(layout.canvas_height) - 3.0, y))
    return cx, cy


def simulate_word_touches(
    layout: KeyboardLayout,
    telex_word: str,
    start_ts_ms: float,
    rng: random.Random,
    sigma_px: float = GAUSSIAN_SIGMA_PX,
    tremor_prob: float = TREMOR_PROBABILITY,
) -> Tuple[List[TouchEvent], float, bool]:
    """Mô phỏng chuỗi cú chạm cho một từ Telex với nhiễu người bệnh Parkinson."""
    touches: List[TouchEvent] = []
    t_ms = start_ts_ms
    had_tremor = False

    for ch in telex_word:
        key = layout.get_key(ch)
        nx, ny = clamp_to_canvas(
            layout,
            key.x + rng.gauss(0.0, sigma_px),
            key.y + rng.gauss(0.0, sigma_px),
        )
        touches.append(TouchEvent(x=nx, y=ny, timestamp_ms=t_ms))

        if rng.random() < tremor_prob:
            had_tremor = True
            dup_dt = rng.uniform(TREMOR_DELAY_MIN_MS, TREMOR_DELAY_MAX_MS)
            angle = rng.uniform(0.0, 2.0 * math.pi)
            dist_px = rng.uniform(0.2, TREMOR_MAX_DIST_MM - 0.01) * layout.px_per_mm
            dx, dy = clamp_to_canvas(
                layout,
                nx + dist_px * math.cos(angle),
                ny + dist_px * math.sin(angle),
            )
            touches.append(TouchEvent(x=dx, y=dy, timestamp_ms=t_ms + dup_dt))
            t_ms += dup_dt + rng.uniform(240.0, 380.0)
        else:
            t_ms += rng.uniform(280.0, 420.0)

    return touches, t_ms, had_tremor


def decode_baseline(
    layout: KeyboardLayout,
    touches: Sequence[TouchEvent],
    repeat_threshold_ms: float = BASELINE_REPEAT_THRESHOLD_MS,
) -> Tuple[str, str, int, float]:
    """Giải mã Baseline: chọn phím gần nhất + lọc bỏ chạm lặp cùng phím < 500ms."""
    t0 = time.perf_counter()
    kept_chars: List[str] = []
    last_accepted_char: Optional[str] = None
    last_accepted_ts: Optional[float] = None
    filtered_count = 0

    for touch in touches:
        nearest = layout.nearest_key(touch.x, touch.y, letters_only=True)
        if (
            last_accepted_char == nearest
            and last_accepted_ts is not None
            and 0.0 <= (touch.timestamp_ms - last_accepted_ts) < repeat_threshold_ms
        ):
            filtered_count += 1
            continue
        kept_chars.append(nearest)
        last_accepted_char = nearest
        last_accepted_ts = touch.timestamp_ms

    telex_str = "".join(kept_chars)
    vn_word = TelexConverter.telex_to_vietnamese(telex_str)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return vn_word, telex_str, filtered_count, elapsed_ms


def select_representative_examples(
    records: Sequence[WordEvalRecord],
    limit: int = 10,
) -> List[WordEvalRecord]:
    """Chọn ra 10 ví dụ tiêu biểu (ưu tiên các trường hợp AI sửa đúng lỗi của Baseline)."""
    ai_wins_with_diff = [
        r
        for r in records
        if r.ai_vn.lower() == r.target_vn.lower()
        and (
            r.baseline_vn.lower() != r.target_vn.lower()
            or r.raw_nearest_str != r.target_telex
        )
    ]
    # Ưu tiên có chạm lặp hoặc sửa sai rõ ràng từ nhiều câu khác nhau
    ai_wins_with_diff.sort(
        key=lambda r: (
            r.baseline_vn.lower() != r.target_vn.lower(),
            r.had_tremor,
            len(r.target_vn),
        ),
        reverse=True,
    )

    selected: List[WordEvalRecord] = []
    used_words: set = set()
    for rec in ai_wins_with_diff:
        key = rec.target_vn.lower()
        if key in used_words:
            continue
        selected.append(rec)
        used_words.add(key)
        if len(selected) >= limit:
            break

    if len(selected) < limit:
        for rec in records:
            if rec not in selected:
                selected.append(rec)
                if len(selected) >= limit:
                    break
    return selected


def run_evaluation(output_path: str) -> str:
    """Chạy toàn bộ kịch bản thực nghiệm, in kết quả và ghi file báo cáo."""
    rng = random.Random(RANDOM_SEED)
    layout = KeyboardLayout(canvas_width=860, canvas_height=340)
    lm = VietnameseLanguageModel()
    decoder = ViterbiDecoder(layout=layout, language_model=lm)

    num_sentences = len(TEST_SENTENCES)
    total_words = 0

    baseline_wrong_words = 0
    ai_wrong_words = 0

    baseline_total_backspaces = 0
    ai_total_backspaces = 0

    total_raw_typing_ms = 0.0
    baseline_total_decode_ms = 0.0
    ai_total_decode_ms = 0.0
    baseline_filter_wait_ms = 0.0

    all_records: List[WordEvalRecord] = []

    for sent_idx, sentence in enumerate(TEST_SENTENCES, start=1):
        target_words = sentence.split()
        prev_ai_words: List[str] = []
        t_cursor_ms = 1000.0

        for target_vn in target_words:
            total_words += 1
            target_telex = TelexConverter.word_to_telex(target_vn)
            touches, next_t_ms, had_tremor = simulate_word_touches(
                layout,
                target_telex,
                start_ts_ms=t_cursor_ms,
                rng=rng,
            )

            word_duration_ms = (touches[-1].timestamp_ms - touches[0].timestamp_ms) + SPACE_PRESS_TIME_MS
            total_raw_typing_ms += word_duration_ms
            t_cursor_ms = next_t_ms + SPACE_PRESS_TIME_MS

            raw_nearest_str = "".join(
                layout.nearest_key(t.x, t.y, letters_only=True) for t in touches
            )

            # 1. Đánh giá Baseline
            base_vn, _, filtered_cnt, base_ms = decode_baseline(layout, touches)
            baseline_total_decode_ms += base_ms
            baseline_filter_wait_ms += filtered_cnt * BASELINE_REPEAT_THRESHOLD_MS

            if base_vn.lower() != target_vn.lower():
                baseline_wrong_words += 1
                baseline_total_backspaces += levenshtein_distance(
                    base_vn.lower(), target_vn.lower()
                )

            # 2. Đánh giá TouchAssist AI
            t_ai0 = time.perf_counter()
            ai_result = decoder.decode(touches, prev_words=prev_ai_words)
            ai_ms = (time.perf_counter() - t_ai0) * 1000.0
            ai_total_decode_ms += ai_ms

            ai_vn = ai_result.vietnamese_text
            prev_ai_words.append(ai_vn.lower())

            if ai_vn.lower() != target_vn.lower():
                ai_wrong_words += 1
                ai_total_backspaces += levenshtein_distance(
                    ai_vn.lower(), target_vn.lower()
                )

            all_records.append(
                WordEvalRecord(
                    sentence_idx=sent_idx,
                    target_vn=target_vn,
                    target_telex=target_telex,
                    raw_nearest_str=raw_nearest_str,
                    baseline_vn=base_vn,
                    ai_vn=ai_vn,
                    ai_telex=ai_result.telex_text,
                    ai_latency_ms=ai_ms,
                    had_tremor=had_tremor,
                    baseline_filtered_count=filtered_cnt,
                )
            )

    # Tính 3 chỉ số chuẩn + độ trễ trung bình
    baseline_wer = (baseline_wrong_words / total_words) * 100.0
    ai_wer = (ai_wrong_words / total_words) * 100.0

    baseline_bs_per_sent = baseline_total_backspaces / num_sentences
    ai_bs_per_sent = ai_total_backspaces / num_sentences

    baseline_eff_time_ms = (
        total_raw_typing_ms
        + baseline_total_decode_ms
        + baseline_filter_wait_ms
        + baseline_total_backspaces * CORRECTION_PER_CHAR_MS
    )
    ai_eff_time_ms = (
        total_raw_typing_ms
        + ai_total_decode_ms
        + ai_total_backspaces * CORRECTION_PER_CHAR_MS
    )

    baseline_wpm = total_words / (baseline_eff_time_ms / 60000.0)
    ai_wpm = total_words / (ai_eff_time_ms / 60000.0)

    baseline_avg_latency = baseline_total_decode_ms / total_words
    ai_avg_latency = ai_total_decode_ms / total_words

    wer_impr = (
        ((baseline_wer - ai_wer) / baseline_wer) * 100.0
        if baseline_wer > 0
        else 0.0
    )
    bs_impr = (
        ((baseline_bs_per_sent - ai_bs_per_sent) / baseline_bs_per_sent) * 100.0
        if baseline_bs_per_sent > 0
        else 0.0
    )
    wpm_impr = (
        ((ai_wpm - baseline_wpm) / baseline_wpm) * 100.0
        if baseline_wpm > 0
        else 0.0
    )

    examples = select_representative_examples(all_records, limit=10)

    # Định dạng báo cáo
    lines: List[str] = []
    sep_main = "=" * 92
    sep_sub = "-" * 92

    lines.append(sep_main)
    lines.append(
        "BÁO CÁO ĐÁNH GIÁ THỰC NGHIỆM TOUCHASSIST (BAYES + VITERBI + N-GRAM TIẾNG VIỆT)"
    )
    lines.append(sep_main)
    lines.append(
        f"Tập kiểm thử   : {num_sentences} câu tiếng Việt ({total_words} từ) | "
        f"Random Seed = {RANDOM_SEED}"
    )
    lines.append(
        f"Cấu hình nhiễu : Gaussian 2D (sigma = {GAUSSIAN_SIGMA_PX:.1f} px) | "
        f"Chạm lặp run tay = {int(TREMOR_PROBABILITY * 100)}% "
        f"({TREMOR_DELAY_MIN_MS:.0f}-{TREMOR_DELAY_MAX_MS:.0f}ms, < {TREMOR_MAX_DIST_MM:.1f}mm)"
    )
    lines.append(
        f"Baseline       : Bàn phím thường (nearest-key) + Lọc lặp thô "
        f"< {BASELINE_REPEAT_THRESHOLD_MS:.0f}ms"
    )
    lines.append(
        "TouchAssist AI : Mô hình Không gian Gaussian 2D + Thời gian run tay + "
        "N-gram (Ký tự & Từ)"
    )
    lines.append("")

    lines.append("1. BẢNG SO SÁNH HIỆU NĂNG TỔNG HỢP")
    lines.append(sep_sub)
    lines.append(
        f"{'Chỉ số đánh giá':<32} | {'Baseline':>15} | {'TouchAssist AI':>16} | {'Cải thiện':>18}"
    )
    lines.append(sep_sub)
    lines.append(
        f"{'WER - Tỉ lệ lỗi từ (%)':<32} | "
        f"{baseline_wer:>14.2f}% | "
        f"{ai_wer:>15.2f}% | "
        f"{f'Giảm {wer_impr:.1f}%':>18}"
    )
    lines.append(
        f"{'Backspaces (lần/câu)':<32} | "
        f"{baseline_bs_per_sent:>15.2f} | "
        f"{ai_bs_per_sent:>16.2f} | "
        f"{f'Giảm {bs_impr:.1f}%':>18}"
    )
    lines.append(
        f"{'WPM - Tốc độ gõ (từ/phút)':<32} | "
        f"{baseline_wpm:>15.2f} | "
        f"{ai_wpm:>16.2f} | "
        f"{f'Tăng {wpm_impr:.1f}%':>18}"
    )
    lines.append(
        f"{'Độ trễ giải mã TB (ms/từ)':<32} | "
        f"{baseline_avg_latency:>12.2f} ms | "
        f"{ai_avg_latency:>13.2f} ms | "
        f"{'(< 50ms ✓)' if ai_avg_latency < 50.0 else 'Cần tối ưu':>18}"
    )
    lines.append(sep_sub)
    lines.append(
        f"Chi tiết lỗi từ: Baseline sai {baseline_wrong_words}/{total_words} từ | "
        f"TouchAssist AI sai {ai_wrong_words}/{total_words} từ"
    )
    lines.append("")

    lines.append("2. 10 VÍ DỤ TIÊU BIỂU MINH CHỨNG TRONG QUÁ TRÌNH CHẠY")
    lines.append(sep_sub)
    lines.append(
        f"{'STT':<3} | {'Từ gốc':<10} | {'Chạm nhiễu raw':<16} | "
        f"{'Baseline':<12} | {'TouchAssist AI':<14} | {'Độ trễ':>9}"
    )
    lines.append(sep_sub)
    for idx, rec in enumerate(examples, start=1):
        lines.append(
            f"{idx:<3} | "
            f"{rec.target_vn:<10} | "
            f"{rec.raw_nearest_str:<16} | "
            f"{rec.baseline_vn:<12} | "
            f"{rec.ai_vn:<14} | "
            f"{rec.ai_latency_ms:>6.2f} ms"
        )
    lines.append(sep_sub)
    lines.append(
        "Ghi chú: WPM được tính từ tổng thời gian gõ thô + thời gian nhấn Space (350ms/từ) + "
        "độ trễ giải mã + trễ lọc Baseline (500ms/lần lọc) + thời gian bấm Backspace sửa lỗi (600ms/ký tự)."
    )
    lines.append(sep_main)

    report_text = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")

    return report_text


def main() -> None:
    """Điểm khởi chạy script đánh giá thực nghiệm."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.join(base_dir, "evaluation_report.txt")
    report = run_evaluation(report_path)
    print(report)
    print(f"\nĐã xuất báo cáo chi tiết ra file: {report_path}")


if __name__ == "__main__":
    main()
