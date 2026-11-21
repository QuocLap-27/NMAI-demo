"""Module giải mã cú chạm Bayes kết hợp Quy hoạch động Viterbi cho tiếng Việt.

Hỗ trợ lọc nhiễu chạm lặp do run tay Parkinson dựa trên xác suất không gian
(Gaussian 2D), xác suất thời gian (temporal tremor model) và mô hình ngôn ngữ
N-gram tiếng Việt (Telex + mức từ).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple, Union

from keyboard_layout import KeyboardLayout
from ngram_model import TelexConverter, VietnameseLanguageModel


@dataclass(frozen=True)
class TouchEvent:
    """Biểu diễn một sự kiện chạm trên màn hình cảm ứng / Canvas."""

    x: float
    y: float
    timestamp_ms: float


@dataclass(frozen=True)
class StepAction:
    """Hành động tối ưu được giải mã tại một bước chạm."""

    touch_index: int
    action: str  # "EMIT" hoặc "SKIP"
    char: Optional[str]
    cumulative_log_prob: float


@dataclass
class DecodeResult:
    """Kết quả giải mã đầy đủ cho một chuỗi cú chạm trong buffer."""

    telex_text: str
    vietnamese_text: str
    log_score: float
    actions: List[StepAction]
    top_candidates: List[Tuple[str, str, float]]


TouchInput = Union[TouchEvent, Tuple[float, float, float]]


class ViterbiDecoder:
    """Bộ giải mã cú chạm Bayes sử dụng quy hoạch động Viterbi."""

    def __init__(
        self,
        layout: KeyboardLayout,
        language_model: VietnameseLanguageModel,
        tremor_time_ms: float = 200.0,
        tremor_dist_mm: float = 3.0,
        spatial_weight: float = 1.0,
        temporal_weight: float = 1.0,
        lm_weight: float = 1.0,
        word_lm_weight: float = 0.8,
        known_word_bonus: float = 2.0,
    ) -> None:
        if tremor_time_ms <= 0 or tremor_dist_mm <= 0:
            raise ValueError("Ngưỡng thời gian và khoảng cách run tay phải > 0.")

        self.layout = layout
        self.lm = language_model
        self.tremor_time_ms = float(tremor_time_ms)
        self.tremor_dist_mm = float(tremor_dist_mm)
        self.spatial_weight = float(spatial_weight)
        self.temporal_weight = float(temporal_weight)
        self.lm_weight = float(lm_weight)
        self.word_lm_weight = float(word_lm_weight)
        self.known_word_bonus = float(known_word_bonus)

        self.char_keys: List[str] = list(self.layout.char_keys)
        self.num_chars: int = len(self.char_keys)
        self.bos_state_idx: int = self.num_chars  # Trạng thái 26 đại diện cho <BOS>
        self.num_states: int = self.num_chars + 1

        # Trạng thái lưu kết quả giải mã gần nhất phục vụ traceback()
        self.last_dp_matrix: List[List[float]] = []
        self.last_backpointer: List[List[Optional[Tuple[int, str, Optional[str]]]]] = []
        self.last_best_state: Optional[int] = None
        self.last_touches: List[TouchEvent] = []

    @staticmethod
    def _coerce_touches(touches: Sequence[TouchInput]) -> List[TouchEvent]:
        """Chuẩn hóa danh sách input (tuple hoặc TouchEvent) thành List[TouchEvent]."""
        normalized: List[TouchEvent] = []
        for item in touches:
            if isinstance(item, TouchEvent):
                normalized.append(item)
            else:
                x, y, ts = item
                normalized.append(TouchEvent(float(x), float(y), float(ts)))
        return normalized

    def is_tremor_duplicate(
        self,
        prev_touch: Optional[TouchEvent],
        curr_touch: TouchEvent,
    ) -> bool:
        """Kiểm tra cặp chạm liên tiếp có thỏa điều kiện chạm lặp run tay hay không."""
        if prev_touch is None:
            return False
        dt = curr_touch.timestamp_ms - prev_touch.timestamp_ms
        dist_px = self.layout.euclidean_distance(
            prev_touch.x, prev_touch.y, curr_touch.x, curr_touch.y
        )
        tremor_dist_px = self.layout.mm_to_px(self.tremor_dist_mm)
        return 0.0 <= dt < self.tremor_time_ms and dist_px < tremor_dist_px

    def temporal_probability(
        self,
        prev_touch: Optional[TouchEvent],
        curr_touch: TouchEvent,
    ) -> Dict[str, float]:
        """Tính xác suất thời gian P_t cho 2 trạng thái 'EMIT' và 'SKIP'.

        Nếu khoảng cách thời gian < 200ms và khoảng cách không gian < 3mm
        (quy đổi sang pixel trên layout), đánh trọng số cao cho trạng thái 'SKIP'
        nhằm loại bỏ cú chạm lặp do run tay Parkinson.
        """
        if prev_touch is None:
            return {"EMIT": 0.99, "SKIP": 0.01}

        dt = curr_touch.timestamp_ms - prev_touch.timestamp_ms
        dist_px = self.layout.euclidean_distance(
            prev_touch.x, prev_touch.y, curr_touch.x, curr_touch.y
        )
        tremor_dist_px = self.layout.mm_to_px(self.tremor_dist_mm)

        if 0.0 <= dt < self.tremor_time_ms and dist_px < tremor_dist_px:
            time_closeness = 1.0 - (dt / self.tremor_time_ms)
            dist_closeness = 1.0 - (dist_px / tremor_dist_px)
            p_skip = 0.88 + 0.09 * (0.5 * time_closeness + 0.5 * dist_closeness)
            p_emit = 1.0 - p_skip
            return {"EMIT": p_emit, "SKIP": p_skip}

        return {"EMIT": 0.995, "SKIP": 0.005}

    @staticmethod
    def _extract_history(
        backpointer: List[List[Optional[Tuple[int, str, Optional[str]]]]],
        step_idx: int,
        state_idx: int,
        max_len: int = 2,
    ) -> List[str]:
        """Truy ngược nhanh tối đa max_len ký tự đã EMIT trước đó tại ô (step_idx, state_idx)."""
        if step_idx < 0 or state_idx < 0:
            return []

        chars_rev: List[str] = []
        cur_step = step_idx
        cur_state = state_idx

        while cur_step >= 0 and cur_state >= 0 and len(chars_rev) < max_len:
            entry = backpointer[cur_step][cur_state]
            if entry is None:
                break
            prev_state, action, emitted_char = entry
            if action == "EMIT" and emitted_char is not None:
                chars_rev.append(emitted_char)
            cur_state = prev_state
            cur_step -= 1

        chars_rev.reverse()
        return chars_rev

    def decode(
        self,
        touches: Sequence[TouchInput],
        prev_words: Optional[Sequence[str]] = None,
        top_k: int = 3,
    ) -> DecodeResult:
        """Giải mã danh sách cú chạm thành từ tiếng Việt tối ưu bằng Viterbi.

        Tối ưu log-score:
            log Score(C | I) = sum(log P_s + log P_t + log P_LM)
        """
        events = self._coerce_touches(touches)
        n = len(events)

        self.last_touches = events
        self.last_dp_matrix = []
        self.last_backpointer = []
        self.last_best_state = None

        if n == 0:
            return DecodeResult(
                telex_text="",
                vietnamese_text="",
                log_score=0.0,
                actions=[],
                top_candidates=[],
            )

        neg_inf = -math.inf
        dp: List[List[float]] = [
            [neg_inf] * self.num_states for _ in range(n)
        ]
        backpointer: List[List[Optional[Tuple[int, str, Optional[str]]]]] = [
            [None] * self.num_states for _ in range(n)
        ]

        log_skip_spatial_bg = math.log(1.0 / self.num_chars)

        # Bước i = 0 (cú chạm đầu tiên trong buffer)
        first_touch = events[0]
        p_t0 = self.temporal_probability(None, first_touch)
        log_pt0_emit = self.temporal_weight * math.log(p_t0["EMIT"])
        log_pt0_skip = self.temporal_weight * math.log(p_t0["SKIP"])
        p_s0 = self.layout.normalized_spatial_probabilities(first_touch.x, first_touch.y)

        for c_idx, ch in enumerate(self.char_keys):
            log_ps = self.spatial_weight * math.log(max(p_s0[ch], 1e-300))
            log_plm = self.lm_weight * self.lm.log_char_lm_prob([], ch)
            dp[0][c_idx] = log_ps + log_pt0_emit + log_plm
            backpointer[0][c_idx] = (-1, "EMIT", ch)

        dp[0][self.bos_state_idx] = (
            self.spatial_weight * log_skip_spatial_bg + log_pt0_skip
        )
        backpointer[0][self.bos_state_idx] = (-1, "SKIP", None)

        # Các bước i = 1 .. n - 1
        for i in range(1, n):
            prev_touch = events[i - 1]
            curr_touch = events[i]

            p_t = self.temporal_probability(prev_touch, curr_touch)
            log_pt_emit = self.temporal_weight * math.log(p_t["EMIT"])
            log_pt_skip = self.temporal_weight * math.log(p_t["SKIP"])

            p_s = self.layout.normalized_spatial_probabilities(
                curr_touch.x, curr_touch.y
            )
            log_ps_map = {
                ch: self.spatial_weight * math.log(max(p_s[ch], 1e-300))
                for ch in self.char_keys
            }

            for prev_j in range(self.num_states):
                prev_score = dp[i - 1][prev_j]
                if prev_score == neg_inf:
                    continue

                hist = self._extract_history(backpointer, i - 1, prev_j, max_len=2)

                # Hành động 1: EMIT (chọn ký tự c)
                for c_idx, ch in enumerate(self.char_keys):
                    log_plm = self.lm_weight * self.lm.log_char_lm_prob(hist, ch)
                    cand_score = prev_score + log_ps_map[ch] + log_pt_emit + log_plm
                    if cand_score > dp[i][c_idx]:
                        dp[i][c_idx] = cand_score
                        backpointer[i][c_idx] = (prev_j, "EMIT", ch)

                # Hành động 2: SKIP (bỏ qua cú chạm nhiễu do run tay)
                skip_score = (
                    prev_score + self.spatial_weight * log_skip_spatial_bg + log_pt_skip
                )
                if skip_score > dp[i][prev_j]:
                    dp[i][prev_j] = skip_score
                    backpointer[i][prev_j] = (prev_j, "SKIP", None)

        self.last_dp_matrix = dp
        self.last_backpointer = backpointer

        # Kết thúc từ: cộng xác suất kết thúc <EOS>, bonus từ vựng và Word-level N-gram
        context_words = list(prev_words) if prev_words else []
        scored_candidates: List[Tuple[int, str, str, float]] = []

        for state_idx in range(self.num_states):
            base_score = dp[n - 1][state_idx]
            if base_score == neg_inf:
                continue

            hist = self._extract_history(backpointer, n - 1, state_idx, max_len=2)
            eos_log_prob = self.lm_weight * self.lm.log_char_lm_prob(hist, self.lm.EOS)
            total_score = base_score + eos_log_prob

            self.last_best_state = state_idx
            telex_cand, vn_cand, _ = self.traceback(end_state_idx=state_idx)

            if telex_cand:
                if self.lm.is_known_telex_word(telex_cand):
                    total_score += self.known_word_bonus
                total_score += self.word_lm_weight * self.lm.log_word_lm_prob(
                    context_words, vn_cand
                )

            scored_candidates.append((state_idx, telex_cand, vn_cand, total_score))

        if not scored_candidates:
            return DecodeResult(
                telex_text="",
                vietnamese_text="",
                log_score=neg_inf,
                actions=[],
                top_candidates=[],
            )

        scored_candidates.sort(key=lambda item: item[3], reverse=True)
        best_state_idx, best_telex, best_vn, best_score = scored_candidates[0]
        self.last_best_state = best_state_idx

        _, _, best_actions = self.traceback(end_state_idx=best_state_idx)
        top_list = [
            (telex_c, vn_c, score)
            for _, telex_c, vn_c, score in scored_candidates[: max(1, top_k)]
        ]

        return DecodeResult(
            telex_text=best_telex,
            vietnamese_text=best_vn,
            log_score=best_score,
            actions=best_actions,
            top_candidates=top_list,
        )

    def traceback(
        self,
        end_state_idx: Optional[int] = None,
    ) -> Tuple[str, str, List[StepAction]]:
        """Truy vết mảng Backpointer để trả về (chuỗi_telex, tiếng_việt, chuỗi_hành_động)."""
        if not self.last_backpointer:
            return "", "", []

        n = len(self.last_backpointer)
        curr_state = (
            self.last_best_state if end_state_idx is None else end_state_idx
        )
        if curr_state is None or not (0 <= curr_state < self.num_states):
            raise ValueError(f"Trạng thái kết thúc không hợp lệ: {curr_state}")

        emitted_chars_rev: List[str] = []
        actions_rev: List[StepAction] = []

        for step_idx in range(n - 1, -1, -1):
            entry = self.last_backpointer[step_idx][curr_state]
            if entry is None:
                break
            prev_state, action, emitted_char = entry
            actions_rev.append(
                StepAction(
                    touch_index=step_idx,
                    action=action,
                    char=emitted_char,
                    cumulative_log_prob=self.last_dp_matrix[step_idx][curr_state],
                )
            )
            if action == "EMIT" and emitted_char is not None:
                emitted_chars_rev.append(emitted_char)
            curr_state = prev_state

        emitted_chars_rev.reverse()
        actions_rev.reverse()

        telex_text = "".join(emitted_chars_rev)
        vietnamese_text = TelexConverter.telex_to_vietnamese(telex_text)
        return telex_text, vietnamese_text, actions_rev


def _simulate_parkinson_touches(
    layout: KeyboardLayout,
    telex_word: str,
    seed: int = 42,
    spatial_jitter_px: float = 12.0,
    tremor_indices: Optional[Sequence[int]] = None,
) -> List[TouchEvent]:
    """Hàm phụ trợ mô phỏng chuỗi cú chạm có nhiễu vị trí và chạm lặp run tay."""
    rng = random.Random(seed)
    tremor_set = set(tremor_indices or [])
    touches: List[TouchEvent] = []
    t_ms = 1000.0

    for idx, ch in enumerate(telex_word):
        key = layout.get_key(ch)
        x = key.x + rng.gauss(0.0, spatial_jitter_px)
        y = key.y + rng.gauss(0.0, spatial_jitter_px)
        touches.append(TouchEvent(x=x, y=y, timestamp_ms=t_ms))

        # Mô phỏng chạm lặp run tay (< 200ms, < 3mm)
        if idx in tremor_set:
            tremor_dt = rng.uniform(55.0, 145.0)
            max_px = layout.mm_to_px(2.4)
            dx = rng.uniform(-max_px * 0.5, max_px * 0.5)
            dy = rng.uniform(-max_px * 0.5, max_px * 0.5)
            touches.append(
                TouchEvent(x=x + dx, y=y + dy, timestamp_ms=t_ms + tremor_dt)
            )

        t_ms += rng.uniform(260.0, 420.0)

    return touches


if __name__ == "__main__":
    kb_layout = KeyboardLayout()
    lang_model = VietnameseLanguageModel()
    decoder = ViterbiDecoder(layout=kb_layout, language_model=lang_model)

    demo_cases = [
        ("xin", "xin", [], []),
        ("chào", "chaof", [2], ["xin"]),
        ("thuốc", "thuoocs", [1, 4], ["uống"]),
        ("nước", "nuowcs", [2], ["uống"]),
        ("việt", "vieetj", [3], ["tiếng"]),
        ("cảm", "camr", [1], []),
        ("ơn", "own", [0], ["cảm"]),
        ("được", "dduowcj", [3], []),
    ]

    print("=== DEMO GIẢI MÃ CÚ CHẠM BAYES + VITERBI (TOUCHASSIST) ===")
    for idx, (target_vn, target_telex, tremors, prev_ctx) in enumerate(demo_cases):
        simulated = _simulate_parkinson_touches(
            kb_layout,
            target_telex,
            seed=100 + idx,
            spatial_jitter_px=14.0,
            tremor_indices=tremors,
        )
        raw_nearest = "".join(
            kb_layout.nearest_key(t.x, t.y) for t in simulated
        )
        result = decoder.decode(simulated, prev_words=prev_ctx)
        action_summary = " ".join(
            f"{a.action}({a.char or '-'})" for a in result.actions
        )
        print(
            f"\nMục tiêu: '{target_vn}' (Telex: '{target_telex}', "
            f"ngữ cảnh: {prev_ctx})"
        )
        print(f"  Phím gần nhất thô : {raw_nearest}")
        print(
            f"  Viterbi giải mã   : '{result.vietnamese_text}' "
            f"(Telex: '{result.telex_text}', score={result.log_score:.2f})"
        )
        print(f"  Hành động DP      : {action_summary}")
