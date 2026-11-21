"""Module mô hình ngôn ngữ N-gram tiếng Việt và chuyển đổi Telex.

Hỗ trợ N-gram (Bigram/Trigram) ở cả mức ký tự Telex và mức từ tiếng Việt với
làm mượt Laplace / Add-k smoothing cho hệ thống TouchAssist.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# Bảng ánh xạ ký tự tiếng Việt có dấu -> (ký tự gốc, modifier Telex, dấu thanh Telex)
_VN_CHAR_MAP: Dict[str, Tuple[str, str, str]] = {
    "a": ("a", "", ""),
    "á": ("a", "", "s"),
    "à": ("a", "", "f"),
    "ả": ("a", "", "r"),
    "ã": ("a", "", "x"),
    "ạ": ("a", "", "j"),
    "ă": ("a", "w", ""),
    "ắ": ("a", "w", "s"),
    "ằ": ("a", "w", "f"),
    "ẳ": ("a", "w", "r"),
    "ẵ": ("a", "w", "x"),
    "ặ": ("a", "w", "j"),
    "â": ("a", "a", ""),
    "ấ": ("a", "a", "s"),
    "ầ": ("a", "a", "f"),
    "ẩ": ("a", "a", "r"),
    "ẫ": ("a", "a", "x"),
    "ậ": ("a", "a", "j"),
    "e": ("e", "", ""),
    "é": ("e", "", "s"),
    "è": ("e", "", "f"),
    "ẻ": ("e", "", "r"),
    "ẽ": ("e", "", "x"),
    "ẹ": ("e", "", "j"),
    "ê": ("e", "e", ""),
    "ế": ("e", "e", "s"),
    "ề": ("e", "e", "f"),
    "ể": ("e", "e", "r"),
    "ễ": ("e", "e", "x"),
    "ệ": ("e", "e", "j"),
    "i": ("i", "", ""),
    "í": ("i", "", "s"),
    "ì": ("i", "", "f"),
    "ỉ": ("i", "", "r"),
    "ĩ": ("i", "", "x"),
    "ị": ("i", "", "j"),
    "o": ("o", "", ""),
    "ó": ("o", "", "s"),
    "ò": ("o", "", "f"),
    "ỏ": ("o", "", "r"),
    "õ": ("o", "", "x"),
    "ọ": ("o", "", "j"),
    "ô": ("o", "o", ""),
    "ố": ("o", "o", "s"),
    "ồ": ("o", "o", "f"),
    "ổ": ("o", "o", "r"),
    "ỗ": ("o", "o", "x"),
    "ộ": ("o", "o", "j"),
    "ơ": ("o", "w", ""),
    "ớ": ("o", "w", "s"),
    "ờ": ("o", "w", "f"),
    "ở": ("o", "w", "r"),
    "ỡ": ("o", "w", "x"),
    "ợ": ("o", "w", "j"),
    "u": ("u", "", ""),
    "ú": ("u", "", "s"),
    "ù": ("u", "", "f"),
    "ủ": ("u", "", "r"),
    "ũ": ("u", "", "x"),
    "ụ": ("u", "", "j"),
    "ư": ("u", "w", ""),
    "ứ": ("u", "w", "s"),
    "ừ": ("u", "w", "f"),
    "ử": ("u", "w", "r"),
    "ữ": ("u", "w", "x"),
    "ự": ("u", "w", "j"),
    "y": ("y", "", ""),
    "ý": ("y", "", "s"),
    "ỳ": ("y", "", "f"),
    "ỷ": ("y", "", "r"),
    "ỹ": ("y", "", "x"),
    "ỵ": ("y", "", "j"),
}

_BASE_TONE_TO_CHAR: Dict[Tuple[str, str], str] = {
    ("a", ""): "a",
    ("a", "s"): "á",
    ("a", "f"): "à",
    ("a", "r"): "ả",
    ("a", "x"): "ã",
    ("a", "j"): "ạ",
    ("ă", ""): "ă",
    ("ă", "s"): "ắ",
    ("ă", "f"): "ằ",
    ("ă", "r"): "ẳ",
    ("ă", "x"): "ẵ",
    ("ă", "j"): "ặ",
    ("â", ""): "â",
    ("â", "s"): "ấ",
    ("â", "f"): "ầ",
    ("â", "r"): "ẩ",
    ("â", "x"): "ẫ",
    ("â", "j"): "ậ",
    ("e", ""): "e",
    ("e", "s"): "é",
    ("e", "f"): "è",
    ("e", "r"): "ẻ",
    ("e", "x"): "ẽ",
    ("e", "j"): "ẹ",
    ("ê", ""): "ê",
    ("ê", "s"): "ế",
    ("ê", "f"): "ề",
    ("ê", "r"): "ể",
    ("ê", "x"): "ễ",
    ("ê", "j"): "ệ",
    ("i", ""): "i",
    ("i", "s"): "í",
    ("i", "f"): "ì",
    ("i", "r"): "ỉ",
    ("i", "x"): "ĩ",
    ("i", "j"): "ị",
    ("o", ""): "o",
    ("o", "s"): "ó",
    ("o", "f"): "ò",
    ("o", "r"): "ỏ",
    ("o", "x"): "õ",
    ("o", "j"): "ọ",
    ("ô", ""): "ô",
    ("ô", "s"): "ố",
    ("ô", "f"): "ồ",
    ("ô", "r"): "ổ",
    ("ô", "x"): "ỗ",
    ("ô", "j"): "ộ",
    ("ơ", ""): "ơ",
    ("ơ", "s"): "ớ",
    ("ơ", "f"): "ờ",
    ("ơ", "r"): "ở",
    ("ơ", "x"): "ỡ",
    ("ơ", "j"): "ợ",
    ("u", ""): "u",
    ("u", "s"): "ú",
    ("u", "f"): "ù",
    ("u", "r"): "ủ",
    ("u", "x"): "ũ",
    ("u", "j"): "ụ",
    ("ư", ""): "ư",
    ("ư", "s"): "ứ",
    ("ư", "f"): "ừ",
    ("ư", "r"): "ử",
    ("ư", "x"): "ữ",
    ("ư", "j"): "ự",
    ("y", ""): "y",
    ("y", "s"): "ý",
    ("y", "f"): "ỳ",
    ("y", "r"): "ỷ",
    ("y", "x"): "ỹ",
    ("y", "j"): "ỵ",
}

_TONE_KEYS = {"s", "f", "r", "x", "j"}
_PLAIN_VOWELS = set("aeiouy")
_DIACRITIC_VOWELS = set("ăâêôơư")
_ALL_VOWELS = _PLAIN_VOWELS | _DIACRITIC_VOWELS


class TelexConverter:
    """Chuyển đổi hai chiều giữa tiếng Việt có dấu và chuỗi gõ Telex."""

    @staticmethod
    def word_to_telex(word: str) -> str:
        """Chuyển một từ tiếng Việt có dấu thành chuỗi phím Telex (a-z)."""
        chars_out: List[str] = []
        tones: List[str] = []
        lowered = word.lower()

        # Tách nguyên âm gốc + modifier trước để xử lý tổ hợp ươ -> uow gọn hơn
        decomposed: List[Tuple[str, str]] = []
        for ch in lowered:
            if ch == "đ":
                decomposed.append(("dd", ""))
            elif ch in _VN_CHAR_MAP:
                base, mod, tone = _VN_CHAR_MAP[ch]
                decomposed.append((base + mod, tone))
            elif "a" <= ch <= "z":
                decomposed.append((ch, ""))

        idx = 0
        n = len(decomposed)
        while idx < n:
            token, tone = decomposed[idx]
            if tone:
                tones.append(tone)

            # Tổ hợp 'ươ' liên tiếp -> gõ chuẩn Telex 'uow'
            if (
                token == "uw"
                and idx + 1 < n
                and decomposed[idx + 1][0] == "ow"
            ):
                next_tone = decomposed[idx + 1][1]
                if next_tone:
                    tones.append(next_tone)
                chars_out.append("uow")
                idx += 2
                continue

            chars_out.append(token)
            idx += 1

        return "".join(chars_out) + "".join(tones)

    @classmethod
    def vietnamese_to_telex(cls, text: str) -> str:
        """Chuyển một câu tiếng Việt sang chuỗi Telex gồm chữ thường a-z và khoảng trắng."""
        words = re.findall(r"[a-zA-ZÀ-ỹ]+", text)
        telex_words = [cls.word_to_telex(w) for w in words]
        return " ".join(w for w in telex_words if w)

    @classmethod
    def telex_word_to_vietnamese(cls, telex_word: str) -> str:
        """Chuyển một từ Telex sang từ tiếng Việt có dấu hoàn chỉnh."""
        if not telex_word:
            return ""

        is_all_upper = telex_word.isupper() and len(telex_word) > 1
        is_capitalized = telex_word[0].isupper()
        lowered = telex_word.lower()

        chars: List[str] = []
        tone = ""

        for ch in lowered:
            if not ("a" <= ch <= "z"):
                chars.append(ch)
                continue

            vowel_indices = [i for i, c in enumerate(chars) if c in _ALL_VOWELS]
            has_vowel = len(vowel_indices) > 0

            # Quy tắc dd -> đ ở đầu âm tiết
            if ch == "d" and not has_vowel and chars == ["d"]:
                chars[0] = "đ"
                continue

            # Quy tắc dấu thanh: chỉ áp dụng khi đã có nguyên âm trong âm tiết
            if ch in _TONE_KEYS and has_vowel:
                tone = ch
                continue

            # Quy tắc aa -> â, ee -> ê, oo -> ô
            if ch in ("a", "e", "o") and has_vowel:
                target_map = {"a": "â", "e": "ê", "o": "ô"}
                last_vowel_idx = vowel_indices[-1]
                # Áp dụng nếu ngay sau nguyên âm đó hoặc sau phụ âm cuối (vd: tieeng, tene)
                if chars[-1] == ch or (
                    chars[last_vowel_idx] == ch
                    and all(c not in _ALL_VOWELS for c in chars[last_vowel_idx + 1 :])
                ):
                    chars[last_vowel_idx] = target_map[ch]
                    continue

            # Quy tắc w -> ă, ơ, ư (và uo + w -> ươ)
            if ch == "w" and has_vowel:
                effective_vowels = cls._effective_vowel_indices(chars)
                if not effective_vowels:
                    effective_vowels = vowel_indices

                # Kiểm tra cặp uo / ưo / ươ trước w
                applied_uo = False
                if len(effective_vowels) >= 2:
                    v_prev = effective_vowels[-2]
                    v_last = effective_vowels[-1]
                    if (
                        v_last == v_prev + 1
                        and chars[v_prev] in ("u", "ư")
                        and chars[v_last] == "o"
                    ):
                        chars[v_prev] = "ư"
                        chars[v_last] = "ơ"
                        applied_uo = True

                if not applied_uo:
                    w_map = {"a": "ă", "o": "ơ", "u": "ư"}
                    applied_single = False
                    for idx in reversed(effective_vowels):
                        if chars[idx] in w_map:
                            chars[idx] = w_map[chars[idx]]
                            applied_single = True
                            break
                    if not applied_single:
                        chars.append(ch)
                continue

            chars.append(ch)

        result = cls._apply_tone_to_chars(chars, tone)
        if is_all_upper:
            return result.upper()
        if is_capitalized and result:
            return result[0].upper() + result[1:]
        return result

    @classmethod
    def telex_to_vietnamese(cls, telex_text: str) -> str:
        """Chuyển một chuỗi hoặc câu Telex sang tiếng Việt có dấu, giữ nguyên dấu câu."""
        return re.sub(
            r"[a-zA-Z]+",
            lambda match: cls.telex_word_to_vietnamese(match.group(0)),
            telex_text,
        )

    @staticmethod
    def _effective_vowel_indices(chars: List[str]) -> List[int]:
        """Xác định vị trí các nguyên âm thuộc phần vần (loại bỏ u trong qu-, i trong gi-)."""
        vowel_indices = [i for i, c in enumerate(chars) if c in _ALL_VOWELS]
        if not vowel_indices:
            return []

        onset_end = 0
        if (
            len(chars) >= 3
            and chars[0] == "q"
            and chars[1] == "u"
            and chars[2] in _ALL_VOWELS
        ):
            onset_end = 2
        elif (
            len(chars) >= 3
            and chars[0] == "g"
            and chars[1] == "i"
            and chars[2] in _ALL_VOWELS
        ):
            onset_end = 2

        return [i for i in vowel_indices if i >= onset_end]

    @classmethod
    def _apply_tone_to_chars(cls, chars: List[str], tone: str) -> str:
        """Đặt dấu thanh đúng vị trí quy chuẩn chính tả tiếng Việt."""
        if not tone:
            return "".join(chars)

        nucleus = cls._effective_vowel_indices(chars)
        if not nucleus:
            return "".join(chars)

        # 1. Ưu tiên nguyên âm có dấu phụ (â, ê, ô, ă, ơ, ư); nếu có ươ chọn ơ (cuối)
        diacritic_indices = [i for i in nucleus if chars[i] in _DIACRITIC_VOWELS]
        if diacritic_indices:
            target_idx = diacritic_indices[-1]
        elif len(nucleus) == 1:
            target_idx = nucleus[0]
        elif len(nucleus) == 2:
            has_coda = nucleus[-1] < len(chars) - 1
            target_idx = nucleus[1] if has_coda else nucleus[0]
        else:
            target_idx = nucleus[1]

        base_vowel = chars[target_idx]
        if (base_vowel, tone) in _BASE_TONE_TO_CHAR:
            chars[target_idx] = _BASE_TONE_TO_CHAR[(base_vowel, tone)]
        return "".join(chars)


# Corpus mẫu ~125 câu giao tiếp tiếng Việt cơ bản & hỗ trợ người bệnh Parkinson
DEFAULT_VIETNAMESE_CORPUS: List[str] = [
    "xin chào mọi người",
    "chào buổi sáng",
    "chào buổi trưa",
    "chào buổi tối",
    "tạm biệt và hẹn gặp lại",
    "chúc bạn một ngày tốt lành",
    "chúc ngủ ngon",
    "cảm ơn bạn rất nhiều",
    "cảm ơn bác sĩ đã giúp đỡ",
    "xin lỗi tôi gõ hơi chậm",
    "không sao đâu bạn cứ từ từ",
    "hôm nay bạn cảm thấy thế nào",
    "hôm nay tôi cảm thấy khỏe hơn rồi",
    "hôm nay tôi hơi mệt một chút",
    "tôi bị run tay nên gõ phím khó khăn",
    "hãy nói chậm lại một chút giúp tôi",
    "tôi muốn uống một cốc nước ấm",
    "cho tôi xin thêm nước lọc",
    "tôi đói bụng rồi muốn ăn cơm",
    "bữa sáng hôm nay có phở và bánh mì",
    "bữa trưa tôi ăn cơm với cá và rau",
    "bữa tối ăn cháo cho dễ tiêu",
    "món ăn này rất ngon và vừa miệng",
    "tôi không muốn ăn thêm nữa",
    "đã đến giờ uống thuốc rồi",
    "tôi cần uống thuốc đúng giờ mỗi ngày",
    "bác sĩ dặn uống thuốc sau bữa ăn",
    "hãy lấy giúp tôi hộp thuốc trên bàn",
    "tôi thấy hơi chóng mặt và buồn ngủ",
    "tay phải của tôi đang run nhiều",
    "chân tôi hơi mỏi muốn ngồi nghỉ",
    "tôi cần đi nghỉ ngơi một lát",
    "làm ơn gọi giúp tôi người nhà",
    "hãy gọi điện cho con trai tôi",
    "hãy gọi điện cho con gái tôi",
    "vợ tôi đang ở ngoài phòng khách",
    "chồng tôi sắp về đến nhà rồi",
    "gia đình luôn ở bên cạnh chăm sóc tôi",
    "tôi muốn ra ngoài đi dạo một chút",
    "thời tiết hôm nay mát mẻ dễ chịu",
    "trời hôm nay nắng đẹp quá",
    "ngoài trời đang mưa rất to",
    "hãy bật quạt lên giúp tôi với",
    "làm ơn tắt điều hòa vì hơi lạnh",
    "tôi muốn mở cửa sổ cho thoáng",
    "tôi muốn đi vệ sinh ngay bây giờ",
    "hãy đỡ tôi đứng dậy cẩn thận",
    "cảm ơn bạn đã dìu tôi đi lại",
    "bạn có thể đọc lại câu vừa rồi không",
    "tôi chưa nghe rõ xin hãy nhắc lại",
    "tôi hiểu ý của bạn rồi",
    "tôi đồng ý với phương án đó",
    "tôi không đồng ý lắm với việc này",
    "bạn tên là gì vậy",
    "tôi tên là minh rất vui được làm quen",
    "nhà bạn ở đâu thế",
    "tôi sống cùng gia đình tại hà nội",
    "bây giờ là mấy giờ rồi",
    "sáng mai tôi có lịch tái khám bệnh",
    "chiều nay bác sĩ sẽ kiểm tra lại",
    "tối nay cả nhà cùng xem ti vi",
    "tôi thích nghe nhạc nhẹ vào buổi tối",
    "tôi thích đọc sách và uống trà nóng",
    "bạn có khỏe không dạo này thế nào",
    "mọi người trong nhà đều bình an",
    "xin chào tiếng việt thật đẹp",
    "hệ thống hỗ trợ gõ phím tiếng việt",
    "công nghệ giúp cuộc sống tốt đẹp hơn",
    "trí tuệ nhân tạo hỗ trợ người bệnh",
    "bàn phím thông minh sửa lỗi chạm nhầm",
    "lọc nhiễu rung tay khi nhấn màn hình",
    "tôi đang tập gõ câu tiếng việt có dấu",
    "xin vui lòng chờ tôi thêm vài giây",
    "đừng vội vàng cứ bình tĩnh làm việc",
    "bạn thật tốt bụng và kiên nhẫn",
    "tôi rất trân trọng sự giúp đỡ này",
    "nhờ bạn kiểm tra tin nhắn mới giúp tôi",
    "hãy gửi tin nhắn này cho bác sĩ",
    "tôi đang ở nhà không đi đâu cả",
    "khi nào bạn rảnh thì ghé thăm tôi nhé",
    "cuối tuần này các cháu sẽ về chơi",
    "tôi rất nhớ con và các cháu",
    "chúc mừng sinh nhật bạn nhiều sức khỏe",
    "chúc mọi người luôn vui vẻ và hạnh phúc",
    "đừng quên uống đủ nước mỗi ngày nhé",
    "ngủ đủ giấc giúp tinh thần tỉnh táo",
    "tập thể dục nhẹ nhàng rất tốt cho sức khỏe",
    "mỗi sáng tôi tập đi bộ quanh sân",
    "hít thở sâu để giảm bớt căng thẳng",
    "tôi bị đau nhức ở vai và lưng",
    "vui lòng lấy cho tôi cái gối và chăn",
    "đèn phòng ngủ hơi chói mắt quá",
    "tôi muốn tắt đèn để đi ngủ sớm",
    "sáng nay tôi thức dậy lúc sáu giờ",
    "tối qua tôi ngủ khá ngon giấc",
    "hôm qua trời lạnh hơn hôm nay nhiều",
    "ngày mai gia đình mình đi khám định kỳ",
    "tuần sau tôi cần lấy thêm đơn thuốc mới",
    "thuốc này uống vào buổi sáng và buổi tối",
    "mỗi lần uống một viên với nhiều nước",
    "không được tự ý bỏ thuốc giữa chừng",
    "nếu thấy khó thở phải báo ngay cho bác sĩ",
    "tôi đã cảm thấy đỡ run hơn trước rồi",
    "việc nhắn tin bây giờ dễ dàng hơn hẳn",
    "câu này được giải mã rất chính xác",
    "cho tôi hỏi đường ra bệnh viện gần nhất",
    "tôi cần một ly sữa nóng không đường",
    "làm ơn mở nhỏ tiếng nhạc lại một chút",
    "cảm ơn vì đã lắng nghe tôi chia sẻ",
    "chúng ta cùng cố gắng mỗi ngày nhé",
    "tôi tin mọi chuyện sẽ ổn thôi",
    "quá trình hồi phục cần nhiều thời gian",
    "sức khỏe là điều quý giá nhất của con người",
    "hãy giữ gìn sức khỏe thật tốt nhé",
    "bạn làm việc vất vả cả ngày rồi",
    "mời bạn ngồi xuống uống chén trà",
    "cơm nước đã chuẩn bị xong hết rồi",
    "mọi thứ trong phòng đều gọn gàng sạch sẽ",
    "tôi cần kính mắt để đọc chữ rõ hơn",
    "màn hình chữ to rất dễ nhìn với tôi",
    "hẹn gặp lại bạn vào sáng ngày mai",
    "tạm biệt chúc một buổi tối an lành",
]


class VietnameseLanguageModel:
    """Mô hình ngôn ngữ N-gram (Unigram/Bigram/Trigram) cho tiếng Việt và Telex."""

    BOS = "<BOS>"
    EOS = "<EOS>"
    UNK = "<UNK>"

    def __init__(
        self,
        smoothing_k: float = 0.5,
        load_builtin: bool = True,
        trigram_weight: float = 0.55,
        bigram_weight: float = 0.35,
        unigram_weight: float = 0.10,
    ) -> None:
        if smoothing_k <= 0:
            raise ValueError("Hệ số làm mượt smoothing_k phải lớn hơn 0.")
        total_w = trigram_weight + bigram_weight + unigram_weight
        if total_w <= 0:
            raise ValueError("Tổng trọng số nội suy phải lớn hơn 0.")

        self.smoothing_k = float(smoothing_k)
        self.trigram_weight = trigram_weight / total_w
        self.bigram_weight = bigram_weight / total_w
        self.unigram_weight = unigram_weight / total_w

        # Từ vựng ký tự Telex cố định: a-z và <EOS>
        self.char_vocab: List[str] = [chr(ord("a") + i) for i in range(26)] + [self.EOS]
        self.char_unigram_counts: Counter = Counter()
        self.char_bigram_counts: Dict[str, Counter] = defaultdict(Counter)
        self.char_trigram_counts: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
        self.char_bigram_totals: Counter = Counter()
        self.char_trigram_totals: Counter = Counter()
        self.total_chars: int = 0

        # Bảng từ vựng & N-gram mức từ tiếng Việt
        self.word_vocab: set = {self.EOS}
        self.word_unigram_counts: Counter = Counter()
        self.word_bigram_counts: Dict[str, Counter] = defaultdict(Counter)
        self.word_trigram_counts: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
        self.word_bigram_totals: Counter = Counter()
        self.word_trigram_totals: Counter = Counter()
        self.total_words: int = 0

        # Từ vựng Telex <-> Tiếng Việt hỗ trợ thưởng từ hợp lệ trong bộ giải mã
        self.telex_word_counts: Counter = Counter()
        self.telex_to_vn_counts: Dict[str, Counter] = defaultdict(Counter)
        self.sentence_count: int = 0

        if load_builtin:
            self.train_from_sentences(DEFAULT_VIETNAMESE_CORPUS)

    @staticmethod
    def tokenize_vietnamese_words(sentence: str) -> List[str]:
        """Tách câu tiếng Việt thành danh sách từ chữ thường."""
        return [w.lower() for w in re.findall(r"[a-zA-ZÀ-ỹ]+", sentence) if w]

    def train_sentence(self, sentence: str) -> bool:
        """Huấn luyện N-gram ký tự Telex và N-gram từ từ một câu tiếng Việt."""
        words = self.tokenize_vietnamese_words(sentence)
        if not words:
            return False

        self.sentence_count += 1

        # 1. Cập nhật N-gram mức từ tiếng Việt
        padded_words = [self.BOS, self.BOS] + words + [self.EOS]
        for idx in range(2, len(padded_words)):
            w2, w1, w = padded_words[idx - 2], padded_words[idx - 1], padded_words[idx]
            self.word_vocab.add(w)
            self.word_unigram_counts[w] += 1
            self.word_bigram_counts[w1][w] += 1
            self.word_bigram_totals[w1] += 1
            self.word_trigram_counts[(w2, w1)][w] += 1
            self.word_trigram_totals[(w2, w1)] += 1
            self.total_words += 1

        # 2. Cập nhật N-gram mức ký tự Telex cho từng từ
        for vn_word in words:
            telex_word = TelexConverter.word_to_telex(vn_word)
            if not telex_word:
                continue

            self.telex_word_counts[telex_word] += 1
            self.telex_to_vn_counts[telex_word][vn_word] += 1

            padded_chars = [self.BOS, self.BOS] + list(telex_word) + [self.EOS]
            for idx in range(2, len(padded_chars)):
                c2, c1, c = padded_chars[idx - 2], padded_chars[idx - 1], padded_chars[idx]
                self.char_unigram_counts[c] += 1
                self.char_bigram_counts[c1][c] += 1
                self.char_bigram_totals[c1] += 1
                self.char_trigram_counts[(c2, c1)][c] += 1
                self.char_trigram_totals[(c2, c1)] += 1
                self.total_chars += 1

        return True

    def train_from_sentences(self, sentences: Iterable[str]) -> int:
        """Huấn luyện mô hình từ danh sách các câu tiếng Việt."""
        loaded = 0
        for sentence in sentences:
            if self.train_sentence(sentence):
                loaded += 1
        return loaded

    def load_corpus_file(self, filepath: str, encoding: str = "utf-8") -> int:
        """Nạp thêm dữ liệu huấn luyện từ file .txt bên ngoài (mỗi dòng 1 câu)."""
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Không tìm thấy file corpus: {filepath}")

        loaded = 0
        with open(filepath, "r", encoding=encoding) as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if self.train_sentence(stripped):
                    loaded += 1
        return loaded

    # ------------------------------------------------------------------
    # Xác suất mức ký tự Telex (Add-k / Laplace smoothing)
    # ------------------------------------------------------------------

    def char_unigram_prob(self, char: str) -> float:
        """Tính P(c) mức ký tự có làm mượt Add-k."""
        v_size = len(self.char_vocab)
        k = self.smoothing_k
        count = self.char_unigram_counts.get(char, 0)
        return (count + k) / (self.total_chars + k * v_size)

    def char_bigram_prob(self, prev_char: str, char: str) -> float:
        """Tính P(c | prev_char) mức ký tự có làm mượt Add-k."""
        v_size = len(self.char_vocab)
        k = self.smoothing_k
        count = self.char_bigram_counts.get(prev_char, {}).get(char, 0)
        total = self.char_bigram_totals.get(prev_char, 0)
        return (count + k) / (total + k * v_size)

    def char_trigram_prob(self, prev2_char: str, prev1_char: str, char: str) -> float:
        """Tính P(c | prev2_char, prev1_char) mức ký tự có làm mượt Add-k."""
        v_size = len(self.char_vocab)
        k = self.smoothing_k
        ctx = (prev2_char, prev1_char)
        count = self.char_trigram_counts.get(ctx, {}).get(char, 0)
        total = self.char_trigram_totals.get(ctx, 0)
        return (count + k) / (total + k * v_size)

    def char_lm_prob(self, history: Sequence[str], char: str) -> float:
        """Tính xác suất nội suy Trigram + Bigram + Unigram mức ký tự."""
        if len(history) == 0:
            c2, c1 = self.BOS, self.BOS
        elif len(history) == 1:
            c2, c1 = self.BOS, history[0]
        else:
            c2, c1 = history[-2], history[-1]

        p1 = self.char_unigram_prob(char)
        p2 = self.char_bigram_prob(c1, char)
        p3 = self.char_trigram_prob(c2, c1, char)
        return (
            self.trigram_weight * p3
            + self.bigram_weight * p2
            + self.unigram_weight * p1
        )

    def log_char_lm_prob(self, history: Sequence[str], char: str) -> float:
        """Tính ln P_LM(char | history) mức ký tự."""
        return math.log(self.char_lm_prob(history, char))

    # ------------------------------------------------------------------
    # Xác suất mức từ tiếng Việt (Add-k / Laplace smoothing)
    # ------------------------------------------------------------------

    def word_unigram_prob(self, word: str) -> float:
        """Tính P(word) mức từ tiếng Việt có làm mượt Add-k."""
        w = word.lower() if word != self.EOS else word
        v_size = max(len(self.word_vocab), 1) + 1  # +1 cho <UNK>
        k = self.smoothing_k
        count = self.word_unigram_counts.get(w, 0)
        return (count + k) / (self.total_words + k * v_size)

    def word_bigram_prob(self, prev_word: str, word: str) -> float:
        """Tính P(word | prev_word) mức từ tiếng Việt có làm mượt Add-k."""
        pw = prev_word.lower() if prev_word != self.BOS else prev_word
        w = word.lower() if word != self.EOS else word
        v_size = max(len(self.word_vocab), 1) + 1
        k = self.smoothing_k
        count = self.word_bigram_counts.get(pw, {}).get(w, 0)
        total = self.word_bigram_totals.get(pw, 0)
        return (count + k) / (total + k * v_size)

    def word_trigram_prob(self, prev2_word: str, prev1_word: str, word: str) -> float:
        """Tính P(word | prev2_word, prev1_word) mức từ tiếng Việt có làm mượt Add-k."""
        w2 = prev2_word.lower() if prev2_word != self.BOS else prev2_word
        w1 = prev1_word.lower() if prev1_word != self.BOS else prev1_word
        w = word.lower() if word != self.EOS else word
        v_size = max(len(self.word_vocab), 1) + 1
        k = self.smoothing_k
        ctx = (w2, w1)
        count = self.word_trigram_counts.get(ctx, {}).get(w, 0)
        total = self.word_trigram_totals.get(ctx, 0)
        return (count + k) / (total + k * v_size)

    def word_lm_prob(self, prev_words: Sequence[str], word: str) -> float:
        """Tính xác suất nội suy Trigram + Bigram + Unigram mức từ tiếng Việt."""
        if len(prev_words) == 0:
            w2, w1 = self.BOS, self.BOS
        elif len(prev_words) == 1:
            w2, w1 = self.BOS, prev_words[0]
        else:
            w2, w1 = prev_words[-2], prev_words[-1]

        p1 = self.word_unigram_prob(word)
        p2 = self.word_bigram_prob(w1, word)
        p3 = self.word_trigram_prob(w2, w1, word)
        return (
            self.trigram_weight * p3
            + self.bigram_weight * p2
            + self.unigram_weight * p1
        )

    def log_word_lm_prob(self, prev_words: Sequence[str], word: str) -> float:
        """Tính ln P_LM(word | prev_words) mức từ tiếng Việt."""
        return math.log(self.word_lm_prob(prev_words, word))

    def is_known_telex_word(self, telex_word: str) -> bool:
        """Kiểm tra một từ Telex có xuất hiện trong từ vựng huấn luyện hay không."""
        return telex_word.lower() in self.telex_word_counts

    def is_known_vietnamese_word(self, vn_word: str) -> bool:
        """Kiểm tra một từ tiếng Việt có dấu có trong từ vựng hay không."""
        return vn_word.lower() in self.word_unigram_counts

    def get_stats(self) -> Dict[str, int]:
        """Trả về thống kê kích thước corpus và từ vựng đã nạp."""
        unique_vn = len(self.word_unigram_counts) - (
            1 if self.EOS in self.word_unigram_counts else 0
        )
        return {
            "sentences": self.sentence_count,
            "total_words": self.total_words,
            "unique_vn_words": unique_vn,
            "unique_telex_words": len(self.telex_word_counts),
            "total_telex_chars": self.total_chars,
        }


if __name__ == "__main__":
    lm = VietnameseLanguageModel()
    print("Thống kê Language Model:", lm.get_stats())

    test_vn_words = [
        "xin",
        "chào",
        "tiếng",
        "việt",
        "thuốc",
        "nước",
        "người",
        "được",
        "cảm",
        "ơn",
        "khỏe",
        "giờ",
        "quá",
        "rất",
        "sáng",
        "bệnh",
    ]
    print("\nKiểm thử chuyển đổi Vietnamese <-> Telex:")
    for w in test_vn_words:
        tx = TelexConverter.word_to_telex(w)
        back = TelexConverter.telex_word_to_vietnamese(tx)
        status = "OK" if back == w else f"LỆCH ({back})"
        print(f"  {w:8s} -> {tx:10s} -> {back:8s} [{status}]")

    print("\nKiểm thử xác suất N-gram:")
    print(
        f"  P_char('a' | 'ch')  = {lm.char_lm_prob(['c', 'h'], 'a'):.5f} "
        f"vs P_char('z' | 'ch') = {lm.char_lm_prob(['c', 'h'], 'z'):.5f}"
    )
    print(
        f"  P_word('chào' | ['xin']) = {lm.word_lm_prob(['xin'], 'chào'):.5f} "
        f"vs P_word('thuốc' | ['xin']) = {lm.word_lm_prob(['xin'], 'thuốc'):.5f}"
    )
