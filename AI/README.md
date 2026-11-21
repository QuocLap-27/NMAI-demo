# TouchAssist: Hệ thống bàn phím ảo hỗ trợ người bệnh Parkinson sử dụng mô hình Giải mã cú chạm Bayes và Quy hoạch động Viterbi cho tiếng Việt

> Đồ án môn **Nhập môn Trí tuệ nhân tạo** — Hệ thống bàn phím ảo tiếng Việt (Telex) tích hợp mô hình xác suất không gian Gaussian 2D, mô hình lọc nhiễu thời gian run tay và quy hoạch động Viterbi kết hợp N-gram tiếng Việt.

---

## 1. Tổng quan bài toán

Người bệnh Parkinson thường gặp hai loại nhiễu đặc trưng khi nhập liệu trên màn hình cảm ứng:
1. **Nhiễu chạm lệch không gian (Spatial Jitter)**: Đầu ngón tay rung làm điểm chạm lệch khỏi tâm phím mục tiêu sang các phím lân cận.
2. **Nhiễu chạm lặp do run tay (Tremor Bounce)**: Ngón tay bật nảy nhanh tạo ra cú chạm lặp ngoài ý muốn với khoảng cách thời gian rất ngắn (`< 200 ms`) và khoảng dịch chuyển nhỏ (`< 3 mm`).

Các phương pháp lọc thô truyền thống (như bỏ qua mọi lần lặp `< 500 ms`) không sửa được chạm lệch và thường xóa nhầm các cặp ký tự lặp hợp lệ trong bộ gõ Telex tiếng Việt (như `aa -> â`, `ee -> ê`, `oo -> ô`, `dd -> đ`).

**TouchAssist** giải quyết bài toán bằng mô hình giải mã xác suất Bayes kết hợp quy hoạch động **Viterbi**:

$$\log \text{Score}(C \mid I) = \sum_i \Big( \log P_s(x_i, y_i \mid k_i) + \log P_t(\text{action}_i \mid \Delta t_i, d_i) + \log P_{LM}(c_i \mid \text{history}) \Big)$$

Trong đó tại mỗi bước chạm, thuật toán cân nhắc giữa hai hành động: **`EMIT`** (chọn ký tự `a–z`) và **`SKIP`** (bỏ qua cú chạm nhiễu do run tay), sau đó truy vết chuỗi tối ưu và chuyển đổi sang tiếng Việt có dấu hoàn chỉnh theo chuẩn Telex.

---

## 2. Cấu trúc dự án

```text
TouchAssist/
├── keyboard_layout.py      # Layout QWERTY (860x340) & mô hình xác suất không gian Gaussian 2D
├── ngram_model.py          # Bộ chuyển đổi Telex <-> Tiếng Việt & N-gram (ký tự/từ) làm mượt Add-k
├── viterbi_decoder.py      # Thuật toán Giải mã cú chạm Bayes + Quy hoạch động Viterbi
├── main_gui.py             # Giao diện demo trực quan Tkinter (AI vs Baseline, mô phỏng run tay)
├── evaluate_simulation.py  # Script thực nghiệm đối chứng (20 câu, 217 từ, xuất báo cáo)
├── evaluation_report.txt   # Kết quả thực nghiệm trích xuất từ evaluate_simulation.py
├── requirements.txt        # Ghi chú thư viện (chỉ dùng Python Standard Library)
├── .gitignore              # Cấu hình loại trừ file tạm cho Git
├── LICENSE                 # Giấy phép MIT
└── README.md               # Tài liệu hướng dẫn dự án
```

---

## 3. Kiến trúc các module chính

- **`KeyboardLayout` ([`keyboard_layout.py`](keyboard_layout.py))**:
  - Khởi tạo 28 phím chuẩn QWERTY (`a–z`, `SPACE`, `BACKSPACE`) với tọa độ tâm `(x, y)` và kích thước cụ thể trên Canvas `860x340`.
  - Tính xác suất không gian `spatial_probability(touch_x, touch_y, key_char)` theo phân phối chuẩn Gaussian 2D và chuẩn hóa trên 26 chữ cái bằng log-sum-exp.
  - Quy đổi đơn vị vật lý `mm <-> pixel` (`96 DPI`).

- **`TelexConverter` & `VietnameseLanguageModel` ([`ngram_model.py`](ngram_model.py))**:
  - Hỗ trợ chuyển đổi hai chiều tiếng Việt $\leftrightarrow$ Telex (`aa->â`, `ee->ê`, `oo->ô`, `dd->đ`, `aw->ă`, `ow->ơ`, `uw->ư`, `uow->ươ` và dấu thanh `s, f, r, x, j`).
  - Tích hợp sẵn corpus **122 câu giao tiếp tiếng Việt cơ bản** (379 từ đơn duy nhất, 5.141 ký tự Telex), huấn luyện Unigram/Bigram/Trigram cả ở **mức ký tự Telex** và **mức từ tiếng Việt** với làm mượt Add-k / Laplace (`smoothing_k = 0.5`).
  - Hỗ trợ nạp thêm dữ liệu từ file `.txt` ngoài qua `load_corpus_file(filepath)`.

- **`ViterbiDecoder` ([`viterbi_decoder.py`](viterbi_decoder.py))**:
  - Nhận buffer các `TouchEvent(x, y, timestamp_ms)` trước khi nhấn Space.
  - `temporal_probability()` đánh trọng số cao cho trạng thái `SKIP` khi hai cú chạm liên tiếp có `0 <= dt < 200ms` và khoảng cách `< 3mm`.
  - Xây dựng ma trận quy hoạch động `F(i, j)` và mảng `Backpointer` trên không gian log để tránh tràn số dưới, kết hợp kiểm tra từ vựng và Trigram mức từ ở cuối từ.

- **`TouchAssistApp` ([`main_gui.py`](main_gui.py))**:
  - Giao diện demo gồm khung **Output Text**, bảng **Debug & Top-3 ứng viên Viterbi** (kèm độ trễ xử lý `< 50ms`), **Canvas bàn phím ảo** trực quan hóa chấm chạm nhiễu (đỏ) và đường nối phím giải mã (xanh lá).
  - Bảng điều khiển cho phép chuyển đổi **TouchAssist AI** vs **Baseline (`< 500ms`)**, bật/tắt **Giả lập run tay Parkinson**, điều chỉnh thanh trượt `sigma` (`5–25 px`) và chạy tự động **3 câu mẫu**.

---

## 4. Yêu cầu môi trường & Cài đặt

Dự án thiết kế hướng đối tượng (OOP) và **chỉ sử dụng thư viện chuẩn của Python** (`tkinter`, `math`, `random`, `time`, `collections`, `dataclasses`, `re`, `os`, `typing`), không cần cài đặt thư viện ngoài.

- **Yêu cầu**: Python 3.10 trở lên (có sẵn `tkinter`).

```bash
git clone <url-repository-cua-ban>
cd TouchAssist
pip install -r requirements.txt
```

*(Trên Linux nếu thiếu `tkinter`, cài thêm gói hệ thống tương ứng như `sudo apt install python3-tk`.)*

---

## 5. Hướng dẫn chạy nhanh

### Chạy giao diện demo trực quan (Tkinter)
```bash
python main_gui.py
```

### Chạy thực nghiệm đối chứng (Terminal)
```bash
python evaluate_simulation.py
```
Kết quả sẽ được in trực tiếp ra terminal và lưu vào file `evaluation_report.txt`.

### Kiểm thử từng module độc lập
```bash
python keyboard_layout.py
python ngram_model.py
python viterbi_decoder.py
```

---

## 6. Kết quả thực nghiệm mô phỏng

Thiết lập kiểm thử trong [`evaluate_simulation.py`](evaluate_simulation.py) (`RANDOM_SEED = 42`):
- **Tập kiểm thử**: 20 câu tiếng Việt giao tiếp & hỗ trợ người bệnh (**217 từ**).
- **Mô phỏng người bệnh Parkinson**: Nhiễu tọa độ Gaussian 2D (`sigma = 15.0 px`) + `30%` cú chạm lặp (`50–180 ms`, `< 2.5 mm`).
- **Đối chứng**: **Baseline** (phím gần nhất + lọc lặp thô `< 500 ms`) vs **TouchAssist AI** (`ViterbiDecoder`).

| Chỉ số đánh giá | Baseline | TouchAssist AI | Cải thiện |
|---|---:|---:|---:|
| **WER - Tỉ lệ lỗi từ (%)** | `33.64%` (`73/217` từ) | `5.53%` (`12/217` từ) | **Giảm 83.6%** |
| **Backspaces (lần/câu)** | `4.15` | `0.65` | **Giảm 84.3%** |
| **WPM - Tốc độ gõ (từ/phút)** | `22.14` | `34.78` | **Tăng 57.1%** |
| **Độ trễ giải mã TB (ms/từ)** | `0.03 ms` | `3.11 ms` | **< 50 ms ✓** |

Một số ví dụ trích từ [`evaluation_report.txt`](evaluation_report.txt):

| Từ gốc | Chạm nhiễu raw | Baseline | TouchAssist AI | Độ trễ |
|---|---|---|---|---:|
| `thuốc` | `thuoocss` | `thuóc` | `thuốc` | `4.72 ms` |
| `không` | `knnooongg` | `knong` | `không` | `4.45 ms` |
| `nhiều` | `nnhieeeuuff` | `nhièu` | `nhiều` | `5.81 ms` |
| `thấy` | `thaaayss` | `tháy` | `thấy` | `5.12 ms` |
| `đình` | `ddinnhff` | `dình` | `đình` | `3.95 ms` |

---

## 7. Mở rộng corpus tiếng Việt

Có thể bổ sung thêm câu giao tiếp từ file văn bản `.txt` bên ngoài (mỗi dòng một câu):

```python
from ngram_model import VietnameseLanguageModel

lm = VietnameseLanguageModel()
lm.load_corpus_file("custom_corpus.txt")
```

---

## 8. Giấy phép

Dự án phát hành theo giấy phép [MIT License](LICENSE).
