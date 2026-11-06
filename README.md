# HiveMindRL on graphs

Tái tạo kết quả của Soma và cộng sự (2026), *The hive mind is a single
reinforcement learning agent*, rồi đặt mô hình bầy đàn lên các đồ thị giao
tiếp cố định để xem cấu trúc đồ thị thay đổi tốc độ học như thế nào.

## Cài đặt

Yêu cầu Python 3.10+.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

> `requirements.txt` is mirrored in `pyproject.toml`; project metadata
> (name, version, Python requirement) lives in `pyproject.toml`.

## Chạy thí nghiệm

```bash
# Kiểm tra nhanh toàn bộ pipeline (vài giây)
python scripts/run_all.py --quick

# Chạy đầy đủ: N = 1000 / N = 500, 100 seed (khoảng 4–5 phút trên CPU)
python scripts/run_all.py

# Chỉ chạy một phần
python scripts/run_all.py --only reproduction
python scripts/run_all.py --only graph

# Vẽ lại hình từ kết quả đã lưu, không mô phỏng lại
python scripts/run_all.py --only graph --replot
```

Kết quả ghi vào `results/`, hình ghi vào `figures/`.

Kiểm tra bổ sung trên mạng đuôi nặng (Bảng 2 trong báo cáo):

```bash
python scripts/check_heavy_tail.py
```

Can thiệp gieo nguồn cho đồ thị hai buồng (Hình 4) và chẩn đoán thống kê cho
quy luật `t90 ~ L`:

```bash
python scripts/check_two_room.py      # ~30 s; ghi results/two_room_source*.csv
python scripts/analyze_law.py         # ghi results/law_diagnostics.json
```

Bản nhân rộng của thí nghiệm này (nhiều đồ thị, nhiều `N`, tách thời gian
chờ ở cầu khỏi thời gian lan trong buồng, hồi quy theo độ dẫn `φ`):

```bash
python scripts/check_two_room_replicates.py          # ~20 phút; Hình 5
python scripts/check_two_room_replicates.py --quick  # vài giây
python scripts/check_two_room_replicates.py --refit  # tính lại fit từ CSV đã lưu
```

`check_two_room.py` chỉ gieo type tốt nhất vào một buồng, rồi đo thời gian
lan sang buồng còn lại theo độ rộng cầu. Đây là thiết kế can thiệp thay cho
cấu hình hai buồng cũ (cả hai buồng đều đã có type tốt, nên cầu không mang
thông tin).

## Giao diện mô phỏng

Giao diện chạy cục bộ, không cần thêm dependency (dùng `http.server` của
Python chuẩn):

```bash
python scripts/run_ui.py
# hoặc
python scripts/run_ui.py --port 9000 --no-browser
```

Mở `http://127.0.0.1:8765/`. Giao diện là một dashboard sáu tab:

- **Tổng quan** — bốn chỉ số chính, ba phát hiện, và thẻ “Quy luật t₉₀ ~ L”
  đọc trực tiếp `/api/law` (hệ số, R², CI bootstrap, cảnh báo chẩn đoán).
- **Tái tạo** — chạy Thí nghiệm A với N, bước, seed, kịch bản q tuỳ chọn;
  xem ba đường học, sai lệch P-MCL vs MRD và bảng t₉₀.
- **Đồ thị đơn** — chọn WS, RGG, hai buồng, BA hoặc ER; xem τ, L, đường học
  và cấu trúc đồ thị ở trạng thái cuối.
- **So sánh** — chạy nhiều topology cùng lúc, tái hiện bảng và biểu đồ
  t₉₀ theo τ và theo L; kết quả trả kèm CI và cảnh báo chẩn đoán.
- **Kết quả lưu** — đọc trực tiếp các CSV trong `results/` (kể cả
  `two_room_source*.csv`) và tải về.
- **Kiểm thử** — chạy `pytest -q` ngay trên trình duyệt và xem output.

Mô phỏng chạy trong tiến trình nền một worker; đóng tab không làm mất kết
quả đang chạy. Server bind `127.0.0.1` và tự thử cổng kế tiếp nếu cổng bận.
Nút preset **Nhanh / Vừa / Đầy đủ** đặt N, bước và seed tương ứng. Phím
`Ctrl + Enter` gửi form đang mở.

## Kiểm thử

```bash
python -m pytest -q
```

`pyproject.toml` ([tool.pytest.ini_options]) đã thêm `src` vào `pythonpath`,
nên không cần đặt biến môi trường `PYTHONPATH` khi chạy test.
`scripts/run_all.py` cũng tự thêm `src` vào `sys.path`.

Ngoài các test đơn vị cho bandit, luật học, đồ thị và metrics, bộ test còn
kiểm tra:

- tính tất định theo seed và callback tiến độ của `simulate_on_graph`;
- thiết kế gieo nguồn hai buồng: type tốt không xuất hiện trong buồng đích;
- chẩn đoán quy luật `t90 ~ L`: loại điểm can thiệp / censor / không liên
  thông, CI bootstrap, Cook's D và leave-one-out;
- vòng đời job của web API (tạo, chạy, lấy kết quả, trả PNG);
- validation đầu vào (N, bước, seed, họ đồ thị, tham số) và phục vụ UI tĩnh;
- endpoint `/api/results`, `/api/law` và chặn tên tệp ngoài danh sách cho phép.

Một test tích hợp dùng `N` nhỏ và vài seed nên toàn bộ bộ test vẫn chạy
trong khoảng vài giây.

## Dựng báo cáo PDF

```bash
python scripts/build_report.py
```

Script đọc `report/report.md`, chèn hình từ `figures/`, và ghi
`report/report.pdf`. Cần `reportlab` và `Pillow` (đã có trong
`requirements.txt`).

## Cấu trúc

```
src/hive/          thư viện mô phỏng
  bandit.py        môi trường multi-armed bandit
  models.py        TRD, MRD, P-CL, P-MCL
  swarm.py         weighted voter well-mixed và trên đồ thị
  graphs.py        sinh đồ thị, tau, đường đi trung bình
  metrics.py       t90, final mass, consensus, L2
  reproduce.py     thí nghiệm A (tái tạo)
  graph_sweep.py   thí nghiệm B (khảo sát đồ thị) + can thiệp hai buồng
  analysis.py      chẩn đoán thống kê cho quy luật t90 ~ L
  bridge.py        nhân rộng gieo nguồn: t_first, độ dẫn của cầu, fit log-log
  plots.py         hình cho báo cáo
  config.py        cấu hình chung (RUN_ALL, HEAVY_TAIL, TWO_ROOM_SOURCE)
  webapp.py        server cục bộ + job runner + API + biểu đồ cho UI
  ui/              index.html, styles.css, app.js (không CDN)
scripts/           pipeline, kiểm tra đuôi nặng, hai buồng, chẩn đoán,
                   dựng PDF, chạy UI
tests/             kiểm thử đơn vị và tích hợp (gồm web API)
results/           CSV / NPZ / JSON sinh bởi pipeline (gitignored)
figures/           hình PNG / PDF (gitignored)
report/            report.md (nguồn) và report.pdf (tự động sinh)
```

## Bản đồ hình và bảng

| Đầu ra | Lệnh |
|---|---|
| Hình 1, sai lệch P-MCL vs MRD | `run_all.py --only reproduction` |
| Bảng 1, Hình 2, Hình 3 | `run_all.py --only graph` |
| Bảng 2 (mạng đuôi nặng) | `check_heavy_tail.py` |
| Hình 4 (gieo nguồn hai buồng) | `check_two_room.py` |
| Bảng 1c, Hình 5 (nhân rộng gieo nguồn) | `check_two_room_replicates.py` |
| CI / Cook's D / leave-one-out | `analyze_law.py` → `results/law_diagnostics.json` |

## Ghi chú

- `t90` dùng trung bình trượt 5 bước với đệm biên bằng giá trị đầu/cuối,
  tránh việc các seed đạt ngưỡng sát cuối cửa sổ bị đánh dấu censored.
- `relaxation_time` dùng nhánh dense cho `n <= 512` và `eigsh(which="LA")`
  cho `n > 512`, nên đúng cả với đồ thị bipartite.
- Kết quả trong báo cáo được sinh từ cùng một lần chạy với seed cố định;
  chạy lại có thể lệch ở mức làm tròn nhưng các kết luận định tính giữ nguyên.
- `results/`, `figures/` và `report/report.pdf` là đầu ra tự động sinh và
  đã được `.gitignore`, nên không bị commit vào repository.
