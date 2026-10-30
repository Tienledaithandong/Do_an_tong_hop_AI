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

## Kiểm thử

```bash
python -m pytest -q
```

`pytest.ini` đã thêm `src` vào `pythonpath`, nên không cần đặt biến môi
trường `PYTHONPATH` khi chạy test. `scripts/run_all.py` cũng tự thêm `src`
vào `sys.path`.

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
  graph_sweep.py   thí nghiệm B (khảo sát đồ thị)
  plots.py         hình cho báo cáo
scripts/           pipeline, kiểm tra đuôi nặng, dựng PDF
tests/             kiểm thử đơn vị
results/           CSV / NPZ sinh bởi pipeline
figures/           hình PNG / PDF
report/            report.md và report.pdf
```

## Bản đồ hình và bảng

| Đầu ra | Lệnh |
|---|---|
| Hình 1, sai lệch P-MCL vs MRD | `run_all.py --only reproduction` |
| Bảng 1, Hình 2, Hình 3 | `run_all.py --only graph` |
| Bảng 2 (mạng đuôi nặng) | `check_heavy_tail.py` |

## Ghi chú

- `t90` dùng trung bình trượt 5 bước với đệm biên bằng giá trị đầu/cuối,
  tránh việc các seed đạt ngưỡng sát cuối cửa sổ bị đánh dấu censored.
- `relaxation_time` dùng nhánh dense cho `n <= 512` và `eigsh(which="LA")`
  cho `n > 512`, nên đúng cả với đồ thị bipartite.
- Kết quả trong báo cáo được sinh từ cùng một lần chạy với seed cố định;
  chạy lại có thể lệch ở mức làm tròn nhưng các kết luận định tính giữ nguyên.
