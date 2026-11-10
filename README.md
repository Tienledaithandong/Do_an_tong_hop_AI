# HiveMindRL on graphs

Tái tạo kết quả của Soma và cộng sự (2026), *The hive mind is a single
reinforcement learning agent*, rồi đặt mô hình bầy đàn lên các đồ thị giao
tiếp cố định để xem cấu trúc đồ thị thay đổi tốc độ học như thế nào.

## Cài đặt

Code khai báo Python 3.10+ với dependency tối thiểu trong `requirements.txt`
(và `pyproject.toml`). Để tái lập môi trường đã kiểm thử, dùng **Python 3.13**
và bộ phiên bản cố định trong `requirements-lock.txt`:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

python -m pip install -r requirements-lock.txt
python -m pip check
```

Bộ pin hiện cần Python >= 3.12 (do NumPy/SciPy), không dùng được cho Python
3.10–3.11. Với các bản Python đó, dùng `python -m pip install -r requirements.txt`;
đó là môi trường theo ràng buộc tối thiểu, **không** phải bộ phiên bản đã khóa.
Metadata ghi phiên bản thực dùng để đối chiếu. Trên Windows, nếu terminal không
hỗ trợ tiếng Việt, đặt `PYTHONIOENCODING=utf-8` trước khi chạy.

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

Chế độ đầy đủ ghi vào `results/` và `figures/`; `run_all.py --quick` chỉ ghi
vào `results/quick/` và `figures/quick/`. Các script `check_*` có `--quick`
dùng hậu tố `_quick`, không ghi đè đầu ra đầy đủ. Không có `--quick` ở
`check_heavy_tail.py` và `check_two_room.py`.

Vẽ lại cũng chọn đúng chế độ: `run_all.py --quick --replot` đọc dữ liệu quick,
không mô phỏng lại. Chẩn đoán quick dùng
`python scripts/analyze_law.py --input results/quick/graph_sweep.csv`; JSON được
ghi cạnh CSV đầu vào, không đè `results/law_diagnostics.json` đầy đủ.

Kiểm tra bổ sung trên mạng đuôi nặng (Bảng 2 trong báo cáo):

```bash
python scripts/check_heavy_tail.py
```

Can thiệp gieo nguồn cho đồ thị hai buồng (Hình 4) và chẩn đoán thống kê cho
quy luật `t90 ~ L`:

```bash
python scripts/check_two_room.py      # ~30 s; ghi results/two_room_source*.csv (kèm RMST)
python scripts/analyze_law.py         # ghi results/law_diagnostics.json
```

Hiệu chuẩn quy luật `t90 ~ L` trên nhiều graph seed: **142 mẫu graph**
(140 lần vẽ ngẫu nhiên, 2 topology tất định; không chứng nhận mọi mẫu ngẫu nhiên
có ma trận kề khác nhau)
(10 graph × 30 seed mỗi cấu hình; riêng complete và WS `p = 0` là topology tất
định nên chạy 1 graph × 300 seed), tổng **4.800 lần chạy** với seed độc lập theo
từng (cấu hình, graph). Bootstrap phân tầng resample graph trong cấu hình rồi
cặp `(t90, hit)` trong mỗi graph — censoring được xử lý ngay trong từng resample —
để CI phản ánh cả biến thiên giữa các graph. CI này **điều kiện hoá trên tập cấu
hình cố định**; độ nhạy theo họ đồ thị xem `leave_one_family_out`:

```bash
python scripts/check_graph_seeds.py          # ~15–20 phút; Hình 8
python scripts/check_graph_seeds.py --quick  # vài giây
```

Bản nhân rộng của thí nghiệm này (nhiều đồ thị, nhiều `N`, tách thời gian
chờ ở cầu khỏi thời gian lan trong buồng, hồi quy theo độ dẫn `φ`):

```bash
python scripts/check_two_room_replicates.py          # ~20 phút; Hình 5
python scripts/check_two_room_replicates.py --quick  # vài giây
python scripts/check_two_room_replicates.py --refit  # tính lại fit từ CSV đã lưu
```

Tách `τ` khỏi độ dẫn của cầu bằng buồng thưa (đồ thị đều ngẫu nhiên bậc `d`
thay cho clique; cần chạy `check_two_room_replicates.py` trước vì kết quả
được gộp với lưới clique):

```bash
python scripts/check_two_room_sparse.py          # ~15 phút; Hình 6
python scripts/check_two_room_sparse.py --quick  # vài giây
python scripts/check_two_room_sparse.py --refit  # tính lại fit từ CSV đã lưu
```

Buồng không đều (Barabási–Albert) với đầu cầu gắn vào hub, lá hoặc nút ngẫu
nhiên của từng buồng, kèm đối chứng đều bậc 6:

```bash
python scripts/check_two_room_hetero.py          # ~8 phút; Hình 7
python scripts/check_two_room_hetero.py --quick  # vài giây
python scripts/check_two_room_hetero.py --refit  # tính lại fit từ CSV đã lưu
```

Độ nhạy của thời điểm đạt ngưỡng theo định nghĩa phép làm trơn (đối xứng /
nhân quả / thô): chạy lại các cấu hình của quy luật `t90 ~ L` và thí nghiệm
gieo nguồn, rồi đo lại ngưỡng trên *cùng* quỹ đạo theo ba định nghĩa để xem
kết luận nào nhạy với cách đo:

```bash
python scripts/check_threshold_sensitivity.py          # ~2 phút; Hình 9
python scripts/check_threshold_sensitivity.py --quick  # vài giây
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
- Kaplan–Meier / RMST: đường sống sót khi không censor, khi censor toàn bộ,
  khi ties và censor trộn nhau, và RMST bằng trung bình cắt ngọn;
- định nghĩa thời điểm vượt ngưỡng: phép làm trơn đối xứng vượt trước tín hiệu
  gốc, nhân quả/thô định vị đúng thời điểm xuất hiện, nhân quả không dùng dữ
  liệu tương lai, và mọi định nghĩa đều trả censoring khi không đạt ngưỡng;
- bootstrap phân tầng: khôi phục đúng đường thẳng khi dữ liệu tuyến tính chính
  xác, CI hữu hạn và có thứ tự khi nhiễu, xử lý censoring trong từng resample,
  và yêu tối thiểu 3 cấu hình; leave-one-family và `seed_offset` độc lập;
- vòng đời job của web API (tạo, chạy, lấy kết quả, trả PNG);
- validation đầu vào (N, bước, seed, họ đồ thị, tham số) và phục vụ UI tĩnh;
- endpoint `/api/results`, `/api/law` và chặn tên tệp ngoài danh sách cho phép;
- quick không đè full; replot không mô phỏng lại hoặc sửa dữ liệu/sidecar gốc;
- metadata cấu hình, phiên bản và hash source/đầu vào/đầu ra;
- inventory/checksum của ZIP, phát hiện sửa byte và coverage sai, không ghi đè ZIP.

Test tích hợp dùng `N` nhỏ và vài seed. Workflow `.github/workflows/tests.yml`
được cấu hình cho Ubuntu/Windows với Python 3.13, dependency khóa và GitHub
Actions pin theo commit SHA: `pip check`, pytest, pipeline quick và đóng gói/
xác minh quick. CI không chạy toàn bộ sweep đầy đủ hoặc chứng nhận số liệu
trong báo cáo; thời gian test phụ thuộc máy.

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
  graphs.py        sinh đồ thị (kể cả hai buồng clique/thưa), tau, đường đi trung bình
  metrics.py       t90 (đối xứng/nhân quả/thô), final mass, consensus, L2, Kaplan–Meier / RMST
  reproduce.py     thí nghiệm A (tái tạo)
  graph_sweep.py   thí nghiệm B (khảo sát đồ thị) + can thiệp hai buồng
  analysis.py      chẩn đoán thống kê cho quy luật t90 ~ L + bootstrap phân tầng
  bridge.py        nhân rộng gieo nguồn: t_first, độ dẫn của cầu, fit log-log
  hetero.py        buồng không đều: vị trí đầu cầu (hub/lá), Σ 1/bậc đầu cầu
  plots.py         hình cho báo cáo
  config.py        cấu hình chung (RUN_ALL, HEAVY_TAIL, TWO_ROOM_SOURCE, GRAPH_SEEDS)
  provenance.py    config, seed, phiên bản, git và SHA256 cho artifact mới
  webapp.py        server cục bộ + job runner + API + biểu đồ cho UI
  ui/              index.html, styles.css, app.js (không CDN)
scripts/           pipeline, kiểm tra đuôi nặng, hai buồng, chẩn đoán,
                   dựng PDF, chạy UI, package_artifacts.py (ZIP + checksum)
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
| Bảng 1d–1e, Hình 6 (buồng thưa, `τ` vs `cầu/bậc`) | `check_two_room_sparse.py` |
| Bảng 1f, Hình 7 (vị trí đầu cầu, hub/lá) | `check_two_room_hetero.py` |
| Bảng 1b′ (RMST gieo nguồn) | `check_two_room.py` → `results/two_room_source_summary.csv` |
| Hình 8, CI phân tầng nhiều graph seed | `check_graph_seeds.py` → `results/graph_seeds_law.json` |
| Hình 9, độ nhạy định nghĩa ngưỡng | `check_threshold_sensitivity.py` → `results/threshold_sensitivity_fit.json` |
| CI / Cook's D / leave-one-out | `analyze_law.py` → `results/law_diagnostics.json` |

## Provenance và gói bảo vệ

Mỗi thí nghiệm mới ghi `*_metadata.json` riêng (quick thêm `_quick`, hoặc nằm
trong thư mục quick): cấu hình thực dùng, lịch seed trong `extra`, timestamp UTC,
Python/thư viện, git SHA/dirty và SHA256 của source cùng đầu vào/đầu ra.
`run_all.py` ghi `reproduction_metadata.json` và `graph_sweep_metadata.json`.
Không dùng một sidecar “latest run” chung cho các thí nghiệm khác nhau.

- `--replot` giữ dữ liệu và metadata mô phỏng, ghi `*_render_metadata.json`.
- `--refit` giữ metadata mô phỏng, ghi `*_fits_metadata.json` với hash CSV đầu vào.
- Dựng PDF ghi `report/report_metadata.json` với hash Markdown, hình và PDF.
- Dữ liệu đầy đủ lịch sử chưa có metadata sản xuất **không được hồi tố** bằng
  config/code hiện tại. Chạy lại thí nghiệm đầy đủ mới tạo được provenance mới.
- Seed ở một số thí nghiệm được dùng lại giữa các ô; metadata mô tả điều này.
  Chỉ sweep nhiều graph hiện dùng block swarm seed không trùng giữa các ô/graph.

```bash
# Đóng gói snapshot đầy đủ hiện có; phải dựng PDF trước
python scripts/build_report.py
python scripts/package_artifacts.py --output out/defense-full.zip
python scripts/package_artifacts.py --verify out/defense-full.zip

# Gói kiểm tra nhanh, không lẫn kết quả đầy đủ
python scripts/run_all.py --quick
python scripts/package_artifacts.py --quick --output out/defense-quick.zip
python scripts/package_artifacts.py --verify out/defense-quick.zip
```

Bỏ `--output` để tên ZIP tự có timestamp; tên đã tồn tại sẽ bị từ chối thay vì
bị ghi đè. ZIP gồm source/test, dependency, cấu hình CI, README, nguồn báo cáo,
dữ liệu/hình theo chế độ; PDF báo cáo chỉ ở gói full. Không đóng gói `.env`,
venv, file scratch, ZIP cũ hay PDF/source upstream.

`manifest.json` chứa inventory, số byte, SHA256 và `provenance_coverage`:
artifact có hash khớp sidecar, artifact chưa quy thuộc, và sidecar có hash đầu ra
lỗi thời. `dependency_drift` báo riêng hash source/đầu vào lệch hoặc thiếu
trong ZIP; không sửa sidecar cũ hoặc phủ nhận rằng hash đầu ra vẫn khớp.
ZIP có thêm `.zip.sha256`; lệnh verify tính lại inventory, byte/hash, coverage,
dependency drift và hash ZIP nếu sidecar có mặt, không giải nén. Hash đầu ra
khớp **không** khẳng định source/đầu vào lịch sử trùng source đang đóng trong ZIP. Gói full có thể tổng hợp nhiều lần chạy; cảnh báo thiếu metadata là
chủ ý, không phải chứng nhận mọi kết quả do source hiện tại tạo ra.

Checksum chứng minh tính toàn vẹn byte so với manifest/sidecar, không phải chữ
ký tác giả, bằng chứng tính đúng khoa học hay kiểm định tái lập số học. Muốn
đối chiếu số liệu đầy đủ, chạy lại các lệnh trong bảng đầu ra và so metadata,
config cùng thống kê; không lấy gói quick làm kết quả bảo vệ.

## Đối chiếu upstream và giấy phép

Nguồn tham chiếu: [MISTLab/HiveMindRL](https://github.com/MISTLab/HiveMindRL),
revision `f24851db0a76e6bb2f917f7f7759ccd79476e7a3` ngày 26/10/2026; chưa xác nhận
đây là publication snapshot. Bảng chi tiết nằm ở §3.4 của `report/report.md`.

| Nội dung | Upstream được đối chiếu | Đồ án |
|---|---|---|
| Phạm vi | NumPy/PyTorch, streaming/parallel RL, nhiều luật và môi trường | NumPy/SciPy, P-CL/P-MCL, TRD/MRD, blind weighted voter |
| Thí nghiệm song song | Notebook: 150 bước, 1000 lần chạy; paper Table 1: 1000 iteration | 300 bước, 100 seed; không phải chạy lại toàn bộ upstream |
| Bandit/thời gian | Không clipping; đường ghi trước cập nhật | Clipping (không tác động cấu hình middle); đường `steps+1`, MRD Euler `dt=1` |
| Đồ thị | Lân cận lấy lại ngẫu nhiên | Đồ thị cố định, gieo nguồn và chẩn đoán cấu trúc là phần mở rộng |

Sai lệch 0,018 là so sánh P-MCL–MRD bên trong cài đặt này, **không** phải kết quả
chạy upstream-versus-local trên cùng mẫu. Fisher–KPP, vận tốc mặt trận không đổi
và xác suất trụ của từng lần xâm nhập là giả thuyết, không phải phép đo đã có.

Bài báo mang CC BY 4.0 với các ngoại lệ; repository upstream tại revision trên
không có giấy phép code rõ ràng. Không suy giấy phép code từ giấy phép bài báo.
Chưa thêm `LICENSE` địa phương vì chủ sở hữu chưa chọn phạm vi/giấy phép cho
đóng góp của họ; gói này không cấp lại quyền cho tài liệu hoặc code bên thứ ba.

## Ghi chú

- `t90` mặc định đọc lần vượt ngưỡng đầu tiên của **trung bình trượt đối xứng
  5 bước** (đệm biên bằng giá trị đầu/cuối, tránh việc seed đạt ngưỡng sát cuối
  cửa sổ bị đánh dấu censored). Vì cửa sổ đối xứng dùng cả dữ liệu tương lai,
  thời điểm này có thể sớm hơn tín hiệu gốc; `threshold_time(..., method=...)`
  hỗ trợ thêm `"causal"` (trung bình trượt đuôi, không nhìn trước) và `"raw"`
  (không làm trơn). `check_threshold_sensitivity.py` đo lại ngưỡng trên cùng
  quỹ đạo theo cả ba định nghĩa để kiểm tra độ nhạy của kết luận.
- `rmst` (Kaplan–Meier) tính thời gian đạt ngưỡng trung bình có tính đến
  censoring: seed không đạt ngưỡng đóng góp trọn thời gian quan sát, nên
  so sánh được giữa các cấu hình có tỉ lệ đạt khác nhau. Bị chặn trên bởi
  trần mô phỏng khi phần lớn seed không đạt ngưỡng.
- `check_graph_seeds.py` dùng bootstrap phân tầng (resample graph trong cấu
  hình, rồi cặp `(t90, hit)` trong graph) để CI của độ dốc phản ánh cả biến thiên
  giữa các graph và swarm seed trong tập cấu hình cố định. `analyze_law.py`
  resample các điểm cấu hình của mẫu một graph, nên đo một câu hỏi khác;
  không thể quy chênh lệch bề rộng hai CI hoàn toàn cho nhiễu vẽ graph.
  Censoring được xử lý trong từng resample; seed độc lập theo (cấu hình, graph);
  topology tất định (complete, WS `p = 0`) tính là một graph. CI điều kiện hoá
  trên tập cấu hình cố định — độ nhạy theo họ xem `leave_one_family_out`.
- `relaxation_time` dùng nhánh dense cho `n <= 512` và `eigsh(which="LA")`
  cho `n > 512`, nên trả về `λ₂` đại số (không phải `-1`) cả với đồ thị
  bipartite. `τ` là thời gian relaxation từ spectral gap; lazy walk có
  relaxation `2τ`, không phải bảo đảm thời gian trộn `2τ`. Bước đi không lazy
  vẫn có thể tuần hoàn trên đồ thị hai phía.
- Kết quả trong báo cáo đến từ nhiều thí nghiệm và lần chạy, không phải một
  snapshot duy nhất. Seed cố định giúp tái lập trong cùng môi trường; thay
  phiên bản thư viện hoặc code có thể đổi đồ thị và số liệu, nên cần đối chiếu
  metadata và checksum thay vì mặc định các kết luận luôn giữ nguyên.
- `results/`, `figures/`, `out/`, PDF báo cáo và sidecar PDF là đầu ra
  tự động sinh đã được `.gitignore`; muốn chia sẻ dữ liệu đầy đủ, dùng ZIP.
