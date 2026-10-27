# HiveMind trên đồ thị — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tái tạo lõi kết quả của paper HiveMindRL (Soma et al., Swarm Intelligence 2026) và trả lời câu hỏi mới: cấu trúc đồ thị giao tiếp ảnh hưởng thế nào tới khả năng học của bầy.

**Architecture:** Một thư viện Python nhỏ (6 module) cài đặt môi trường bandit, các luật cập nhật (TRD, MRD, P-CL, P-MCL), mô hình weighted voter cả well-mixed lẫn trên đồ thị, sinh đồ thị và tính spectral gap. Hai script thí nghiệm sinh dữ liệu CSV/NPZ, một script vẽ hình. Báo cáo tiếng Việt dựng từ chính các hình đó.

**Tech Stack:** Python 3.13, numpy, scipy (sparse), networkx, matplotlib, pandas, pytest.

**Spec:** Thiết kế đã chốt trong hội thoại ngày 27/10/2026 — tóm tắt ở mục "Thiết kế" dưới đây, đóng vai trò spec.

---

## Thiết kế (spec)

**Câu hỏi nghiên cứu.** Paper chứng minh Weighted Voter ≡ Maynard-Cross Learning dưới giả định well-mixed (mục 4.9.2 của paper ghi rõ spatial structure là future work). Khi đặt bầy lên đồ thị, giả định đó sụp. Câu hỏi: cấu trúc đồ thị thay đổi tốc độ và độ chính xác của việc học như thế nào?

**Giả thuyết.** Thời gian học tỉ lệ thuận với thời gian relaxation của đồ thị, `τ = 1/(1−λ₂)`, trong đó λ₂ là eigenvalue lớn thứ hai của ma trận bước đi ngẫu nhiên chuẩn hoá `S = D^{-1/2} A D^{-1/2}`. Hệ quả kiểm chứng được: `t90 ∝ τ` là đường thẳng trên log-log, và hai họ đồ thị khác nhau nằm chung một đường.

**Vì sao τ chứ không phải λ₂.** Với chu trình N=500, λ₂ = 0.9999 — trông gần hoàn hảo nhưng mixing cực chậm. τ mới mang ý nghĩa thời gian.

**Đo lường.** `t90` = bước đầu tiên mà trung bình trượt 5 bước của tỉ lệ chọn action tốt nhất đạt 0.9 (censored nếu không đạt). `final_best_mass`, `consensus` = max_a π_a, `l2_mrd` = RMS giữa đường học và đường MRD.

**Fallback đã định trước.** Nếu collapse theo τ không xảy ra, kết quả vẫn hợp lệ: báo cáo "τ chưa đủ", thêm hình t90 theo hệ số biến thiên bậc. Nếu thiếu thời gian, cắt họ RGG, giữ Watts–Strogatz.

---

## Global Constraints

- n = 10 action; Δ = 0.1; scenario mặc định `middle`, q cách đều trên [0.4, 0.7].
- Reproduction: N = 1000, T = 300, 100 seed. Graph: N = 500, T = 150, 100 seed.
- Reward luôn bị kẹp trong [0, 1] (paper yêu cầu `q_a ± Δ ⊆ [0,1]`).
- Mọi hình xuất cả `.pdf` và `.png` 300 dpi vào `figures/`.
- Vector hoá toàn bộ; không vòng lặp Python trên từng cá thể.
- Không dùng PyTorch, không GPU.

## Review Focus

Năm trường hợp dễ làm hỏng chương trình mà test thường bỏ sót:

1. Đồ thị mất kết nối (RGG dưới ngưỡng percolation) → λ₂ = 0, τ = inf; không được sinh NaN hay treo.
2. Node không có hàng xóm (CSR hàng rỗng) → tổng trọng số bằng 0; phải giữ nguyên type thay vì chia 0.
3. `t90` không bao giờ đạt ngưỡng → phải trả giá trị censored và được tính đúng khi lấy trung bình qua seed.
4. Một type tuyệt chủng (π_a = 0) → MRD/P-MCL không được chia 0 hay sinh NaN.
5. `q_a + Δ > 1` hoặc `q_a − Δ < 0` → reward phải được kẹp, và test phải chứng minh.

---

## Cấu trúc file

```
Đồ án AI - Tổng hợp/
├── s11721-026-00264-7.pdf
├── requirements.txt
├── pytest.ini
├── src/hive/
│   ├── __init__.py
│   ├── bandit.py
│   ├── models.py
│   ├── swarm.py
│   ├── graphs.py
│   ├── metrics.py
│   ├── reproduce.py
│   └── graph_sweep.py
├── scripts/run_all.py
├── tests/
├── results/
├── figures/
└── report/report.md
```

---

### Task 0: Khởi tạo dự án

**Files:** `.gitignore`, `requirements.txt`, `pytest.ini`, `src/hive/__init__.py`

**Interfaces:**
- Produces: repo git, môi trường Python có numpy/scipy/networkx/matplotlib/pandas/pytest, `pytest` chạy được.

- [ ] **Step 1:** Copy `s11721-026-00264-7.pdf` từ Downloads vào thư mục dự án.
- [ ] **Step 2:** `git init`, viết `.gitignore` (`.venv/`, `results/`, `figures/`, `__pycache__/`, `.superpowers/`).
- [ ] **Step 3:** Tạo `venv`, cài dependencies, ghi `requirements.txt`; `pytest.ini` trỏ `testpaths = tests`.
- [ ] **Step 4:** Chạy `pytest` → "no tests ran" (exit 5) là đúng.
- [ ] **Step 5:** Commit `chore: project scaffold`.

---

### Task 1: `bandit.py`

**Files:** Create `src/hive/bandit.py`, `tests/test_bandit.py`

**Interfaces:**
- Produces: `SCENARIOS: dict[str, tuple[float, float]]`; `make_q(scenario="middle", n=10) -> np.ndarray`; `sample_rewards(q, actions, delta, rng) -> np.ndarray`; `class Bandit` với `pull(actions) -> np.ndarray` và property `best -> int`.

- [ ] **Step 1: Viết test fail**

```python
def test_q_middle_is_evenly_spaced_between_04_and_07():
    q = make_q("middle", 10)
    assert np.allclose(np.diff(q), 0.3 / 9)

def test_rewards_stay_in_unit_interval_even_with_extreme_q():
    q = np.array([0.0, 1.0])
    r = sample_rewards(q, np.array([0, 1] * 5000), 0.3, np.random.default_rng(0))
    assert r.min() >= 0.0 and r.max() <= 1.0

def test_best_is_the_argmax_action():
    assert Bandit(make_q("middle")).best == 9
```

- [ ] **Step 2:** `pytest tests/test_bandit.py -v` → FAIL `ImportError`.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** `pytest tests/test_bandit.py -v` → 3 passed.
- [ ] **Step 5:** Commit `feat: bandit environment`.

---

### Task 2: `models.py`

**Files:** Create `src/hive/models.py`, `tests/test_models.py`

**Interfaces:**
- Consumes: `sample_rewards` từ Task 1.
- Produces: `trd_step(pi, q, dt=1.0)`, `mrd_step(pi, q, dt=1.0)`, `p_cl_step(pi, q, rng, B=1000, delta=0.1)`, `p_mcl_step(pi, q, rng, B=1000, delta=0.1)` — đều trả vector policy đã chuẩn hoá, không NaN.

- [ ] **Step 1: Viết test fail**

```python
def test_p_mcl_equals_reward_weighted_frequencies():
    # π'_a = Σ_{i:a_i=a} r_i / Σ_i r_i  (đóng của công thức paper)
    pi = np.full(10, 0.1); q = make_q("middle")
    rng = np.random.default_rng(1)
    new = p_mcl_step(pi, q, rng, B=5000)
    assert new.sum() == pytest.approx(1.0)
    assert np.all(new >= 0)

def test_mrd_concentrates_on_best_action():
    pi = np.full(10, 0.1); q = make_q("middle")
    for _ in range(100):
        pi = mrd_step(pi, q)
    assert pi.argmax() == 9 and pi[9] > 0.5

def test_mrd_survives_a_dead_type():
    pi = np.zeros(10); pi[0] = 1.0; q = make_q("middle")
    assert not np.isnan(mrd_step(pi, q)).any()
```

- [ ] **Step 2:** FAIL `ImportError`.
- [ ] **Step 3:** Implement (P-MCL rút gọn về `bincount(weights=rewards)/rewards.sum()`).
- [ ] **Step 4:** PASS.
- [ ] **Step 5:** Commit `feat: replicator dynamics and MCL update rules`.

---

### Task 3: `swarm.py`

**Files:** Create `src/hive/swarm.py`, `tests/test_swarm.py`

**Interfaces:**
- Produces: `well_mixed_voter(types, rewards, rng, n_types=None, M=None) -> np.ndarray`; `graph_voter(types, rewards, adj, rng, n_types=None) -> np.ndarray` với `adj` là `scipy.sparse.csr_matrix` không self-loop.

- [ ] **Step 1: Viết test fail**

```python
def test_equal_rewards_give_frequency_proportional_votes():
    types = np.array([0] * 30 + [1] * 10)
    rewards = np.ones(40)
    out = well_mixed_voter(types, rewards, np.random.default_rng(0))
    mass = np.bincount(out, minlength=2) / 40
    assert mass[0] == pytest.approx(0.75, abs=0.15)

def test_isolated_node_keeps_its_type():
    adj = sparse.csr_matrix((3, 3))
    types = np.array([1, 1, 1]); rewards = np.ones(3)
    assert np.array_equal(graph_voter(types, rewards, adj, np.random.default_rng(0)), types)

def test_graph_voter_only_produces_types_present_in_neighbourhood():
    # path 0-1-2, node 0 chỉ thấy node 1
    adj = sparse.csr_matrix(([1, 1], ([0, 1], [1, 2])), shape=(3, 3))
    adj = adj + adj.T
    types = np.array([0, 1, 2]); rewards = np.ones(3)
    out = graph_voter(types, rewards, adj, np.random.default_rng(0))
    assert out[0] == 1
```

- [ ] **Step 2:** FAIL.
- [ ] **Step 3:** Implement bằng `W = adj @ onehot*rewards`, sample theo hàng.
- [ ] **Step 4:** PASS.
- [ ] **Step 5:** Commit `feat: weighted voter on well-mixed and graph topologies`.

---

### Task 4: `graphs.py`

**Files:** Create `src/hive/graphs.py`, `tests/test_graphs.py`

**Interfaces:**
- Produces: `ws_graph(n, k, p, seed)`, `rgg_graph(n, radius, seed)` (trả CSR); `is_connected(adj) -> bool`; `relaxation_time(adj) -> tuple[float, float]` trả `(lambda2, tau)`, `tau = inf` khi mất kết nối; `degree_stats(adj) -> dict`.

- [ ] **Step 1: Viết test fail**

```python
def test_complete_graph_relaxes_in_about_one_step():
    _, tau = relaxation_time(complete_graph_adj(50))
    assert tau == pytest.approx(1.0, abs=0.1)

def test_cycle_relaxation_scales_quadratically():
    n = 200
    _, tau = relaxation_time(ws_graph(n, 2, 0.0, seed=0))
    assert 0.5 * n**2 / (2 * np.pi**2) < tau < 1.5 * n**2 / (2 * np.pi**2)

def test_disconnected_graph_returns_infinite_tau():
    adj = sparse.block_diag([complete_graph_adj(5), complete_graph_adj(5)], format="csr")
    lambda2, tau = relaxation_time(adj)
    assert np.isinf(tau)
```

- [ ] **Step 2:** FAIL.
- [ ] **Step 3:** Implement với `scipy.sparse.linalg.eigsh(..., k=2, which="LM")`.
- [ ] **Step 4:** PASS.
- [ ] **Step 5:** Commit `feat: graph generators and spectral relaxation time`.

---

### Task 5: `metrics.py`

**Files:** Create `src/hive/metrics.py`, `tests/test_metrics.py`

**Interfaces:**
- Produces: `best_mass(types, best) -> float`; `t90(curve, threshold=0.9, window=5) -> tuple[float, bool]` (giá trị, đã đạt hay censored); `consensus(pi) -> float`; `l2_to_reference(curve, reference) -> float`.

- [ ] **Step 1: Viết test fail**

```python
def test_t90_finds_first_crossing_of_smoothed_curve():
    curve = np.concatenate([np.linspace(0.1, 0.1, 10), np.full(10, 0.95)])
    t, hit = t90(curve, threshold=0.9, window=5)
    assert hit and 10 <= t <= 15

def test_t90_reports_censoring_when_never_reached():
    t, hit = t90(np.full(50, 0.5), threshold=0.9)
    assert (not hit) and t == 50

def test_consensus_is_max_policy_component():
    assert consensus(np.array([0.1, 0.7, 0.2])) == pytest.approx(0.7)
```

- [ ] **Step 2:** FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** PASS.
- [ ] **Step 5:** Commit `feat: learning-curve metrics`.

---

### Task 6: Thí nghiệm A — tái tạo

**Files:** Create `src/hive/reproduce.py`, `scripts/run_all.py`

**Interfaces:**
- Consumes: toàn bộ Task 1–5.
- Produces: `run_reproduction(n=10, N=1000, steps=300, seeds=100, scenario="middle") -> dict[str, np.ndarray]`; file `results/reproduction.npz`; hình `figures/fig1_reproduction.png`.

- [ ] **Step 1: Viết test fail** — MRD và P-MCL phải trùng nhau ở t=50 trong sai số 0.05 khi B=N.
- [ ] **Step 2:** FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Chạy `python scripts/run_all.py --only reproduction`; kiểm tra hình và file npz tồn tại; PASS.
- [ ] **Step 5:** Commit `feat: reproduction of well-mixed learning curves`.

---

### Task 7: Thí nghiệm B — quét đồ thị

**Files:** Create `src/hive/graph_sweep.py`, `figures` script

**Interfaces:**
- Consumes: Task 1–5.
- Produces: `run_graph_sweep(...) -> pandas.DataFrame`; `results/graph_sweep.csv`; `figures/fig2_law.png`, `fig3_collapse.png`, `fig4_trajectories.png`.

- [ ] **Step 1: Viết test fail** — với đồ thị đầy đủ (τ≈1), `t90` phải xấp xỉ well-mixed.
- [ ] **Step 2:** FAIL.
- [ ] **Step 3:** Implement quét WS (`p ∈ {0, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0}`, k=6) và RGG (`radius = c·sqrt(log N/(πN))`, c ∈ [0.8, 2.0]).
- [ ] **Step 4:** Chạy sweep, xuất CSV + 3 hình; PASS.
- [ ] **Step 5:** Commit `feat: graph structure experiment and law figure`.

---

### Task 8: Báo cáo

**Files:** Create `report/report.md`, `report/figures/`

- [ ] **Step 1:** Viết báo cáo tiếng Việt theo cấu trúc: Giới thiệu → Cơ sở lý thuyết → Tái tạo kết quả → Mở rộng đồ thị → Kết quả & thảo luận → Hạn chế → Tài liệu tham khảo.
- [ ] **Step 2:** Chèn hình từ `figures/`, kiểm tra đọc được khi in đen trắng.
- [ ] **Step 3:** Xuất PDF.
- [ ] **Step 4:** Commit `docs: report`.

---

## Lịch

| Ngày | Việc | Sản phẩm |
|---|---|---|
| 27/10 | Task 0–3 | Môi trường + 3 module lõi có test |
| 28/10 | Task 4–6 | Hình tái tạo đầu tiên |
| 29/10 | Task 7 | Hình quy luật + dữ liệu sweep |
| 30/10 | Task 8 | Báo cáo gần xong |
| 31/10 | Buffer | Đọc lại, đóng gói, nộp |

## Rủi ro & fallback

- **Collapse không xảy ra** → báo cáo "τ chưa đủ"; thêm hình t90 theo hệ số biến thiên bậc. Không mất gì.
- **WS p nhỏ làm λ₂ ≈ 1** → đã xử lý bằng τ và quét p theo log.
- **Chậm tiến độ** → cắt RGG, giữ WS.
- **Kết quả không đẹp** → báo cáo hiện tượng kẹt ở action tầm trung khi đồ thị phân mảnh; đó là nội dung, không phải lỗi.

## Tiêu chí hoàn thành

1. `python scripts/run_all.py` chạy sạch từ đầu đến cuối, sinh đủ 4 hình.
2. `pytest` xanh.
3. Báo cáo có một hình trả lời được: *đồ thị càng dễ trộn thì bầy học càng nhanh — nhanh bao nhiêu và theo quy luật nào.*
