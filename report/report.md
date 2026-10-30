# Bầy ong ra quyết định trên đồ thị: tái tạo và mở rộng HiveMindRL

**Đồ án tổng hợp — Trí tuệ nhân tạo**

*Ngày 30 tháng 10 năm 2026*

---

## Tóm tắt

Bài báo *The hive mind is a single reinforcement learning agent* (Soma và cộng sự, 2026) chứng minh rằng một đàn cá thể chỉ bắt chước lẫn nhau vẫn tạo ra hành vi học tập tương đương một tác tử học tăng cường duy nhất, với điều kiện quần thể được trộn đều. Đồ án này gồm hai phần. Phần thứ nhất tái tạo kết quả cốt lõi của bài báo bằng NumPy: đường học của mô hình bầy đàn, của thuật toán Maynard-Cross Learning song song, và của đường MRD rời rạc trùng nhau với sai lệch lớn nhất 0,018. Phần thứ hai đặt đàn lên một đồ thị giao tiếp cố định — đúng phần mà bài báo để ngỏ — và hỏi cấu trúc đồ thị thay đổi tốc độ học như thế nào. Khảo sát 22 lần chạy trên Watts–Strogatz, random geometric graph và một họ can thiệp hai buồng cho thấy giả thuyết ban đầu (thời gian hội tụ tỉ lệ với thời gian relaxation `τ = 1/(1−λ₂)` của đồ thị) chỉ đúng trong từng họ. Một đại lượng đơn giản hơn — đường đi trung bình `L` — mô tả được cả WS lẫn RGG bằng một quy luật tuyến tính: `t₉₀ = 44,85 + 3,46·L` với `R² = 0,91` trên 16 cấu hình không can thiệp. Ba cấu hình hai buồng là điểm kiểm định độc lập: `L ≈ 2,0` và `t₉₀ = 47,5`, lệch −4,2 bước so với dự đoán, trong khi `τ` tăng sáu lần. Tương quan hạng với `τ` rơi từ 0,69 xuống 0,10, còn với `L` tăng từ 0,82 lên 0,89. Kết quả này phù hợp với trực giác rằng bắt chước có trọng số là một quá trình có thiên lệch, lan truyền theo khoảng cách chứ không khuếch tán theo thời gian trộn.

## 1. Giới thiệu

Một đàn ong chọn tổ mới không có ai chỉ huy. Mỗi con khảo sát một địa điểm, đánh giá chất lượng, rồi truyền thông tin cho những con khác. Cuối cùng cả đàn thường đồng thuận về địa điểm tốt nhất. Hiện tượng này — trí tuệ bầy đàn — đặt một câu hỏi lý thuyết rõ ràng: hành vi học tập ở cấp tập thể có thể được mô tả bằng một thuật toán học tăng cường ở cấp cá nhân không?

Soma và cộng sự (2026) trả lời là có, trong giới hạn của mô hình. Họ chứng minh hai mệnh đề. Mệnh đề thứ nhất chỉ ra rằng quy tắc *bắt chước cá thể thành công* tương đương với Taylor Replicator Dynamics, và do đó tương đương với Cross Learning. Mệnh đề thứ hai chỉ ra rằng *weighted voter model* — nơi mỗi cá thể quảng bá lựa chọn của mình với cường độ tỉ lệ với phần thưởng nhận được — tương đương với một thuật toán mới do họ đề xuất, Maynard-Cross Learning (MCL). MCL khác Cross Learning ở chỗ chia tín hiệu phần thưởng cho giá trị kỳ vọng của chính sách hiện tại, nhờ đó tín hiệu được chuẩn hoá theo chất lượng chung của hệ thống.

Điểm yếu của chứng minh là giả định **trộn đều**: mỗi cá thể được coi là quan sát được toàn bộ quần thể, hoặc một vùng lân cận ngẫu nhiên đủ lớn. Chính bài báo thừa nhận điều này ở mục 4.9.2: *"The well-mixed assumption of Proposition 2 may not hold in the presence of spatial structure… We leave the investigation of spatial structure and heterogeneity in our framework for future work."*

Đồ án này đi vào đúng khoảng trống đó. Câu hỏi nghiên cứu:

> Khi đàn được đặt lên một đồ thị giao tiếp cố định và mỗi cá thể chỉ quan sát hàng xóm của mình, cấu trúc đồ thị thay đổi tốc độ và độ chính xác của việc học như thế nào?

Đóng góp của đồ án gồm ba phần:

1. Một bản cài đặt độc lập, có kiểm thử, tái tạo được kết quả cốt lõi của bài báo (Hình 1).
2. Một khảo sát thực nghiệm trên 22 lần chạy thuộc hai họ ngẫu nhiên (WS, RGG) và một họ can thiệp (hai buồng), với 100 seed mỗi cấu hình (Hình 2, 3; Bảng 1).
3. Một phát hiện: thời gian hội tụ tuân theo **đường đi trung bình** của đồ thị, không phải thời gian relaxation phổ, kèm thí nghiệm tách biến trên đồ thị hai buồng và một lý giải cho hiện tượng đó.

## 2. Cơ sở lý thuyết

### 2.1. Multi-armed bandit

Môi trường gồm `n` hành động. Mỗi hành động `a` có chất lượng ẩn `qₐ`. Khi hành động được chọn, môi trường trả về phần thưởng nhiễu

```
rₐ ~ U(qₐ − Δ, qₐ + Δ)
```

Trong đồ án, `n = 10`, `Δ = 0,1`, và `qₐ` cách đều trên đoạn `[0,4; 0,7]` (kịch bản *middle* của bài báo). Phần thưởng luôn được kẹp vào `[0, 1]` như bài báo yêu cầu.

### 2.2. Cross Learning và Maynard-Cross Learning

Với chính sách `π` và hành động `k` vừa chọn cùng phần thưởng `r_k`:

```
Cross Learning:            π'ₐ = πₐ + α·r_k·(1[a=k] − πₐ)
Maynard-Cross Learning:    π'ₐ = πₐ + α·(r_k / v_π)·(1[a=k] − πₐ)
```

trong đó `v_π = Σₐ πₐqₐ`. Lấy kỳ vọng theo `k ~ π` cho ra hai phương trình vi phân:

```
TRD:  π̇ₐ = πₐ(qₐ − v_π)
MRD:  π̇ₐ = πₐ(qₐ − v_π)/v_π
```

Bản song song của MCL — dùng `B` mẫu hành động/phần thưởng trong một bước — có một dạng rút gọn đáng chú ý. Với `α = 1/B`:

```
π'ₐ = Σ_{i: aᵢ = a} rᵢ / Σᵢ rᵢ
```

Tức là chính sách mới đúng bằng **tần suất hành động có trọng số phần thưởng**. Đây cũng chính là công thức cập nhật của weighted voter. Sự trùng khớp này là nội dung thực chất của Mệnh đề 2, và là lý do hai mô hình cho ra cùng đường học.

### 2.3. Weighted voter model

Mỗi cá thể `i` giữ một lựa chọn `Tᵢ`, nhận phần thưởng `rᵢ`, và bắt chước một hàng xóm. Xác suất chọn phương án `a` tỉ lệ với tổng phần thưởng của những hàng xóm đang chọn `a`:

```
v⁽ⁱ⁾ₐ = Σ_{j ∈ N(i), Tⱼ = a} rⱼ / Σ_{j ∈ N(i)} rⱼ
```

Trong mô hình well-mixed, `N(i)` là toàn bộ quần thể trừ chính `i`. Trong mô hình đồ thị, `N(i)` là tập hàng xóm cố định của `i`.

### 2.4. Hai thước đo cấu trúc đồ thị

**Thời gian relaxation.** Gọi `S = D^{-1/2} A D^{-1/2}` là ma trận kề chuẩn hoá của đồ thị, `λ₂` là eigenvalue lớn thứ hai. Đại lượng

```
τ = 1/(1 − λ₂)
```

là thang thời gian của quá trình khuếch tán trên đồ thị. `τ ≈ 1` tương ứng quần thể trộn đều; `τ` lớn tương ứng thông tin lan chậm. Lưu ý `λ₂` một mình không đủ: với vòng lattice 500 đỉnh bậc 6 (WS `p = 0`), `λ₂ ≈ 0,9996` trông gần hoàn hảo nhưng `τ ≈ 2714` bước.

**Đường đi trung bình.** `L` là khoảng cách ngắn nhất trung bình trên mọi cặp đỉnh liên thông. Đây là thang *khoảng cách* của đồ thị, không phải thang *thời gian trộn*.

## 3. Phương pháp

### 3.1. Cài đặt

Toàn bộ mô phỏng viết bằng Python với NumPy và SciPy, vector hoá theo từng bước quyết định, không dùng GPU. Mã nguồn gồm bảy mô-đun tính toán: môi trường bandit, các luật cập nhật chính sách, mô hình bầy đàn, sinh đồ thị và đại lượng phổ, đo lường, hai kịch bản thí nghiệm, và mã vẽ hình. Mỗi mô-đun có kiểm thử đơn vị; các kiểm thử được viết trước phần cài đặt.

Một điểm kỹ thuật đáng nêu: bước cập nhật trên đồ thị được tính bằng một phép nhân thưa

```
W = A · (onehot(T) ⊙ r)
```

cho ra ma trận `N × n` trọng số phiếu của từng cá thể, sau đó lấy mẫu theo hàng. Nhờ vậy chi phí mỗi bước là `O(|E|·n)` thay vì `O(N²)`, và toàn bộ khảo sát 22 lần chạy × 100 seed × 400 bước chạy trong khoảng 4–5 phút trên CPU thường.

### 3.2. Tham số

**Thí nghiệm A (tái tạo).** `N = 1000` cá thể, `n = 10` phương án, `T = 300` bước, 100 seed, `B = N = 1000` mẫu song song cho P-MCL. Mọi phương pháp xuất phát từ cùng một phân bố đều. Đây là lựa chọn tiết kiệm compute: bài báo dùng 1000 iteration và 1000 seed [1, Table 1], nhưng cả ba đường đã hội tụ trước bước 100 nên 300 bước là đủ để so sánh; các kết luận định tính không phụ thuộc phần đuôi.

**Thí nghiệm B (đồ thị).** `N = 500`, `T = 400` bước, 100 seed, cùng bandit. Hai họ ngẫu nhiên và một họ can thiệp:

- **Watts–Strogatz** với bậc `k = 6`, quét `p ∈ {0; 10⁻⁴; 10⁻³; 0,01; 0,05; 0,1; 0,2; 0,4; 0,7; 1}`. `p = 0` là vòng lattice bậc 6 (mỗi nút nối với 6 láng giềng gần nhất), `p = 1` là đồ thị ngẫu nhiên.
- **Random geometric graph** trên hình vuông đơn vị, bán kính `radius = c·√(log N / (πN))`, quét `c ∈ {1,0; 1,2; …; 2,5}`. Hai giá trị đầu nằm dưới ngưỡng liên thông.
- **Hai buồng.** Hai clique 250 nút nối với nhau bằng một cầu hẹp 1, 3 hoặc 6 cạnh. Đây là họ can thiệp có chủ đích, không phải một họ ngẫu nhiên.

Mỗi cấu hình dùng **một đồ thị cố định** (seed 0) và 100 seed cho phần ngẫu nhiên của quá trình học. Cách này giữ `τ` và `L` cố định trong từng cấu hình, nhờ đó biến thiên quan sát được quy cho động lực học chứ không phải cho việc lấy mẫu đồ thị. Tham chiếu well-mixed dùng đồ thị đầy đủ (`τ ≈ 1`).

### 3.3. Đo lường

- `t₉₀`: bước đầu tiên mà trung bình trượt 5 bước của tỉ lệ cá thể chọn phương án tốt nhất đạt 0,9. Trung bình trượt dùng đệm biên bằng giá trị đầu/cuối, để các seed đạt ngưỡng sát cuối cửa sổ không bị tính nhầm là censored. Nếu không đạt trong `T` bước, giá trị được đánh dấu *censored* và báo cáo riêng thay vì gán bằng `T`.
- `final_best_mass`: trung bình tỉ lệ chọn phương án tốt nhất trong 10% bước cuối.
- `consensus`: khối lượng của phương án phổ biến nhất ở bước cuối.
- `l2_mrd`: khoảng cách RMS giữa đường học thực nghiệm và đường MRD.

## 4. Kết quả tái tạo

Hình 1 so sánh ba đường học trong 300 bước, trung bình trên 100 seed, dải tô là ±1 độ lệch chuẩn.

![Hình 1. Đường học của bầy đàn, P-MCL và đường MRD rời rạc.](figures/fig1_reproduction.png)

Ba đường gần như trùng nhau. Ở bước 50, tỉ lệ chọn phương án tốt nhất lần lượt là 0,916 (bầy đàn), 0,909 (P-MCL) và 0,914 (MRD). Sai lệch tuyệt đối lớn nhất giữa đường trung bình của P-MCL và MRD, tính từ bước 10 trở đi, là **0,018**.

Kết quả này tái tạo đúng kết luận của bài báo: khi số mẫu song song đủ lớn, cập nhật MCL bám sát nghiệm replicator dynamics, và mô hình bầy đàn cho ra cùng đường học. Đây là bằng chứng thực nghiệm cho Mệnh đề 2. Đường MRD ở đây là nghiệm rời rạc `δ = 1` của phương trình replicator, đúng như cách bài báo mô phỏng, không phải nghiệm giải tích liên tục.

## 5. Kết quả trên đồ thị

Bảng 1 tóm tắt 19 topology liên thông phân biệt. Tổng cộng có 22 lần chạy: 20 cấu hình liên thông, trong đó `p = 10⁻⁴` trùng hoàn toàn với `p = 0` nên chỉ còn 19 topology phân biệt; hai cấu hình RGG dưới ngưỡng liên thông có `τ = ∞` nên không đưa vào bảng, nhưng vẫn được mô phỏng và báo cáo ở phần thảo luận.

**Bảng 1. Kết quả khảo sát đồ thị (N = 500, 400 bước, 100 seed).**

| Họ | Tham số | τ | L | t₉₀ | seed đạt t₉₀ | final | seed final ≥ 0,95 |
|---|---|---|---|---|---|---|---|
| đầy đủ | — | 1,0 | 1,0 | 44,0 | 1,00 | 1,00 | 1,00 |
| RGG | c = 2,5 | 34,0 | 4,16 | 48,0 | 1,00 | 1,00 | 1,00 |
| RGG | c = 2,2 | 48,7 | 4,76 | 51,5 | 1,00 | 1,00 | 1,00 |
| RGG | c = 2,0 | 62,6 | 5,26 | 56,5 | 1,00 | 1,00 | 1,00 |
| RGG | c = 1,8 | 83,8 | 6,02 | 58,0 | 1,00 | 1,00 | 1,00 |
| RGG | c = 1,6 | 112,8 | 6,85 | 61,5 | 1,00 | 1,00 | 1,00 |
| RGG | c = 1,4 | 173,9 | 8,08 | 73,0 | 1,00 | 1,00 | 1,00 |
| WS | p = 1,0 | 3,65 | 3,69 | 58,5 | 1,00 | 1,00 | 1,00 |
| WS | p = 0,7 | 3,88 | 3,71 | 58,5 | 1,00 | 1,00 | 1,00 |
| WS | p = 0,4 | 5,50 | 3,92 | 55,5 | 1,00 | 1,00 | 1,00 |
| WS | p = 0,2 | 13,2 | 4,51 | 61,0 | 1,00 | 1,00 | 1,00 |
| WS | p = 0,1 | 31,1 | 5,40 | 70,0 | 1,00 | 1,00 | 1,00 |
| WS | p = 0,05 | 54,4 | 6,43 | 77,5 | 1,00 | 1,00 | 1,00 |
| WS | p = 0,01 | 281 | 11,26 | 110,0 | 1,00 | 1,00 | 1,00 |
| WS | p = 10⁻³ | 1878 | 26,09 | 155,5 | 0,96 | 1,00 | 0,89 |
| WS | p = 0 | 2714 | 42,08 | 173,5 | 0,90 | 1,00 | 0,82 |
| hai buồng | cầu = 6 | 5.228 | 1,98 | 47,5 | 1,00 | 1,00 | 1,00 |
| hai buồng | cầu = 3 | 10.458 | 1,99 | 47,5 | 1,00 | 1,00 | 1,00 |
| hai buồng | cầu = 1 | 31.374 | 2,00 | 47,5 | 1,00 | 1,00 | 1,00 |

`p = 10⁻⁴` trùng hoàn toàn với `p = 0` (cùng đồ thị sinh ra) nên chỉ trình bày một lần. `final` là trung vị của `final_best_mass`; cột cuối là tỉ lệ seed có `final_best_mass ≥ 0,95`.

Hình 2 đặt hai thước đo cạnh nhau.

![Hình 2. (a) Thời gian hội tụ theo thời gian relaxation. (b) Thời gian hội tụ theo đường đi trung bình; đường thẳng hiệu chuẩn trên 16 cấu hình không can thiệp, ba điểm hai buồng là kiểm định held-out.](figures/fig2_law.png)

**Giả thuyết ban đầu không đứng vững ở dạng phổ quát.** Ở đồ thị (a), mỗi họ có xu hướng tăng theo `τ`, nhưng hai họ nằm ở hai đường khác nhau: tại cùng `τ ≈ 100`, RGG cần khoảng 60 bước còn WS cần khoảng 90 bước (nội suy). Hệ số tương quan hạng Spearman giữa `t₉₀` và `τ` là 0,69 khi chỉ xét 16 cấu hình không can thiệp (đầy đủ + WS + RGG), nhưng rơi xuống 0,10 khi thêm họ hai buồng.

**Đường đi trung bình mô tả được cả WS lẫn RGG.** Ở đồ thị (b), các điểm của cả WS lẫn RGG nằm quanh một đường thẳng, hiệu chuẩn trên 16 cấu hình không can thiệp:

```
t₉₀ = 44,85 + 3,46 · L,     R² = 0,91
```

Hệ số tương quan hạng với `L` là 0,82 trên tập hiệu chuẩn và 0,89 khi thêm hai buồng. Đáng chú ý là tung độ gốc 44,85 xấp xỉ `t₉₀ = 44,0` đo được trên đồ thị đầy đủ — nghĩa là có thể đọc quy luật này như **thời gian học nội tại của đàn cộng với thời gian thông tin đi hết đồ thị**.

Hình 3 cho thấy hình dạng đường học thay đổi thế nào khi nới dần vòng lattice.

![Hình 3. Đường học trung bình trên một số cấu hình tiêu biểu, kể cả vòng lattice p = 0.](figures/fig3_trajectories.png)

Đồ thị đầy đủ và MRD gần như trùng nhau. Khi `p` giảm, đường học kéo dài về phía phải nhưng vẫn tiến tới 1. Với vòng lattice (`p = 0`), trung vị `final` vẫn bằng 1,0 nhưng chỉ 82% seed đạt 0,95; đường trung bình đạt khoảng 0,97 ở bước 400, chậm hơn khoảng bốn lần so với đồ thị đầy đủ.

### 5.1. Độ chính xác: trung vị che mất phần đuôi

Cột `final` (trung vị của `final_best_mass`) bằng 1,00 ở **mọi** cấu hình, kể cả vòng lattice — một giá trị bão hoà, không phân biệt được cấu hình nhanh với cấu hình chậm. Vì vậy Bảng 1 thêm cột `seed final ≥ 0,95`: tỉ lệ seed mà khối lượng phương án tốt nhất ở cuối đạt ít nhất 0,95.

Đọc cột đó mới thấy bức tranh thật. Ở 17/19 topology liên thông, **100% seed** đạt 0,95. Nhưng ở hai topology chậm nhất, phần đuôi hiện ra: vòng lattice (`p = 0`) chỉ có **0,82** seed đạt 0,95, và `p = 10⁻³` có **0,89**. Đây đúng là hai cấu hình có tỉ lệ seed đạt `t₉₀ = 0,9` trong 400 bước thấp nhất (0,90 và 0,96) — cùng một nhóm seed "chậm" bị đếm qua hai thước đo khác nhau.

Kết luận cần thận trọng hơn: **cấu trúc đồ thị ảnh hưởng mạnh tới tốc độ, và ảnh hưởng rõ rệt tới độ chính xác chỉ ở những cấu hình mà thời gian học tiến sát cửa sổ 400 bước.** Với 17 topology còn lại, độ chính xác cuối kỳ thực sự không đổi; hai đồ thị RGG bị phân mảnh (không liên thông) cũng đạt 0,88–1,00 vì mỗi thành phần liên thông vẫn đủ lớn để tự học đúng. Nói cách khác, trong dải khảo sát, đồ thị **mua được độ trễ, không mua được độ đúng** — trừ khi độ trễ chạm trần mô phỏng, và khi đó cái mất là vài phần trăm seed, không phải toàn bộ quần thể.

### 5.2. Thí nghiệm tách biến: hai buồng nối bằng cầu hẹp

Ba dòng cuối Bảng 1 là một thí nghiệm can thiệp có chủ đích, không phải một họ đồ thị ngẫu nhiên nữa. Hai clique 250 nút nối với nhau bằng một cầu hẹp (1, 3 hoặc 6 cạnh) tách rời hai đại lượng mà WS và RGG luôn làm biến thiên cùng chiều:

- **`τ` bị thổi phồng.** Nút cổ chai khiến bước đi ngẫu nhiên phải chờ rất lâu mới trộn giữa hai buồng, nên `τ` nhảy từ 5.228 (cầu 6) lên 10.458 (cầu 3) rồi 31.374 (cầu 1) — tăng sáu lần, và lớn hơn đồ thị đầy đủ gần bốn bậc.

- **`L` gần như đứng yên.** Trong mỗi buồng mọi cặp nút cách nhau đúng một bước, cộng thêm một bước qua cầu, nên `L ≈ 2,0` cho cả ba cầu.

- **`t₉₀` đứng yên theo `L`.** Cả ba cấu hình đều cho **47,5** bước, sai khác dưới 0,5 bước, dù `τ` khác nhau sáu lần.

Đường hiệu chuẩn trên 16 cấu hình không can thiệp là `44,85 + 3,46·L` (`R² = 0,91`). Nó dự đoán ba cấu hình hai buồng khoảng 51,7 bước; thực đo 47,5, lệch −4,2 bước (khoảng 8%). Sai lệch này nhỏ so với việc `τ` thay đổi sáu lần, và quan trọng hơn, `t₉₀` không hề tăng theo `τ`. Tương quan hạng của `t₉₀` với `τ` sụp từ 0,69 xuống **0,10**, còn với `L` tăng từ 0,82 lên **0,89**. Đây là bằng chứng trực tiếp, không phải suy luận từ tương quan, rằng thời gian học bám theo **khoảng cách thông tin**, không phải thời gian trộn phổ. Cơ chế được bàn ở §6.

### 5.3. Kiểm tra ngoài phạm vi: mạng đuôi nặng

Ba mạng ngoài phạm vi quét chính được chạy với cùng `N = 500`, 400 bước, 100 seed: Barabási–Albert `m = 2` (đuôi nặng), BA `m = 5` (đuôi nhẹ hơn, dày hơn), và Erdős–Rényi bậc trung bình 6 (đuôi Poisson, điểm neo). Bảng 2 so `t₉₀` đo được với dự đoán từ đường hiệu chuẩn `44,85 + 3,46·L`.

**Bảng 2. Kiểm tra trên mạng ngoài phạm vi quét chính (N = 500, 400 bước, 100 seed).**

| Mạng | τ | L | t₉₀ | dự đoán | residual |
|---|---|---|---|---|---|
| BA m = 2 | 5,66 | 3,79 | 59,0 | 58,0 | +1,0 |
| BA m = 5 | 2,32 | 2,77 | 51,0 | 54,4 | −3,4 |
| ER bậc 6 | ∞ | 3,65 | 58,0 | 57,5 | +0,5 |

Quy luật `L` vẫn giữ trong sai số ±3,4 bước. BA `m = 2` lệch dương, tức mạng đuôi nặng có xu hướng chậm hơn dự đoán; ba điểm chưa đủ để khẳng định quy luật phổ quát cho mọi mạng đuôi nặng. Cần quét có hệ thống theo hệ số biến thiên bậc (degree CV) để kết luận. ER bậc 6 có `τ = ∞` vì đồ thị sinh ra bị phân mảnh, nhưng `L` vẫn hữu hạn trên các cặp đến được.

## 6. Thảo luận

Vì sao đường đi trung bình lại dự báo tốt hơn thời gian relaxation?

Spectral gap đo mode chậm nhất của một quá trình **khuếch tán** — một hạt ngẫu nhiên đi lang thang cho tới khi mất tương quan. Nhưng bắt chước có trọng số không phải quá trình khuếch tán thuần: phương án có phần thưởng cao được sao chép với xác suất cao hơn, tạo ra một **dòng trôi có hướng**. Dưới tác dụng của dòng trôi, kiểu tốt nhất lan ra như một mặt trận chuyển động với vận tốc xấp xỉ không đổi, nên thời gian để nó phủ 90% quần thể bị chi phối bởi **khoảng cách phải đi**, tức `L`, chứ không phải bởi thời gian trộn của bước đi ngẫu nhiên.

Cách giải thích này khớp với ba quan sát. Thứ nhất, tung độ gốc của quy luật xấp xỉ thời gian học của quần thể trộn đều — phần "không phụ thuộc đồ thị" đúng là động lực học chọn lọc thuần tuý. Thứ hai, hai họ đồ thị có cùng `L` nhưng khác số chiều (vòng 1D so với mặt phẳng 2D) lại nằm sát nhau hơn nhiều so với khi so theo `τ`. Thứ ba, đường học trên vòng lattice có dạng kéo dài chứ không phải trễ rồi bùng nổ — dấu hiệu của lan truyền theo khoảng cách.

Thí nghiệm tách biến ở §5.2 là phép kiểm định sạch cho lý giải này. Họ hai buồng giữ `L` cố định ở 2,0 trong khi tăng `τ` sáu lần (và lớn hơn đồ thị đầy đủ gần bốn bậc); nếu thời gian học do trộn phổ quyết định, `t₉₀` phải tăng theo `τ` — nhưng nó đứng yên ở 47,5. Giả thuyết cơ chế là **lan truyền theo mặt trận**: dưới dòng trôi thiên lệch, mật độ cá thể chọn phương án tốt nhất thoả một phương trình kiểu Fisher–KPP [7, 8], có nghiệm sóng chạy với vận tốc không đổi; thời gian phủ 90% quần thể khi đó tỉ lệ với quãng đường mặt trận phải đi, tức `L`, chứ không phải với nghịch đảo spectral gap. Nút cổ chai chỉ làm chậm **khuếch tán**, không làm chậm **mặt trận** đã có hướng — đó là lý do hai đại lượng tách nhau trong họ hai buồng. Đây mới là giả thuyết cơ chế, chưa phải kết quả đo trực tiếp vận tốc mặt trận; §7 nêu rõ giới hạn này.

## 7. Hạn chế

1. **Censoring.** Hai cấu hình chậm nhất có 4–10% seed không đạt 0,9 trong 400 bước, nên `t₉₀` trung vị bị lệch xuống. Đã báo cáo riêng tỉ lệ đạt thay vì che đi.
2. **Cùng một kịch bản chất lượng.** Thí nghiệm đồ thị chỉ dùng `q ∈ [0,4; 0,7]`. Với dải chất lượng thấp hơn, nhiễu tương đối lớn hơn và đàn dễ kẹt ở phương án tầm trung.
3. **Mạng bậc nặng đuôi mới chỉ khảo sát sơ bộ.** Mục 5.3 kiểm tra thêm ba mạng ngoài phạm vi quét chính (BA `m = 2`, BA `m = 5`, ER bậc 6) với 100 seed: quy luật `L` vẫn giữ, nhưng sai lệch so với đường hiệu chuẩn là −3,4…+1,0 bước và BA `m = 2` lệch **dương**, tức ba điểm chưa đủ để khẳng định quy luật phổ quát cho mọi mạng đuôi nặng. Cần quét có hệ thống theo hệ số biến thiên bậc để kết luận.
4. **Không có cơ chế khám phá.** Như bài báo gốc, khi một phương án biến mất khỏi quần thể thì nó không thể quay lại. Trong môi trường thay đổi theo thời gian, đàn sẽ không thích nghi được.
5. **`t₉₀` phụ thuộc ngưỡng và cửa sổ làm trơn.** Đã cố định 0,9 và 5 bước cho mọi cấu hình, nhưng các ngưỡng khác có thể cho hệ số hồi quy khác.
6. **Quy luật `L` là xấp xỉ, không phải định luật chính xác.** Đường thẳng được hiệu chuẩn trên 16 cấu hình không can thiệp; ba cấu hình hai buồng là điểm kiểm định độc lập và lệch −4,2 bước. Sai lệch này chấp nhận được cho một quy luật bậc một, nhưng không nên đọc như một dự đoán chính xác cho mọi đồ thị.

## 8. Kết luận

Đồ án tái tạo thành công kết quả cốt lõi của Soma và cộng sự: một đàn cá thể chỉ bắt chước lẫn nhau cho ra đường học trùng với một tác tử học tăng cường duy nhất, sai lệch dưới 0,02.

Khi phá vỡ giả định trộn đều bằng cách đặt đàn lên đồ thị, kết quả cho thấy tốc độ học bị chi phối bởi **khoảng cách thông tin**, không phải thời gian trộn phổ — được xác nhận trực tiếp bằng thí nghiệm tách biến trên đồ thị hai buồng. Cụ thể, trên 16 cấu hình không can thiệp, `t₉₀ = 44,85 + 3,46·L` với `R² = 0,91`. Ba cấu hình hai buồng là điểm kiểm định độc lập: dự đoán 51,7 bước, thực đo 47,5, lệch −4,2 bước, trong khi `τ` tăng sáu lần. Độ chính xác cuối cùng gần như không đổi trong dải khảo sát.

Về mặt phương pháp, đồ án cho thấy một hướng nghiên cứu nhỏ nhưng trọn vẹn: lấy một chứng minh có điều kiện, phá điều kiện đó, đo cái gì thay đổi, và tìm đại lượng thay thế mô tả được hiện tượng. Thí nghiệm tách biến bằng đồ thị hai buồng đã xác nhận điều này: khi `τ` tăng sáu lần mà `L` giữ nguyên, `t₉₀` không đổi và tương quan với `τ` rơi xuống 0,10. Bước tiếp theo là mở rộng sang mạng có phân bố bậc nặng đuôi, nơi weighted voter thiên lệch theo bậc.

## Tài liệu tham khảo

1. Soma, K., Bouteiller, Y., Hamann, H., Beltrame, G. (2026). The hive mind is a single reinforcement learning agent. *Swarm Intelligence*, 20. https://doi.org/10.1007/s11721-026-00264-7
2. Nowak, M. A. (2006). *Evolutionary Dynamics: Exploring the Equations of Life*. Harvard University Press.
3. Ohtsuki, H., Hauert, C., Lieberman, E., Nowak, M. A. (2006). A simple rule for the evolution of cooperation on graphs and social networks. *Nature*, 441, 502–505.
4. Watts, D. J., Strogatz, S. H. (1998). Collective dynamics of small-world networks. *Nature*, 393, 440–442.
5. Penrose, M. (2003). *Random Geometric Graphs*. Oxford University Press.
6. Sutton, R. S., Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
7. Fisher, R. A. (1937). The wave of advance of advantageous genes. *Annals of Eugenics*, 7(4), 355–369.
8. Kolmogorov, A. N., Petrovsky, I. G., Piskunov, N. S. (1937). Étude de l'équation de la diffusion avec croissance de la quantité de matière et son application à un problème biologique. *Moscow University Bulletin of Mathematics*, 1, 1–25.
