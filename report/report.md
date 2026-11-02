# Bầy ong ra quyết định trên đồ thị: tái tạo và mở rộng HiveMindRL

**Đồ án tổng hợp — Trí tuệ nhân tạo**

*Ngày 2 tháng 11 năm 2026*

---

## Tóm tắt

Bài báo *The hive mind is a single reinforcement learning agent* (Soma và cộng sự, 2026) chứng minh rằng một đàn cá thể chỉ bắt chước lẫn nhau vẫn tạo ra hành vi học tập tương đương một tác tử học tăng cường duy nhất, với điều kiện quần thể được trộn đều. Đồ án này gồm hai phần. Phần thứ nhất tái tạo kết quả cốt lõi của bài báo bằng NumPy: đường học của mô hình bầy đàn, của thuật toán Maynard-Cross Learning song song, và của đường MRD rời rạc trùng nhau với sai lệch lớn nhất 0,018. Phần thứ hai đặt đàn lên một đồ thị giao tiếp cố định — đúng phần mà bài báo để ngỏ — và hỏi cấu trúc đồ thị thay đổi tốc độ học như thế nào. Khảo sát 22 lần chạy trên Watts–Strogatz, random geometric graph và một họ hai buồng cho thấy giả thuyết ban đầu (thời gian hội tụ tỉ lệ với thời gian relaxation `τ = 1/(1−λ₂)`) chỉ đúng trong từng họ. Đường đi trung bình `L` mô tả được cả WS lẫn RGG bằng một quy luật tuyến tính `t₉₀ = 44,85 + 3,46·L` (`R² = 0,91`), nhưng chẩn đoán thống kê cho thấy đây là xấp xỉ mô tả chứ không phải định luật: CI 95% của độ dốc là 3,04–6,67, một cấu hình WS có Cook's D = 10, hiệu ứng họ là +13,4 bước (`p = 0,039`), và bậc trung bình giải thích thêm 5,1% phương sai. Quan trọng hơn, ba cấu hình hai buồng cũ không kiểm định được việc thông tin đi qua cầu: cả hai buồng đều đã chứa type tốt, nên `t₉₀ = 47,5` chỉ phản ánh học cục bộ. Khi chỉ gieo type tốt vào một buồng (`N = 300`, 600 bước, 30 seed), `t₅₀` trung vị giảm từ 425 bước (cầu 1; 27% seed đạt) xuống 132 bước (cầu 12; 93% seed đạt) trong khi `L ≈ 1,9–2,0`. Kết luận: cấu trúc tác động qua ít nhất hai kênh — khoảng cách thông tin và dung lượng nút cổ chai — nên không một thang đo đơn lẻ nào là đủ.

## 1. Giới thiệu

Một đàn ong chọn tổ mới không có ai chỉ huy. Mỗi con khảo sát một địa điểm, đánh giá chất lượng, rồi truyền thông tin cho những con khác. Cuối cùng cả đàn thường đồng thuận về địa điểm tốt nhất. Hiện tượng này — trí tuệ bầy đàn — đặt một câu hỏi lý thuyết rõ ràng: hành vi học tập ở cấp tập thể có thể được mô tả bằng một thuật toán học tăng cường ở cấp cá nhân không?

Soma và cộng sự (2026) trả lời là có, trong giới hạn của mô hình. Họ chứng minh hai mệnh đề. Mệnh đề thứ nhất chỉ ra rằng quy tắc *bắt chước cá thể thành công* tương đương với Taylor Replicator Dynamics, và do đó tương đương với Cross Learning. Mệnh đề thứ hai chỉ ra rằng *weighted voter model* — nơi mỗi cá thể quảng bá lựa chọn của mình với cường độ tỉ lệ với phần thưởng nhận được — tương đương với một thuật toán mới do họ đề xuất, Maynard-Cross Learning (MCL). MCL khác Cross Learning ở chỗ chia tín hiệu phần thưởng cho giá trị kỳ vọng của chính sách hiện tại, nhờ đó tín hiệu được chuẩn hoá theo chất lượng chung của hệ thống.

Điểm yếu của chứng minh là giả định **trộn đều**: mỗi cá thể được coi là quan sát được toàn bộ quần thể, hoặc một vùng lân cận ngẫu nhiên đủ lớn. Chính bài báo thừa nhận điều này ở mục 4.9.2: *"The well-mixed assumption of Proposition 2 may not hold in the presence of spatial structure… We leave the investigation of spatial structure and heterogeneity in our framework for future work."*

Đồ án này đi vào đúng khoảng trống đó. Câu hỏi nghiên cứu:

> Khi đàn được đặt lên một đồ thị giao tiếp cố định và mỗi cá thể chỉ quan sát hàng xóm của mình, cấu trúc đồ thị thay đổi tốc độ và độ chính xác của việc học như thế nào?

Đóng góp của đồ án gồm bốn phần:

1. Một bản cài đặt độc lập, có kiểm thử, tái tạo được kết quả cốt lõi của bài báo (Hình 1).
2. Một khảo sát thực nghiệm trên 22 lần chạy thuộc hai họ ngẫu nhiên (WS, RGG) và một họ hai buồng, với 100 seed mỗi cấu hình (Hình 2, 3; Bảng 1).
3. Một quy luật mô tả `t₉₀ ~ L` cho WS và RGG, kèm chẩn đoán độ bất định: bootstrap CI, Cook's D, leave-one-out, hiệu ứng họ và ảnh hưởng của bậc.
4. Một thí nghiệm can thiệp gieo nguồn trên đồ thị hai buồng, cho thấy ở `L` gần như cố định, độ rộng cầu vẫn làm thời gian lan thông tin thay đổi hơn ba lần. Kết quả này bác bỏ cách đọc cũ rằng đồ thị hai buồng là phép kiểm định sạch cho quy luật `L`.

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

Ba cấu hình hai buồng trong Bảng 1 là **can thiệp rỗng** cho câu hỏi thông tin qua cầu: cả hai buồng đều khởi tạo đủ 10 type, nên type tốt gần như chắc chắn có mặt ở cả hai và cầu không mang thông tin cần thiết. Để kiểm định đúng, đồ án bổ sung thí nghiệm **gieo nguồn**: type tốt chỉ xuất hiện trong buồng nguồn; buồng đích khởi tạo đều trên 9 type còn lại. Đo `t₁₀`, `t₅₀`, `t₉₀` và `final_best_mass` *trong buồng đích* cho cầu rộng 1, 3, 6 và 12 cạnh. Cấu hình: `N = 300`, `T = 600` bước, 30 seed, cùng bandit; mỗi độ rộng cầu dùng một đồ thị cố định (seed 0).

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

`p = 10⁻⁴` trùng hoàn toàn với `p = 0` (cùng đồ thị sinh ra) nên chỉ trình bày một lần. `final` là trung vị của `final_best_mass`; cột cuối là tỉ lệ seed có `final_best_mass ≥ 0,95`. Ba dòng hai buồng là can thiệp rỗng: cả hai buồng đã chứa type tốt, nên `t₉₀ = 47,5` đo học cục bộ chứ không đo thời gian thông tin qua cầu.

Hình 2 đặt hai thước đo cạnh nhau.

![Hình 2. (a) Thời gian hội tụ theo thời gian relaxation. (b) Thời gian hội tụ theo đường đi trung bình; đường thẳng hiệu chuẩn trên 16 cấu hình không can thiệp. Ba điểm hai buồng là can thiệp rỗng và không được dùng làm kiểm định cho quy luật `L`.](figures/fig2_law.png)

**Giả thuyết ban đầu không đứng vững ở dạng phổ quát.** Ở đồ thị (a), mỗi họ có xu hướng tăng theo `τ`, nhưng hai họ nằm ở hai đường khác nhau: tại cùng `τ ≈ 100`, RGG cần khoảng 60 bước còn WS cần khoảng 90 bước (nội suy). Hệ số tương quan hạng Spearman giữa `t₉₀` và `τ` là 0,69 khi chỉ xét 16 cấu hình không can thiệp (đầy đủ + WS + RGG), nhưng rơi xuống 0,10 khi thêm họ hai buồng. Cú rơi này cho thấy `τ` không xếp hạng được các họ khác nhau; nó không phải bằng chứng rằng `L` là cơ chế đúng, vì họ hai buồng là can thiệp rỗng (xem §5.2).

**Đường đi trung bình mô tả được cả WS lẫn RGG.** Ở đồ thị (b), các điểm của cả WS lẫn RGG nằm quanh một đường thẳng, hiệu chuẩn trên 16 cấu hình không can thiệp:

```
t₉₀ = 44,85 + 3,46 · L,     R² = 0,91
```

Hệ số tương quan hạng với `L` là 0,82 trên tập hiệu chuẩn. Đáng chú ý là tung độ gốc 44,85 xấp xỉ `t₉₀ = 44,0` đo được trên đồ thị đầy đủ — nghĩa là có thể đọc quy luật này như **thời gian học nội tại của đàn cộng với thời gian thông tin đi hết đồ thị**. Độ bất định của các hệ số được phân tích ở §5.3.

Hình 3 cho thấy hình dạng đường học thay đổi thế nào khi nới dần vòng lattice.

![Hình 3. Đường học trung bình trên một số cấu hình tiêu biểu, kể cả vòng lattice p = 0.](figures/fig3_trajectories.png)

Đồ thị đầy đủ và MRD gần như trùng nhau. Khi `p` giảm, đường học kéo dài về phía phải nhưng vẫn tiến tới 1. Với vòng lattice (`p = 0`), trung vị `final` vẫn bằng 1,0 nhưng chỉ 82% seed đạt 0,95; đường trung bình đạt khoảng 0,97 ở bước 400, chậm hơn khoảng bốn lần so với đồ thị đầy đủ.

### 5.1. Độ chính xác: trung vị che mất phần đuôi

Cột `final` (trung vị của `final_best_mass`) bằng 1,00 ở **mọi** cấu hình, kể cả vòng lattice — một giá trị bão hoà, không phân biệt được cấu hình nhanh với cấu hình chậm. Vì vậy Bảng 1 thêm cột `seed final ≥ 0,95`: tỉ lệ seed mà khối lượng phương án tốt nhất ở cuối đạt ít nhất 0,95.

Đọc cột đó mới thấy bức tranh thật. Ở 17/19 topology liên thông, **100% seed** đạt 0,95. Nhưng ở hai topology chậm nhất, phần đuôi hiện ra: vòng lattice (`p = 0`) chỉ có **0,82** seed đạt 0,95, và `p = 10⁻³` có **0,89**. Đây đúng là hai cấu hình có tỉ lệ seed đạt `t₉₀ = 0,9` trong 400 bước thấp nhất (0,90 và 0,96) — cùng một nhóm seed "chậm" bị đếm qua hai thước đo khác nhau.

Kết luận cần thận trọng hơn: **cấu trúc đồ thị ảnh hưởng mạnh tới tốc độ, và ảnh hưởng rõ rệt tới độ chính xác chỉ ở những cấu hình mà thời gian học tiến sát cửa sổ 400 bước.** Với 17 topology còn lại, độ chính xác cuối kỳ thực sự không đổi; hai đồ thị RGG bị phân mảnh (không liên thông) cũng đạt 0,88–1,00 vì mỗi thành phần liên thông vẫn đủ lớn để tự học đúng. Nói cách khác, trong dải khảo sát, đồ thị **mua được độ trễ, không mua được độ đúng** — trừ khi độ trễ chạm trần mô phỏng, và khi đó cái mất là vài phần trăm seed, không phải toàn bộ quần thể.

### 5.2. Can thiệp hai buồng: thiết kế cũ là can thiệp rỗng

Ba dòng cuối Bảng 1 được thiết kế để tách `τ` khỏi `L`: hai clique 250 nút nối bằng cầu 1, 3 hoặc 6 cạnh làm `τ` tăng sáu lần trong khi `L ≈ 2,0`. Nhưng thiết kế đó **không kiểm định được** điều nó định kiểm định. Cả hai buồng đều khởi tạo cân bằng trên đủ 10 type, nên xác suất buồng đích *không* chứa type tốt là `0,9²⁵⁰ ≈ 3,6·10⁻¹²`. Type tốt gần như chắc chắn đã có sẵn ở cả hai buồng; cầu không mang thông tin cần thiết nào. Vì vậy `t₉₀ = 47,5` ở cả ba cầu chỉ là thời gian học cục bộ bên trong một clique 250 nút — hơi nhanh hơn đồ thị đầy đủ `N = 500` (`t₉₀ = 44,0`) vì quần thể nhỏ hơn. Ba dòng này không phải bằng chứng rằng `t₉₀` bám theo `L` hay bỏ qua `τ`; chúng chỉ cho thấy một can thiệp rỗng, đúng như dự đoán.

Để kiểm định đúng, đồ án chạy thí nghiệm **gieo nguồn**: type tốt chỉ tồn tại trong buồng nguồn, còn buồng đích khởi tạo đều trên 9 type còn lại. Muốn buồng đích học được, thông tin bắt buộc phải đi qua cầu. Bảng 1b và Hình 4 trình bày kết quả.

**Bảng 1b. Gieo nguồn buồng nguồn (N = 300, 600 bước, 30 seed; đo trong buồng đích).**

| Cầu | τ | L | t₅₀ | seed đạt t₅₀ | t₉₀* | seed đạt t₉₀ | seed final ≥ 0,95 |
|---|---|---|---|---|---|---|---|
| 1 | 11.324 | 2,00 | 425,5 | 0,27 | 385,0 | 0,23 | 0,17 |
| 3 | 3.774 | 1,98 | 248,0 | 0,63 | 285,0 | 0,63 | 0,57 |
| 6 | 1.887 | 1,96 | 228,5 | 0,93 | 253,0 | 0,93 | 0,83 |
| 12 | 943 | 1,93 | 132,5 | 0,93 | 186,5 | 0,93 | 0,93 |

`t₅₀` và `t₉₀*` là trung vị **có điều kiện** trên các seed không bị censored trong 600 bước; cột "seed đạt" là tỉ lệ seed thực sự vượt ngưỡng. Với cầu 1, 73% seed không bao giờ đạt 0,5, nên `t₅₀ = 425,5` chỉ mô tả nhóm seed nhanh nhất và là một cận dưới. Trung vị `t₉₀` trên toàn bộ seed (censored tính bằng 601) là 601 cho cầu 1 và 172,5 cho cầu 12.

![Hình 4. Gieo type tốt vào buồng nguồn. (a) Tỉ lệ chọn type tốt trong buồng đích theo thời gian. (b) Trung vị thời gian đạt 50% và 90% trong buồng đích theo độ rộng cầu, kèm CI 95% bootstrap.](figures/fig4_two_room_source.png)

**Kết quả.** Trong dải `L = 1,93–1,997` — biến thiên chưa tới 4% — `t₅₀` giảm từ 425,5 xuống 132,5 bước, hơn **3,2 lần**, và tỉ lệ seed đạt 50% tăng từ 0,27 lên 0,93. Tỉ lệ seed kết thúc với khối lượng type tốt ≥ 0,95 tăng từ 0,17 lên 0,93. Vì `L` gần như không đổi, thay đổi này **không thể quy cho khoảng cách**: nó do dung lượng của cầu, tức số cạnh nối hai buồng quyết định tốc độ dòng thông tin qua lát cắt. Đây là kênh thứ hai mà một thang đo đơn lẻ như `L` bỏ sót.

Đồng thời, trong thí nghiệm này `τ` và `t₅₀` biến thiên cùng chiều (τ giảm 12 lần, t₅₀ giảm 3,2 lần). Không thể từ đó tuyên bố `τ` vô can: ở đây cả hai đều bị chi phối bởi cùng một nguyên nhân là độ rộng cầu. Điều thí nghiệm này bác bỏ là cách đọc cũ — rằng ba cấu hình hai buồng là phép kiểm định sạch cho quy luật `L`, và rằng bằng chứng ấy cho phép loại `τ` khỏi cơ chế.

Tóm lại, trên WS và RGG — nơi bậc thay đổi vừa phải và không có nút cổ chai rõ rệt — `L` là **thang mô tả hữu ích** cho thời gian học. Nó không phải **thống kê đủ**: khi đồ thị có lát cắt hẹp, dung lượng lát cắt ảnh hưởng tới tốc độ ngang với khoảng cách. Cơ chế được bàn ở §6.

### 5.3. Độ bất định của quy luật `L`

`t₉₀ = 44,85 + 3,46·L` với `R² = 0,91` là một đường hiệu chuẩn đẹp, nhưng cần đọc kèm bốn chẩn đoán (mã trong `src/hive/analysis.py`, kết quả `results/law_diagnostics.json`).

**Khoảng tin cậy.** Bootstrap 95% cho hệ số góc là **3,04–6,67**, tức thời gian học có thể tăng thêm 3 đến 6,7 bước mỗi bước đường đi. Khoảng này rộng vì chỉ có 16 điểm và `L` trải từ 1 đến 42 theo thang gần log. CI cho tung độ gốc là 28,1–50,8, và dự đoán tại `L = 2` là **41,2–57,4** — rộng hơn sai lệch −4,2 bước của thiết kế hai buồng cũ. Nói cách khác, "dự đoán 51,7, thực đo 47,5" không phải một kiểm định có sức mạnh: khoảng dự đoán đã phủ sẵn giá trị thực đo.

**Ảnh hưởng của từng điểm.** Một cấu hình — vòng lattice `WS p = 0`, `L = 42,1`, `t₉₀ = 173,5` — có **Cook's D = 10,0**, lớn hơn ngưỡng 1 một bậc và có leverage 0,73 (trung bình kỳ vọng `2p/n = 0,25`). Điểm này một mình kéo đường hồi quy. Leave-one-out cho thấy hệ số góc dao động 3,17–4,70 và `R²` 0,906–0,941, nên kết luận "có quan hệ tuyến tính" không phụ thuộc vào một điểm, nhưng **giá trị cụ thể của độ dốc thì có**. Bỏ `WS p = 0` làm độ dốc tăng khoảng 36%.

**Hiệu ứng họ.** Thêm một biến giả cho WS làm `R²` tăng lên 0,939 và cho thấy WS lệch **+13,4 bước** so với baseline (`SE = 5,8`, `p = 0,039`). Tức ở cùng `L`, WS chậm hơn RGG khoảng 13 bước. Đây là vi phạm trực tiếp giả định "một đường thẳng cho mọi họ"; hệ số tương tác âm (−2,7) không có ý nghĩa (`p = 0,43`), nên dạng đúng hơn là hai đường song song hơn là hai độ dốc khác nhau.

**Bậc và hình dạng bậc.** Bậc trung bình tương quan hạng âm với `t₉₀` (Spearman −0,70, `p = 0,0025`) và degree CV tương quan Pearson −0,64 (`p = 0,007`). Thêm `log(bậc)` vào mô hình `L` tăng `R²` thêm 1,4 điểm phần trăm; thêm degree CV tăng 2,3; thêm cả hai tăng **5,1** (từ 0,910 lên 0,962). Vì vậy `L` chưa được tách khỏi bậc: các đồ thị có `L` lớn cũng là các đồ thị thưa với bậc thấp, và mô hình hiện tại không phân biệt được hai cách giải thích. Đây là lý do §5.4 kiểm tra thêm mạng đuôi nặng.

Các chẩn đoán này không phủ nhận quy luật; chúng xác định phạm vi đọc nó. `L` giải thích phần lớn biến thiên trong mẫu, nhưng độ dốc bất định, có hiệu ứng họ, và chưa tách khỏi bậc. Nên phát biểu là **"trong các họ WS và RGG đã khảo sát, `t₉₀` tăng xấp xỉ tuyến tính theo `L`"**, không phải "thời gian học bằng một hằng số nhân với `L` trên mọi đồ thị".

### 5.4. Kiểm tra ngoài phạm vi: mạng đuôi nặng

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

Spectral gap đo mode chậm nhất của một quá trình **khuếch tán** — một hạt ngẫu nhiên đi lang thang cho tới khi mất tương quan. Nhưng bắt chước có trọng số không phải quá trình khuếch tán thuần: phương án có phần thưởng cao được sao chép với xác suất cao hơn, tạo ra một **dòng trôi có hướng**. Dưới tác dụng của dòng trôi, kiểu tốt nhất lan ra như một mặt trận chuyển động với vận tốc xấp xỉ không đổi, nên trong các đồ thị không có nút cổ chai rõ rệt, thời gian để nó phủ 90% quần thể bị chi phối bởi **khoảng cách phải đi**, tức `L`, chứ không phải bởi thời gian trộn của bước đi ngẫu nhiên.

Cách giải thích này khớp với ba quan sát. Thứ nhất, tung độ gốc của quy luật xấp xỉ thời gian học của quần thể trộn đều — phần "không phụ thuộc đồ thị" đúng là động lực học chọn lọc thuần tuý. Thứ hai, hai họ đồ thị có cùng `L` nhưng khác số chiều (vòng 1D so với mặt phẳng 2D) lại nằm sát nhau hơn nhiều so với khi so theo `τ`. Thứ ba, đường học trên vòng lattice có dạng kéo dài chứ không phải trễ rồi bùng nổ — dấu hiệu của lan truyền theo khoảng cách.

Thí nghiệm gieo nguồn ở §5.2 vẽ nốt bức tranh. Khi type tốt chỉ có ở buồng nguồn, thông tin buộc phải qua cầu. Ở `L` gần như cố định (1,93–2,00), `t₅₀` vẫn giảm hơn ba lần khi cầu rộng từ 1 lên 12 cạnh. Vậy có ít nhất hai kênh tác động. Kênh thứ nhất là **khoảng cách mặt trận phải đi**: dưới dòng trôi thiên lệch, mật độ cá thể chọn phương án tốt nhất thoả một phương trình kiểu Fisher–KPP [7, 8], có nghiệm sóng chạy với vận tốc không đổi, nên thời gian phủ 90% quần thể tỉ lệ với quãng đường, tức `L`. Kênh thứ hai là **dung lượng lát cắt**: mỗi bước chỉ một phần nhỏ cá thể ở biên buồng nguồn được thay bằng type tốt, và dòng đó tỉ lệ với số cạnh cầu. Khi cầu hẹp, mặt trận bị nghẽn tại cửa; thời gian chờ tại cửa cộng vào thời gian đi trong buồng đích. Cơ chế này giống một điện trở nối tiếp: hai buồng có "trở kháng" thấp, cầu có trở kháng cao, và tổng thời gian do đoạn trở kháng cao nhất quyết định.

Ba thí nghiệm khớp với bức tranh hai kênh. Đồ thị đầy đủ và clique 250 nút có `L` ngắn và không có cổ chai, nên học nhanh (44 và 47,5 bước). WS và RGG có `L` dài nhưng bậc tương đối đều, nên `L` mô tả được. Hai buồng gieo nguồn có `L` ngắn nhưng cổ chai hẹp, nên `L` dự đoán sai và độ rộng cầu mới là biến quyết định. Cách đọc cũ — "hai buồng chứng minh `τ` vô can" — là hệ quả của một can thiệp rỗng, không phải của cơ chế. Phát biểu đúng là: **`L` đo kênh khoảng cách, nhưng không đo kênh dung lượng; một mô hình đầy đủ cần cả hai, hoặc một đại lượng gộp như thời gian qua lát cắt hẹp nhất.** Đây vẫn là giả thuyết cơ chế, chưa phải kết quả đo trực tiếp vận tốc mặt trận hay thông lượng qua cầu; §7 nêu rõ giới hạn này.

## 7. Hạn chế

1. **Censoring.** Hai cấu hình chậm nhất của Bảng 1 có 4–10% seed không đạt 0,9 trong 400 bước. Ở thí nghiệm gieo nguồn, censoring nặng hơn nhiều: với cầu 1, 73% seed không đạt 0,5 trong 600 bước, nên các trung vị `t₅₀`, `t₉₀` chỉ mô tả nhóm seed nhanh và phải đọc cùng tỉ lệ seed đạt. Đã báo cáo riêng tỉ lệ đạt thay vì che đi; `t₁₀` không dùng được vì nhiều cấu hình chạm trần 600 bước ngay từ ngưỡng 10%.
2. **Cùng một kịch bản chất lượng.** Thí nghiệm đồ thị chỉ dùng `q ∈ [0,4; 0,7]`. Với dải chất lượng thấp hơn, nhiễu tương đối lớn hơn và đàn dễ kẹt ở phương án tầm trung.
3. **Mạng bậc nặng đuôi mới chỉ khảo sát sơ bộ.** Mục 5.4 kiểm tra thêm ba mạng ngoài phạm vi quét chính (BA `m = 2`, BA `m = 5`, ER bậc 6) với 100 seed: quy luật `L` vẫn giữ, nhưng sai lệch so với đường hiệu chuẩn là −3,4…+1,0 bước và BA `m = 2` lệch **dương**, tức ba điểm chưa đủ để khẳng định quy luật phổ quát cho mọi mạng đuôi nặng. Cần quét có hệ thống theo hệ số biến thiên bậc để kết luận.
4. **Không có cơ chế khám phá.** Như bài báo gốc, khi một phương án biến mất khỏi quần thể thì nó không thể quay lại. Trong môi trường thay đổi theo thời gian, đàn sẽ không thích nghi được.
5. **`t₉₀` phụ thuộc ngưỡng và cửa sổ làm trơn.** Đã cố định 0,9 và 5 bước cho mọi cấu hình, nhưng các ngưỡng khác có thể cho hệ số hồi quy khác.
6. **Quy luật `L` là xấp xỉ mô tả, không phải định luật.** CI 95% của độ dốc là 3,04–6,67; một điểm (`WS p = 0`) có Cook's D = 10 và leverage 0,73; có hiệu ứng họ +13,4 bước (`p = 0,039`); thêm bậc và degree CV giải thích thêm 5,1% phương sai. Nên đọc quy luật trong phạm vi WS/RGG đã khảo sát, không ngoại suy cho mọi đồ thị.
7. **Thí nghiệm gieo nguồn mới là pilot.** Mỗi độ rộng cầu dùng một đồ thị cố định, `N = 300`, 30 seed, và chỉ quét cầu 1–12. Chưa đo trực tiếp thông lượng qua cầu hay vận tốc mặt trận; `τ` và dung lượng cầu cùng biến thiên nên thí nghiệm này chứng minh `L` không đủ, chứ chưa tách được hai kênh. Cần lặp trên nhiều đồ thị và nhiều `N` để có CI cho hiệu ứng dung lượng.

## 8. Kết luận

Đồ án tái tạo thành công kết quả cốt lõi của Soma và cộng sự: một đàn cá thể chỉ bắt chước lẫn nhau cho ra đường học trùng với một tác tử học tăng cường duy nhất, sai lệch dưới 0,02.

Khi phá vỡ giả định trộn đều bằng cách đặt đàn lên đồ thị, kết quả cho thấy tốc độ học không thể gói gọn trong một thang đo duy nhất. Trên 16 cấu hình không can thiệp (đầy đủ, WS, RGG), `t₉₀ = 44,85 + 3,46·L` với `R² = 0,91`, nhưng độ dốc có CI 95% 3,04–6,67, tồn tại hiệu ứng họ +13,4 bước, và bậc giải thích thêm 5,1% phương sai. Quy luật này là **mô tả hữu ích**, không phải định luật phổ quát.

Quan trọng hơn, thiết kế hai buồng cũ không kiểm định được việc thông tin qua cầu: cả hai buồng đều đã chứa type tốt, nên `t₉₀ = 47,5` chỉ là học cục bộ. Khi gieo type tốt vào một buồng, ở `L ≈ 1,9–2,0`, `t₅₀` giảm từ 425 xuống 132 bước và tỉ lệ seed đạt 50% tăng từ 0,27 lên 0,93 chỉ nhờ tăng độ rộng cầu. Kết luận đúng là cấu trúc tác động qua **ít nhất hai kênh**: khoảng cách thông tin (được `L` đo) và dung lượng lát cắt (bị `L` bỏ qua). Một lý thuyết đầy đủ về học tập trên đồ thị cần cả hai, hoặc một đại lượng gộp như thời gian qua lát cắt hẹp nhất.

Về mặt phương pháp, đồ án cho thấy một hướng nghiên cứu nhỏ nhưng trọn vẹn: lấy một chứng minh có điều kiện, phá điều kiện đó, đo cái gì thay đổi, và kiểm tra xem thiết kế thí nghiệm có thực sự can thiệp vào cơ chế hay không. Bài học lớn nhất không phải là một công thức `t₉₀ ~ L`, mà là việc phát hiện thiết kế hai buồng ban đầu là can thiệp rỗng — và sửa nó bằng cách gieo nguồn. Bước tiếp theo là lặp thí nghiệm gieo nguồn trên nhiều đồ thị và nhiều `N`, đo trực tiếp thông lượng qua cầu, và mở rộng sang mạng đuôi nặng nơi weighted voter thiên lệch theo bậc.

## Tài liệu tham khảo

1. Soma, K., Bouteiller, Y., Hamann, H., Beltrame, G. (2026). The hive mind is a single reinforcement learning agent. *Swarm Intelligence*, 20. https://doi.org/10.1007/s11721-026-00264-7
2. Nowak, M. A. (2006). *Evolutionary Dynamics: Exploring the Equations of Life*. Harvard University Press.
3. Ohtsuki, H., Hauert, C., Lieberman, E., Nowak, M. A. (2006). A simple rule for the evolution of cooperation on graphs and social networks. *Nature*, 441, 502–505.
4. Watts, D. J., Strogatz, S. H. (1998). Collective dynamics of small-world networks. *Nature*, 393, 440–442.
5. Penrose, M. (2003). *Random Geometric Graphs*. Oxford University Press.
6. Sutton, R. S., Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
7. Fisher, R. A. (1937). The wave of advance of advantageous genes. *Annals of Eugenics*, 7(4), 355–369.
8. Kolmogorov, A. N., Petrovsky, I. G., Piskunov, N. S. (1937). Étude de l'équation de la diffusion avec croissance de la quantité de matière et son application à un problème biologique. *Moscow University Bulletin of Mathematics*, 1, 1–25.
