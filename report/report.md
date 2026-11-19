# Bầy ong ra quyết định trên đồ thị: tái tạo và mở rộng HiveMindRL

**Đồ án tổng hợp — Trí tuệ nhân tạo**

*Cập nhật ngày 19 tháng 11 năm 2026*

---

## Tóm tắt

Đồ án tái tạo kết quả của Soma và cộng sự (2026) trong một kịch bản bandit trộn đều. Đường học trung bình của Maynard–Cross Learning song song sai lệch tối đa 0,018 so với tham chiếu MRD từ bước 10. Sau đó, đồ án giới hạn giao tiếp trên các đồ thị cố định. Trong 16 cấu hình hiệu chuẩn, đường đi trung bình `L` có quan hệ gần tuyến tính với trung vị Kaplan–Meier của `t₉₀`. Kết quả giữ cùng chiều khi lấy thêm mẫu đồ thị và khi đổi cách đo ngưỡng, nhưng độ dốc vẫn nhạy với tập cấu hình.

Để đo việc truyền phương án tốt qua một nút cổ chai, thí nghiệm gieo phương án tốt chỉ vào buồng nguồn. Khi tăng độ rộng cầu từ 1 lên 12 cạnh, RMST₅₀ đến trần 600 bước giảm từ 550,6 xuống 185,8, dù `L` chỉ đổi từ 2,00 xuống 1,93. Bản nhân rộng giữ một topology clique cho mỗi cặp kích thước và độ rộng cầu, với 80 quỹ đạo độc lập mỗi cấu hình. Thí nghiệm buồng thưa và buồng Barabási–Albert tiếp tục đối chiếu bậc đầu cầu, thời gian relaxation `τ` và thời gian đạt ngưỡng ở buồng đích. Trong các cấu hình khảo sát, đại lượng dự đoán tốt lần tới đầu tiên có thể khác đại lượng dự đoán tốt `t₅₀`. Cơ chế mặt trận lan truyền và xác suất trụ lại của từng lần xâm nhập vẫn cần đo trực tiếp.

## 1. Giới thiệu

Một đàn ong chọn tổ mới không có ai chỉ huy. Mỗi con khảo sát một địa điểm, đánh giá chất lượng, rồi truyền thông tin cho những con khác. Cuối cùng cả đàn thường đồng thuận về địa điểm tốt nhất. Hiện tượng này — trí tuệ bầy đàn — đặt một câu hỏi lý thuyết rõ ràng: hành vi học tập ở cấp tập thể có thể được mô tả bằng một thuật toán học tăng cường ở cấp cá nhân không?

Soma và cộng sự (2026) trả lời là có, trong giới hạn của mô hình. Họ chứng minh hai mệnh đề. Mệnh đề thứ nhất chỉ ra rằng quy tắc *bắt chước cá thể thành công* tương đương với Taylor Replicator Dynamics, và do đó tương đương với Cross Learning. Mệnh đề thứ hai chỉ ra rằng *weighted voter model* — nơi mỗi cá thể quảng bá lựa chọn của mình với cường độ tỉ lệ với phần thưởng nhận được — tương đương với một thuật toán mới do họ đề xuất, Maynard-Cross Learning (MCL). MCL khác Cross Learning ở chỗ chia tín hiệu phần thưởng cho giá trị kỳ vọng của chính sách hiện tại, nhờ đó tín hiệu được chuẩn hoá theo chất lượng chung của hệ thống.

Điểm yếu của chứng minh là giả định **trộn đều**: mỗi cá thể được coi là quan sát được toàn bộ quần thể, hoặc một vùng lân cận ngẫu nhiên đủ lớn. Chính bài báo thừa nhận điều này ở mục 4.9.2: *"The well-mixed assumption of Proposition 2 may not hold in the presence of spatial structure… We leave the investigation of spatial structure and heterogeneity in our framework for future work."*

Đồ án này đi vào đúng khoảng trống đó. Câu hỏi nghiên cứu:

> Khi đàn được đặt lên một đồ thị giao tiếp cố định và mỗi cá thể chỉ quan sát hàng xóm của mình, cấu trúc đồ thị thay đổi tốc độ và độ chính xác của việc học như thế nào?

Đóng góp của đồ án gồm bốn phần:

1. Một bản tái cài đặt NumPy/SciPy, có kiểm thử, tái tạo quan hệ cốt lõi trong cấu hình đã chọn (Hình 1), với các khác biệt upstream được nêu ở §3.4.
2. Một khảo sát thực nghiệm trên 22 cấu hình, gồm baseline đầy đủ, hai họ ngẫu nhiên WS và RGG cùng ba đồ thị hai buồng. Mỗi cấu hình dùng 100 seed (Hình 2, 3; Bảng 1).
3. Một quy luật mô tả `t₉₀ ~ L` cho WS và RGG, kèm chẩn đoán độ bất định: bootstrap CI (một graph và phân tầng nhiều graph), Cook's D, leave-one-out, leave-one-family, hiệu ứng họ và ảnh hưởng của bậc.
4. Thí nghiệm gieo nguồn cho thấy độ rộng cầu ảnh hưởng đến thời gian lan dù `L` gần như cố định. Ba bản mở rộng khảo sát kích thước quần thể, bậc của buồng và vị trí đầu cầu (§5.5–5.7).

## 2. Cơ sở lý thuyết

### 2.1. Multi-armed bandit

Môi trường gồm `n` hành động. Mỗi hành động `a` có chất lượng ẩn `qₐ`. Khi hành động được chọn, môi trường trả về phần thưởng nhiễu

```
rₐ ~ U(qₐ − Δ, qₐ + Δ)
```

Trong đồ án, `n = 10`, `Δ = 0,1`, và `qₐ` cách đều trên đoạn `[0,4; 0,7]` (kịch bản *middle* của bài báo). Cài đặt này kẹp phần thưởng vào `[0, 1]`, trong khi `BanditLinear` của repository gốc không kẹp. Ở cấu hình middle trên, phần thưởng đã nằm trong `[0,3; 0,8]` nên khác biệt không tác động tới số liệu; với cấu hình gây clipping, tham chiếu MRD dùng kỳ vọng phần thưởng sau kẹp.

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

là thang thời gian của quá trình khuếch tán trên đồ thị. `τ ≈ 1` tương ứng quần thể trộn đều; `τ` lớn tương ứng thông tin lan chậm. Lưu ý `λ₂` một mình không đủ: với vòng lattice 500 đỉnh bậc 6 (WS `p = 0`), `λ₂ ≈ 0,9996` trông gần hoàn hảo nhưng `τ ≈ 2.714` bước.

`λ₂` ở đây là eigenvalue lớn thứ hai **theo giá trị đại số**, nên `τ` là một đại lượng spectral gap: bước đi ngẫu nhiên *lazy* có thời gian relaxation `2τ`, còn thời gian trộn tới một sai số cho trước phụ thuộc thêm phân bố dừng và ngưỡng sai số. Nó không bảo đảm trộn cho bước đi không lazy trên đồ thị tuần hoàn (hai phía): đồ thị hai nút có `τ = 0,5` nhưng sao chép đồng bộ chỉ đổi chỗ type mãi mãi. Mọi topology khảo sát trong đồ án đều không hai phía, nên lưu ý này không thay đổi số liệu nào.

**Đường đi trung bình.** `L` là khoảng cách ngắn nhất trung bình trên mọi cặp đỉnh liên thông. Đây là thang *khoảng cách* của đồ thị, không phải thang *thời gian trộn*.

## 3. Phương pháp

### 3.1. Cài đặt

Mô phỏng dùng Python với NumPy và SciPy, vector hoá theo từng bước quyết định, không dùng GPU. Các mô-đun tách môi trường bandit, luật cập nhật, mô hình bầy đàn, đồ thị, đo lường, thí nghiệm và chẩn đoán thống kê; bộ kiểm thử đơn vị và tích hợp kiểm tra các bất biến cùng pipeline tái lập.

Một điểm kỹ thuật đáng nêu: bước cập nhật trên đồ thị được tính bằng một phép nhân thưa

```
W = A · (onehot(T) ⊙ r)
```

cho ra ma trận `N × n` trọng số phiếu của từng cá thể, sau đó lấy mẫu theo hàng. Chi phí mỗi bước xấp xỉ `O(|E|·n + N·n)`; lợi thế rõ nhất trên đồ thị thưa, còn clique vẫn có số cạnh bậc `N²`. Thời gian chạy phụ thuộc cấu hình, thư viện và máy; các script báo thời gian thực đo khi hoàn thành.

### 3.2. Tham số

**Thí nghiệm A (tái tạo).** `N = 1000` cá thể, `n = 10` phương án, `T = 300` bước, 100 seed, `B = N = 1000` mẫu song song cho P-MCL; mọi phương pháp bắt đầu từ phân bố đều. Table 1 của bài báo ghi 1000 iteration và 1000 seed, nhưng notebook tại revision upstream đối chiếu ở §3.4 dùng 150 bước và 1000 lần chạy. Cấu hình địa phương là lựa chọn riêng, không phải bản chạy lại nguyên bộ thí nghiệm upstream; 300 bước bao phủ pha hội tụ quan sát được của thí nghiệm A, chưa kiểm định mọi kịch bản hay phần đuôi.

**Thí nghiệm B (đồ thị).** `N = 500`, `T = 400` bước, 100 seed, cùng bandit. Hai họ ngẫu nhiên và một họ can thiệp:

- **Watts–Strogatz** với bậc `k = 6`, quét `p ∈ {0; 10⁻⁴; 10⁻³; 0,01; 0,05; 0,1; 0,2; 0,4; 0,7; 1}`. `p = 0` là vòng lattice bậc 6 (mỗi nút nối với 6 láng giềng gần nhất), `p = 1` là đồ thị ngẫu nhiên.
- **Random geometric graph** trên hình vuông đơn vị, bán kính `radius = c·√(log N / (πN))`, quét `c ∈ {1,0; 1,2; …; 2,5}`. Hai giá trị đầu nằm dưới ngưỡng liên thông.
- **Hai buồng.** Hai clique 250 nút nối với nhau bằng một cầu hẹp 1, 3 hoặc 6 cạnh. Đây là họ can thiệp có chủ đích, không phải một họ ngẫu nhiên.

Ba cấu hình hai buồng trong Bảng 1 là **can thiệp rỗng** cho câu hỏi thông tin qua cầu: cả hai buồng đều khởi tạo đủ 10 type, nên type tốt gần như chắc chắn có mặt ở cả hai và cầu không mang thông tin cần thiết. Để kiểm định đúng, đồ án bổ sung thí nghiệm **gieo nguồn**: type tốt chỉ xuất hiện trong buồng nguồn; buồng đích khởi tạo đều trên 9 type còn lại. Đo `t₁₀`, `t₅₀`, `t₉₀` và `final_best_mass` *trong buồng đích* cho cầu rộng 1, 3, 6 và 12 cạnh. Cấu hình: `N = 300`, `T = 600` bước, 30 seed, cùng bandit; mỗi độ rộng cầu dùng một đồ thị cố định (seed 0).

Mỗi cấu hình dùng **một đồ thị cố định** (seed 0) và 100 seed cho phần ngẫu nhiên của quá trình học. Cách này giữ `τ` và `L` cố định trong từng cấu hình, nhờ đó biến thiên quan sát được quy cho động lực học chứ không phải cho việc lấy mẫu đồ thị. Tham chiếu well-mixed dùng đồ thị đầy đủ (`τ ≈ 1`).

### 3.3. Đo lường

- `t₉₀`: bước đầu tiên mà **trung bình trượt đối xứng 5 bước** (centered moving average) của tỉ lệ cá thể chọn phương án tốt nhất đạt 0,9. Trung bình trượt dùng đệm biên bằng giá trị đầu/cuối, để các seed đạt ngưỡng sát cuối cửa sổ không bị tính nhầm là censored. Nếu không đạt trong `T` bước, giá trị được đánh dấu *censored* và báo cáo riêng thay vì gán bằng `T`. Cần nhấn mạnh: vì cửa sổ đối xứng dùng cả dữ liệu tương lai, `t₉₀` là thời điểm vượt ngưỡng của *đường đã làm trơn*, không phải thời điểm vật lý đầu tiên tín hiệu xuất hiện — nó có thể sớm hơn vài bước. `threshold_time`/`t90` nhận thêm `method="causal"` (trung bình trượt đuôi, chỉ dùng quá khứ + hiện tại) và `method="raw"` (không làm trơn); §5.8 so sánh ba định nghĩa trên cùng quỹ đạo để kiểm tra kết luận nào nhạy với cách đo. Mặc định toàn báo cáo giữ `"symmetric"` để nhất quán với các con số headline.
- **Trung vị Kaplan–Meier:** thời điểm đầu tiên xác suất chưa vượt ngưỡng ước lượng xuống 50% hoặc thấp hơn. Nếu không quan sát được mốc đó, trung vị được để trống. Bảng 1, Hình 2, Hình 8, Hình 9 và tab So sánh dùng cùng định nghĩa. Hình 4 ghi rõ các trung vị tính có điều kiện trong nhóm đạt và báo cáo RMST riêng.
- `final_best_mass`: trung bình tỉ lệ chọn phương án tốt nhất trong 10% bước cuối.
- `consensus`: khối lượng của phương án phổ biến nhất ở bước cuối.
- `l2_mrd`: khoảng cách RMS giữa đường học thực nghiệm và đường MRD.
- `RMST` (restricted mean survival time): diện tích dưới đường sống sót Kaplan–Meier của thời gian đạt ngưỡng, tính tới trần `T` bước. Mỗi seed bị censored đóng góp trọn thời gian quan sát được thay vì bị loại, nên RMST so sánh được giữa các cấu hình có tỉ lệ đạt ngưỡng khác nhau — khác với trung vị có điều kiện chỉ mô tả nhóm seed nhanh. Cài đặt trong `src/hive/metrics.py` (`kaplan_meier`, `rmst`).

### 3.4. Đối chiếu với repository gốc và khả năng tái lập

Repository được bài báo dẫn là MISTLab/HiveMindRL (https://github.com/MISTLab/HiveMindRL), do Karthik Soma, Yann Bouteiller, Heiko Hamann và Giovanni Beltrame công bố. Đối chiếu dưới đây dùng commit `f24851db0a76e6bb2f917f7f7759ccd79476e7a3` ngày 26/10/2026; đây là revision kiểm tra được, không phải một publication snapshot đã được xác nhận bằng tag/release.

| Hạng mục | Upstream tại revision đối chiếu | Cài đặt trong đồ án |
|---|---|---|
| Công cụ và phạm vi | Notebook; NumPy cho bản song song/replicator, PyTorch cho các thuật toán population/streaming RL; nhiều luật bắt chước và môi trường | NumPy/SciPy; TRD, MRD, P-CL, P-MCL và blind weighted voter; không cài streaming MCL hay toàn bộ môi trường upstream |
| Bandit tuyến tính | `BanditLinear` lấy nhiễu đều, không kẹp phần thưởng | Kẹp vào `[0,1]`; không tác động ở cấu hình middle của báo cáo (§2.1) |
| P-MCL và MRD | Chuẩn hoá theo phần thưởng trung bình batch; replicator được tích phân Euler | Công thức tần suất có trọng số tương đương về đại số khi tổng phần thưởng dương; MRD Euler `dt = 1`, khác `delta = 0,1` của nhiễu |
| Lân cận weighted voter | `neighbourhood=True` lấy lại hàng xóm không trùng, loại self mỗi bước; `False` dùng cả self | Well-mixed loại self; phần mở rộng dùng hàng xóm của một đồ thị cố định |
| Thí nghiệm song song được đối chiếu | Notebook dùng `N/B = 1000`, 10 hành động, 150 bước, 1000 lần chạy; Table 1 của bài báo ghi 1000 iteration | Thí nghiệm A dùng `N = B = 1000`, 10 hành động, 300 bước, 100 seed |
| RNG và chỉ số thời gian | Seed NumPy/Torch theo đồng hồ/global RNG; đường ghi trước cập nhật, không chứa trạng thái sau bước cuối | `default_rng(seed)` được chỉ định; đường có dạng `(seed, steps+1)`, gồm trạng thái đầu và sau cập nhật cuối; voter và P-MCL không dùng cùng mẫu hành động/phần thưởng |
| Cấu trúc không gian | Các lân cận ngẫu nhiên được lấy lại, chưa phải lưới đồ thị cố định của đồ án | WS, RGG, hai buồng và BA/ER; `τ`, `L`, gieo nguồn, censoring và chẩn đoán là phần mở rộng địa phương |

Đây là đối chiếu source và quy ước, không phải kiểm định chạy hai repository trên cùng dữ liệu ngẫu nhiên. Sai lệch 0,018 ở §4 kiểm tra các mô hình *bên trong cài đặt địa phương*; nó không chứng nhận tương đương toàn bộ code upstream. Từ “thiết kế hai buồng cũ” trong báo cáo chỉ một phiên bản trước của đồ án, không phải thí nghiệm của tác giả bài báo.

Pipeline hiện tách đầu ra nhanh khỏi đầy đủ và ghi sidecar metadata cho các lần chạy mới: cấu hình thực dùng, lịch seed, phiên bản Python/thư viện, git SHA, trạng thái dirty và SHA256 của source/đầu vào/đầu ra. Vẽ lại hoặc khớp lại chỉ ghi provenance của thao tác dẫn xuất, không gán cấu hình mô phỏng mới cho dữ liệu cũ. Gói bảo vệ chứa source, báo cáo, dữ liệu và manifest checksum; checksum xác minh byte không đổi, không chứng minh tính đúng khoa học, tính xác thực tác giả hay tái lập số học thành công. Các đầu ra lịch sử thiếu metadata vẫn được liệt kê là chưa quy thuộc, không hồi tố bằng môi trường hiện tại; hướng dẫn ở `README.md`.

Bài báo có giấy phép CC BY 4.0 với các ngoại lệ được nêu trong bài; repository code tại revision đối chiếu không có `LICENSE` và GitHub không khai báo giấy phép code. Không suy giấy phép code từ giấy phép bài báo hay từ nhãn “open-source”. Gói địa phương không chứa PDF bài báo hoặc một bản sao source upstream; giấy phép cho đóng góp địa phương cần được chủ sở hữu xác nhận riêng.

## 4. Kết quả tái tạo

Hình 1 so sánh ba đường học trong 300 bước, trung bình trên 100 seed, dải tô là ±1 độ lệch chuẩn.

![Hình 1. Đường học của bầy đàn, P-MCL và đường MRD rời rạc.](figures/fig1_reproduction.png)

Ba đường gần như trùng nhau. Ở bước 50, tỉ lệ chọn phương án tốt nhất lần lượt là 0,916 (bầy đàn), 0,909 (P-MCL) và 0,914 (MRD). Sai lệch tuyệt đối lớn nhất giữa đường trung bình của P-MCL và MRD, tính từ bước 10 trở đi, là **0,018**.

Kết quả phù hợp với quan hệ cốt lõi được bài báo mô tả giữa weighted voter, P-MCL và MRD trong cấu hình well-mixed này; các đường trung bình gần nhau không phải chứng minh tương đương cho mọi tham số hay một kiểm định upstream-versus-local đã thực thi. MRD là tích phân Euler với bước thời gian `dt = 1`, không phải nghiệm giải tích liên tục. `dt` này khác với `delta = 0,1` của code, vốn là nửa bề rộng nhiễu phần thưởng.

## 5. Kết quả trên đồ thị

Bảng 1 tóm tắt 19 topology liên thông phân biệt trong 22 cấu hình khảo sát, mỗi cấu hình có 100 seed. Có 20 cấu hình liên thông, trong đó `p = 10⁻⁴` trùng hoàn toàn với `p = 0` nên chỉ còn 19 topology phân biệt; hai cấu hình RGG dưới ngưỡng liên thông có `τ = ∞` nên không đưa vào bảng, nhưng vẫn được mô phỏng và báo cáo ở phần thảo luận.

**Bảng 1. Kết quả khảo sát đồ thị (N = 500, 400 bước, 100 seed).**

| Họ | Tham số | τ | L | t₉₀ | seed đạt t₉₀ | final | seed final ≥ 0,95 |
|---|---|---|---|---|---|---|---|
| đầy đủ | — | 1,0 | 1,0 | 44,0 | 1,00 | 1,00 | 1,00 |
| RGG | c = 2,5 | 34,0 | 4,16 | 48,0 | 1,00 | 1,00 | 1,00 |
| RGG | c = 2,2 | 48,7 | 4,76 | 51,0 | 1,00 | 1,00 | 1,00 |
| RGG | c = 2,0 | 62,6 | 5,26 | 56,0 | 1,00 | 1,00 | 1,00 |
| RGG | c = 1,8 | 83,8 | 6,02 | 58,0 | 1,00 | 1,00 | 1,00 |
| RGG | c = 1,6 | 112,8 | 6,85 | 60,0 | 1,00 | 1,00 | 1,00 |
| RGG | c = 1,4 | 173,9 | 8,08 | 73,0 | 1,00 | 1,00 | 1,00 |
| WS | p = 1,0 | 3,65 | 3,69 | 58,0 | 1,00 | 1,00 | 1,00 |
| WS | p = 0,7 | 3,88 | 3,71 | 58,0 | 1,00 | 1,00 | 1,00 |
| WS | p = 0,4 | 5,50 | 3,92 | 55,0 | 1,00 | 1,00 | 1,00 |
| WS | p = 0,2 | 13,2 | 4,51 | 61,0 | 1,00 | 1,00 | 1,00 |
| WS | p = 0,1 | 31,1 | 5,40 | 70,0 | 1,00 | 1,00 | 1,00 |
| WS | p = 0,05 | 54,4 | 6,43 | 77,0 | 1,00 | 1,00 | 1,00 |
| WS | p = 0,01 | 281 | 11,26 | 110,0 | 1,00 | 1,00 | 1,00 |
| WS | p = 10⁻³ | 1.878 | 26,09 | 155,0 | 0,96 | 1,00 | 0,89 |
| WS | p = 0 | 2.714 | 42,08 | 173,0 | 0,90 | 1,00 | 0,82 |
| hai buồng | cầu = 6 | 5.229 | 1,98 | 47,0 | 1,00 | 1,00 | 1,00 |
| hai buồng | cầu = 3 | 10.458 | 1,99 | 47,0 | 1,00 | 1,00 | 1,00 |
| hai buồng | cầu = 1 | 31.374 | 2,00 | 47,0 | 1,00 | 1,00 | 1,00 |

`p = 10⁻⁴` trùng hoàn toàn với `p = 0` (cùng đồ thị sinh ra) nên chỉ trình bày một lần. `final` là trung vị của `final_best_mass`; cột cuối là tỉ lệ seed có `final_best_mass ≥ 0,95`. Cột `t₉₀` dùng trung vị Kaplan–Meier của **toàn bộ seed**, có tính đến censoring, theo quy ước là thời điểm đầu tiên đường sống sót không vượt 0,5; nếu trung vị chưa quan sát được thì không thay bằng trần mô phỏng. Quy ước này cũng được dùng cho từng graph ở §5.3, nhưng khác trung vị chỉ trên nhóm đạt ngưỡng tại thí nghiệm hai buồng §5.2. Ba dòng hai buồng là can thiệp rỗng: cả hai buồng đã chứa type tốt, nên `t₉₀ = 47,0` đo học cục bộ chứ không đo thời gian thông tin qua cầu.

Hình 2 đặt hai thước đo cạnh nhau.

![Hình 2. (a) Thời gian hội tụ theo thời gian relaxation. (b) Thời gian hội tụ theo đường đi trung bình; đường thẳng hiệu chuẩn trên 16 cấu hình không can thiệp. Ba điểm hai buồng là can thiệp rỗng và không được dùng làm kiểm định cho quy luật `L`.](figures/fig2_law.png)

**Giả thuyết ban đầu không đứng vững ở dạng phổ quát.** Ở đồ thị (a), mỗi họ có xu hướng tăng theo `τ`, nhưng hai họ nằm ở hai đường khác nhau: tại cùng `τ ≈ 100`, RGG cần khoảng 60 bước còn WS cần khoảng 90 bước (nội suy). Hệ số tương quan hạng Spearman giữa `t₉₀` và `τ` là 0,69 khi chỉ xét 16 cấu hình không can thiệp (đầy đủ + WS + RGG), nhưng rơi xuống 0,10 khi thêm họ hai buồng. Cú rơi này cho thấy `τ` không xếp hạng được các họ khác nhau; nó không phải bằng chứng rằng `L` là cơ chế đúng, vì họ hai buồng là can thiệp rỗng (xem §5.2).

**Đường đi trung bình mô tả được cả WS lẫn RGG.** Ở đồ thị (b), các điểm của cả WS lẫn RGG nằm quanh một đường thẳng, hiệu chuẩn trên 16 cấu hình không can thiệp:

```
t₉₀ = 44,55 + 3,45 · L,     R² = 0,91
```

Hệ số tương quan hạng với `L` là 0,83 trên tập hiệu chuẩn. Tung độ gốc 44,55 gần `t₉₀ = 44,0` đo được trên đồ thị đầy đủ; cách diễn giải nó thành thời gian học nội tại cộng thời gian lan truyền là giả thuyết, chưa phải phép đo tách hai thành phần. Độ bất định của các hệ số được phân tích ở §5.3.

Hình 3 cho thấy hình dạng đường học thay đổi thế nào khi nới dần vòng lattice.

![Hình 3. Đường học trung bình trên một số cấu hình tiêu biểu, kể cả vòng lattice p = 0.](figures/fig3_trajectories.png)

Đồ thị đầy đủ và MRD gần như trùng nhau. Khi `p` giảm, đường học kéo dài về phía phải nhưng vẫn tiến tới 1. Với vòng lattice (`p = 0`), trung vị `final` vẫn bằng 1,0 nhưng chỉ 82% seed đạt 0,95; đường trung bình đạt khoảng 0,97 ở bước 400, chậm hơn khoảng bốn lần so với đồ thị đầy đủ.

### 5.1. Độ chính xác: trung vị che mất phần đuôi

Cột `final` (trung vị của `final_best_mass`) bằng 1,00 ở **mọi** cấu hình, kể cả vòng lattice — một giá trị bão hoà, không phân biệt được cấu hình nhanh với cấu hình chậm. Vì vậy Bảng 1 thêm cột `seed final ≥ 0,95`: tỉ lệ seed mà khối lượng phương án tốt nhất ở cuối đạt ít nhất 0,95.

Đọc cột đó mới thấy bức tranh thật. Ở 17/19 topology liên thông, **100% seed** đạt 0,95. Nhưng ở hai topology chậm nhất, phần đuôi hiện ra: vòng lattice (`p = 0`) chỉ có **0,82** seed đạt 0,95, và `p = 10⁻³` có **0,89**. Đây đúng là hai cấu hình có tỉ lệ seed đạt `t₉₀ = 0,9` trong 400 bước thấp nhất (0,90 và 0,96) — cùng một nhóm seed "chậm" bị đếm qua hai thước đo khác nhau.

Kết luận cần thận trọng hơn: **cấu trúc đồ thị ảnh hưởng mạnh tới tốc độ, và ảnh hưởng rõ rệt tới độ chính xác chỉ ở những cấu hình mà thời gian học tiến sát cửa sổ 400 bước.** Với 17 topology còn lại, độ chính xác cuối kỳ thực sự không đổi; hai đồ thị RGG bị phân mảnh (không liên thông) cũng đạt 0,88–1,00 vì mỗi thành phần liên thông vẫn đủ lớn để tự học đúng. Nói cách khác, trong dải khảo sát, đồ thị **mua được độ trễ, không mua được độ đúng** — trừ khi độ trễ chạm trần mô phỏng, và khi đó cái mất là vài phần trăm seed, không phải toàn bộ quần thể.

### 5.2. Can thiệp hai buồng: thiết kế cũ là can thiệp rỗng

Ba dòng cuối Bảng 1 được thiết kế để tách `τ` khỏi `L`: hai clique 250 nút nối bằng cầu 1, 3 hoặc 6 cạnh làm `τ` tăng sáu lần trong khi `L ≈ 2,0`. Nhưng thiết kế đó **không kiểm định được** điều nó định kiểm định. Mã khởi tạo cân bằng 50 cá thể mỗi type trên toàn quần thể 500 cá thể rồi hoán vị ngẫu nhiên; xác suất buồng đích *không* chứa type tốt là `C(250,50)/C(500,50) ≈ 5,82·10⁻¹⁷`. Type tốt vì thế gần như chắc chắn có sẵn ở cả hai buồng, nên cầu không bắt buộc phải mang thông tin về type này. Giá trị `t₉₀ = 47,0` ở cả ba cầu phù hợp với quá trình học trong từng clique; nó không kiểm định riêng việc truyền thông tin qua cầu. Ba dòng này không phải bằng chứng rằng `t₉₀` bám theo `L` hay bỏ qua `τ`.

Để kiểm định đúng, đồ án chạy thí nghiệm **gieo nguồn**: type tốt chỉ tồn tại trong buồng nguồn, còn buồng đích khởi tạo đều trên 9 type còn lại. Muốn buồng đích học được, thông tin bắt buộc phải đi qua cầu. Thí nghiệm pilot dùng một graph cho mỗi độ rộng cầu và tái sử dụng cùng tập seed 0–29 giữa các cầu, do đó các phép chạy giữa cấu hình không nhất thiết độc lập. Bảng 1b và Hình 4 trình bày kết quả.

**Bảng 1b. Gieo nguồn buồng nguồn (N = 300, 600 bước, 30 seed; đo trong buồng đích).**

| Cầu | τ | L | t₅₀ | seed đạt t₅₀ | t₉₀* | seed đạt t₉₀ | seed final ≥ 0,95 |
|---|---|---|---|---|---|---|---|
| 1 | 11.324 | 2,00 | 425,5 | 0,27 | 385,0 | 0,23 | 0,17 |
| 3 | 3.774 | 1,98 | 248,0 | 0,63 | 285,0 | 0,63 | 0,57 |
| 6 | 1.887 | 1,96 | 228,5 | 0,93 | 253,0 | 0,93 | 0,83 |
| 12 | 943 | 1,93 | 132,5 | 0,93 | 186,5 | 0,93 | 0,93 |

`t₅₀` và `t₉₀*` là trung vị **có điều kiện** trên các seed không bị censored trong 600 bước; cột "seed đạt" là tỉ lệ seed thực sự vượt ngưỡng. Với cầu 1, 73% seed không đạt 0,5 trong thời gian quan sát, nên `t₅₀ = 425,5` chỉ mô tả nhóm seed đạt. Trung vị Kaplan–Meier `t₉₀` của toàn bộ seed chưa xác định với cầu 1 và bằng 169 bước với cầu 12.

Vì trung vị có điều kiện loại bỏ phần lớn seed ở cầu hẹp, Bảng 1b đọc kèm một thống kê sống sót có tính đến censoring: **RMST** tới trần 600 bước (Bảng 1b′). RMST₅₀ giảm đơn điệu từ **550,6** (cầu 1) xuống **185,8** (cầu 12); RMST₉₀ giảm từ **559,5** xuống **227,7**. Ở cầu 1, RMST₅₀ = 550,6 sát trần 600 vì phần lớn seed không đạt 0,5 trong thời gian quan sát. Với trần chung, RMST cho phép so sánh thời gian chờ bị chặn giữa các cầu dù tỉ lệ đạt khác nhau (0,27 so với 0,93); nó không ước lượng thời gian vượt ngưỡng đầy đủ sau trần.

**Bảng 1b′. RMST thời gian đạt ngưỡng trong buồng đích (trần 600 bước).**

| Cầu | RMST₅₀ | RMST₉₀ |
|---|---|---|
| 1 | 550,6 | 559,5 |
| 3 | 392,1 | 418,4 |
| 6 | 268,5 | 301,4 |
| 12 | 185,8 | 227,7 |

![Hình 4. Gieo type tốt vào buồng nguồn. (a) Tỉ lệ chọn type tốt trong buồng đích theo thời gian. (b) Trung vị và CI bootstrap 95% chỉ tính trên seed đạt ngưỡng 50% hoặc 90%; nhãn ghi số seed đạt trên tổng số. Đọc cùng RMST ở Bảng 1b′.](figures/fig4_two_room_source.png)

**Kết quả.** Trong dải `L = 1,93–1,997` — biến thiên chưa tới 4% — trung vị `t₅₀` trên nhóm đạt giảm từ 425,5 xuống 132,5 bước; tỉ lệ đạt tăng từ 0,27 lên 0,93. Hai trung vị có điều kiện mô tả các nhóm được chọn khác nhau, nên không nên dùng tỉ số của chúng làm mức thay đổi tốc độ cho toàn bộ seed. RMST₅₀ giảm từ 550,6 xuống 185,8 bước. Tỉ lệ seed kết thúc với khối lượng type tốt ≥ 0,95 tăng từ 0,17 lên 0,93. Các kết quả trên lưới này cho thấy `L` một mình không đủ mô tả khác biệt khi thay đổi số cạnh cầu; chúng chưa trực tiếp đo thông lượng qua lát cắt.

Đồng thời, trong thí nghiệm này `τ` và RMST₅₀ cùng giảm khi tăng cầu. Không thể từ đó tuyên bố `τ` vô can: cả hai thay đổi theo cùng một can thiệp là độ rộng cầu. Thí nghiệm cũng không chứng minh `τ` là cơ chế quyết định.

Trên WS và RGG đã khảo sát, `L` là một thang mô tả hữu ích cho thời gian học. Thí nghiệm gieo nguồn cho thấy cần xem xét thêm cấu trúc cầu trong phạm vi thiết kế này; cơ chế cụ thể được bàn ở §6.

### 5.3. Độ bất định của quy luật `L`

`t₉₀ = 44,55 + 3,45·L` với `R² = 0,91` cần được đọc kèm các chẩn đoán sau (mã trong `src/hive/analysis.py`, kết quả `results/law_diagnostics.json`).

**Khoảng tin cậy.** Bootstrap bằng cách resample 16 điểm cấu hình cho CI 95% của hệ số góc **3,03–6,68**. Mỗi điểm dùng một graph cố định; phép resample này phản ánh độ nhạy đối với thành phần các điểm quan sát, không tách riêng nhiễu swarm-seed hay biến thiên giữa các graph. CI tung độ gốc là 27,5–50,6; tại `L = 2`, CI của **giá trị trung bình trên đường hồi quy** là **40,8–57,1**. Đây không phải khoảng dự đoán cho một topology mới: nó không cộng nhiễu residual hay sai đặc tả mô hình. So sánh "dự đoán 51,5, thực đo 47,0" của thiết kế hai buồng cũ vì thế không đủ để kiểm định tính phổ quát của quy luật.

**Ảnh hưởng của từng điểm.** Một cấu hình — vòng lattice `WS p = 0`, `L = 42,1`, `t₉₀ = 173,0` — có **Cook's D = 9,9**, lớn hơn ngưỡng 1 một bậc và có leverage 0,73 (trung bình kỳ vọng `2p/n = 0,25`). Điểm này một mình kéo đường hồi quy. Leave-one-out cho thấy hệ số góc dao động 3,17–4,69 và `R²` 0,904–0,940; độ dốc dương trong các phép bỏ từng điểm, nhưng giá trị cụ thể của nó nhạy với điểm được bỏ. Bỏ `WS p = 0` làm độ dốc tăng khoảng 36%.

**Hiệu ứng họ.** Thêm biến chỉ báo cho các họ làm `R²` tăng lên 0,938 và hệ số WS lệch **+13,5 bước** so với baseline (`SE = 5,8`, `p = 0,039`). Trong mô hình cộng đang xét, WS chậm hơn RGG khoảng 13 bước ở cùng `L`. Tương tác âm (−2,7) chưa được xác định rõ (`p = 0,44`); kết quả này chưa chứng minh hai độ dốc bằng nhau hoặc xác lập quan hệ nhân quả.

**Bậc và hình dạng bậc.** Bậc trung bình tương quan hạng âm với `t₉₀` (Spearman −0,70, `p = 0,0026`) và degree CV tương quan Pearson −0,65 (`p = 0,007`). Thêm `log(bậc)` vào mô hình `L` tăng `R²` thêm 1,4 điểm phần trăm; thêm degree CV tăng 2,4; thêm cả hai tăng **5,2** (từ 0,909 lên 0,961). Vì vậy `L` chưa được tách khỏi bậc: các đồ thị có `L` lớn cũng là các đồ thị thưa với bậc thấp, và mô hình hiện tại không phân biệt được hai cách giải thích. Đây là lý do §5.4 kiểm tra thêm mạng đuôi nặng.

Các chẩn đoán này không phủ nhận quy luật; chúng xác định phạm vi đọc nó. `L` giải thích phần lớn biến thiên trong mẫu, nhưng độ dốc bất định, có hiệu ứng họ, và chưa tách khỏi bậc. Nên phát biểu là **"trong các họ WS và RGG đã khảo sát, `t₉₀` tăng xấp xỉ tuyến tính theo `L`"**, không phải "thời gian học bằng một hằng số nhân với `L` trên mọi đồ thị".

**Hiệu chuẩn lại trên nhiều graph seed.** Khác với bootstrap các điểm cấu hình ở trên, thí nghiệm này giữ nguyên tập cấu hình rồi resample graph và swarm seed bên trong từng cấu hình. Đây là một câu hỏi bất định khác, không phải chỉ tăng cỡ mẫu của cùng phép bootstrap. `scripts/check_graph_seeds.py` vẽ lại 10 graph cho mỗi cấu hình ngẫu nhiên (RGG được điều kiện hoá liên thông) và chạy 30 seed trên mỗi graph. Hai topology tất định — đầy đủ và WS `p = 0` — được chạy **một** graph với 300 seed độc lập. Tổng cộng **142 mẫu graph** (140 lần vẽ ngẫu nhiên và 2 topology tất định) và **4.800 lần chạy** với block swarm seed không trùng giữa các (cấu hình, graph). Số mẫu graph không chứng nhận mọi topology ngẫu nhiên đều khác nhau. Bootstrap phân tầng lấy mẫu graph trong cấu hình rồi cặp `(t₉₀, hit)` trong từng graph; mỗi graph dùng trung vị Kaplan–Meier trên toàn bộ quỹ đạo. Khi trung vị chưa quan sát được, cả `L` lẫn trung vị của graph đó đều được bỏ khỏi điểm trung bình của cấu hình; mẫu bootstrap bị bỏ nếu có cấu hình không còn median hữu hạn (`src/hive/analysis.py`, `results/graph_seeds_law.json`). Kết quả ở Hình 8:

```
t₉₀ = 44,97 + 3,37 · L,     R² = 0,92
```

CI 95% của độ dốc là **3,12–3,63**; CI tung độ gốc 43,2–46,9 và CI của giá trị trung bình trên đường hồi quy tại `L = 2` là 50,4–53,3 (ước lượng 51,7). Vòng lattice `WS p = 0` có trung vị Kaplan–Meier 169 bước trên 300 seed, so với 173 bước trên 100 seed ở Bảng 1. Các cấu hình khác cũng thay đổi mẫu graph và lịch seed; chưa có thí nghiệm đối chứng để quy chênh lệch hệ số riêng cho vòng lattice. Ở WS thưa `p = 10⁻³`, `L` có SD giữa graph 6,6 và `t₉₀` có SD 16,6 bước. **Phạm vi của CI:** bootstrap phân tầng *điều kiện hoá trên tập 16 cấu hình cố định* — nó đo bất định do vẽ graph và gieo seed trong các cấu hình đã chọn, chứ không đo độ nhạy khi thêm/bớt cấu hình hoặc họ đồ thị. Hai bootstrap lấy mẫu lại các đơn vị khác nhau nên không thể quy chênh lệch bề rộng CI hoàn toàn cho nhiễu của một lần vẽ graph. CI này cũng không giải quyết sai dạng mô hình; leave-one-family bên dưới là một chẩn đoán bổ sung.

**Cách tổng hợp ở hai đường hiệu chuẩn.** Cả hai dùng trung vị Kaplan–Meier thay vì trung vị riêng nhóm đạt ngưỡng. Bản một graph dùng một trung vị cho mỗi cấu hình; bản nhiều graph lấy trung bình các trung vị theo graph trong cấu hình, sau đó mới hồi quy. Vì vậy chúng vẫn khác đơn vị tổng hợp, mẫu graph và lịch seed; không nên diễn giải chênh lệch hệ số như một hiệu chỉnh riêng do một nguồn nhiễu.

**Độ nhạy theo họ (leave-one-family).** Vì CI phân tầng không trả lời "độ dốc phụ thuộc thế nào vào *những họ nào đang có*", ta khớp lại sau khi bỏ lần lượt từng họ (`results/law_diagnostics.json`): bỏ RGG cho độ dốc 3,30 (`R² = 0,92`), bỏ đầy đủ cho 3,43, bỏ WS cho 3,86 (`R² = 0,84`). Dao động 3,30–3,86 tương đương khoảng 16% độ dốc headline. Đây là leave-one-family trên **mẫu một graph**, không phải CI hay refit của mẫu 142 graph; nó đặt câu hỏi khác với CI phân tầng.

![Hình 8. Quy luật L hiệu chuẩn trên nhiều graph seed (10 graph mỗi cấu hình ngẫu nhiên; complete và WS p=0 là một graph tất định với 300 seed). Thanh ngang/dọc là SD giữa các graph, bằng 0 cho topology tất định. Đường thẳng và CI từ bootstrap phân tầng điều kiện hoá trên tập cấu hình cố định.](figures/fig8_graph_seeds.png)

Vòng lattice `WS p = 0` (`L = 42`) vẫn là điểm có đòn bẩy cao, và các cấu hình WS/RGG có độ lệch khác nhau so với đường khớp. Khoảng CI phân tầng phản ánh độ bất định trong tập cấu hình đã chọn, không kiểm định đầy đủ các giới hạn của dạng mô hình.

### 5.4. Kiểm tra ngoài phạm vi: mạng đuôi nặng

Ba mạng ngoài phạm vi quét chính được chạy với cùng `N = 500`, 400 bước, 100 seed: Barabási–Albert `m = 2` (đuôi nặng), BA `m = 5` (đuôi nhẹ hơn, dày hơn), và Erdős–Rényi bậc trung bình 6 (đuôi Poisson, điểm neo). Bảng 2 so trung vị Kaplan–Meier `t₉₀` đo được với dự đoán từ đường hiệu chuẩn được tính lại trực tiếp từ `results/graph_sweep.csv`, xấp xỉ `44,55 + 3,45·L`.

**Bảng 2. Kiểm tra trên mạng ngoài phạm vi quét chính (N = 500, 400 bước, 100 seed).**

| Mạng | τ | L | t₉₀ | dự đoán | residual |
|---|---|---|---|---|---|
| BA m = 2 | 5,66 | 3,79 | 59,0 | 57,6 | +1,4 |
| BA m = 5 | 2,32 | 2,77 | 51,0 | 54,1 | −3,1 |
| ER bậc 6 | 3,70 | 3,65 | 56,0 | 57,2 | −1,2 |

Trong ba graph được chọn, residual nằm khoảng −3,1 đến +1,4 bước; ba điểm chưa đủ để khẳng định quy luật phổ quát hay xu hướng chung của mạng đuôi nặng. Cần quét có hệ thống theo hệ số biến thiên bậc (degree CV) và nhiều graph seed để kết luận. ER bậc 6 được lấy mẫu **có điều kiện liên thông** (thử lại tới khi đồ thị liên thông) để `τ` và `L` cùng nghĩa với tập hiệu chuẩn; bản ER không điều kiện trước đó bị phân mảnh (`τ = ∞`, `L` chỉ đo trên các cặp đến được) nên không phải một điểm kiểm định sạch.

### 5.5. Nhân rộng thí nghiệm gieo nguồn: độ dẫn của cầu

Thí nghiệm ở §5.2 chỉ có một đồ thị cho mỗi độ rộng cầu và 30 seed. Đồ án lặp lại trên lưới `N ∈ {200, 400, 600}` × cầu `{1, 2, 4, 8, 16}`. Với mỗi ô, mọi đồ thị gồm hai clique và một tập cạnh cầu không chung đầu mút đều đẳng cấu, nên chỉ cần **một topology** và 80 trajectory có seed không trùng; toàn lưới có 1.200 trajectory. Horizon tăng theo `N` (800, 1.600 và 2.400 bước). Mã trong `src/hive/bridge.py`, kết quả trong `results/two_room_replicates*.csv`. CI bootstrap 95% lấy lại mẫu trajectory trong mỗi ô: khoảng này đo biến thiên mô phỏng trên topology clique đã chọn, không đo biến thiên giữa các topology.

`t_first` là bước đầu tiên buồng đích có ít nhất một bản sao type tốt. Để phân rã độ trễ, trên **cùng từng trajectory đạt cả hai sự kiện**, ta tính `t_first` và khoảng từ `t_first` đến lần đầu tỉ lệ thực trong buồng đích đạt 50% (`t₅₀` thô, không làm trơn); thời gian sau lần tới có thể bao gồm cả mất type rồi tái xâm nhập. Hình 5b lấy trung bình hai phần trên cùng nhóm đạt và ghi số seed đạt/tổng. Các trung vị `t₅₀` ở bảng và Hình 5a vẫn dùng ngưỡng làm trơn đối xứng như phần còn lại của báo cáo. Tỉ số `t_first/t₅₀` thô được tính trước cho từng trajectory đạt cả hai sự kiện rồi lấy trung vị; nó không phải tỉ số của hai trung vị hay thống kê cho seed bị censor. Định nghĩa `φ = cầu / (N/2)` là số cạnh cầu trên mỗi nút trong buồng. Với clique, `φ` gần `cầu/bậc`; hai đại lượng tách ra trong §5.6. Trung vị ở bảng là trung vị Kaplan–Meier, xét mọi seed và đánh dấu chưa xác định nếu đường sống sót chưa xuống 0,5 trong thời gian quan sát.

**Bảng 1c. Gieo nguồn nhân rộng (một topology clique × 80 seed mỗi ô; trung vị Kaplan–Meier và CI bootstrap 95% theo trajectory).**

| N | Cầu | φ | t_first | t₅₀ | seed đạt t₅₀ | seed final ≥ 0,95 |
|---|---|---|---|---|---|---|
| 200 | 1 | 0,010 | 85 [64–108] | chưa xác định [718–∞] | 0,44 | 0,36 |
| 200 | 2 | 0,020 | 66 [52–78] | 488 [350–678] | 0,71 | 0,66 |
| 200 | 4 | 0,040 | 33 [26–44] | 269 [190–322] | 0,86 | 0,81 |
| 200 | 8 | 0,080 | 20 [15–24] | 152 [116–177] | 0,98 | 0,96 |
| 200 | 16 | 0,160 | 13 [11–16] | 126 [105–138] | 0,98 | 0,98 |
| 400 | 1 | 0,005 | 200 [151–217] | 1.334 [1.047–∞] | 0,56 | 0,50 |
| 400 | 2 | 0,010 | 89 [66–112] | 706 [565–839] | 0,81 | 0,78 |
| 400 | 4 | 0,020 | 50 [42–62] | 361 [278–468] | 0,91 | 0,90 |
| 400 | 8 | 0,040 | 35 [26–42] | 250 [174–337] | 1,00 | 1,00 |
| 400 | 16 | 0,080 | 21 [17–26] | 172 [141–231] | 1,00 | 1,00 |
| 600 | 1 | 0,003 | 230 [169–281] | chưa xác định [1.958–∞] | 0,48 | 0,43 |
| 600 | 2 | 0,007 | 100 [88–144] | 897 [660–1.103] | 0,85 | 0,83 |
| 600 | 4 | 0,013 | 69 [60–90] | 577 [474–698] | 0,95 | 0,94 |
| 600 | 8 | 0,027 | 41 [34–48] | 329 [237–397] | 1,00 | 1,00 |
| 600 | 16 | 0,053 | 30 [23–34] | 219 [181–252] | 1,00 | 1,00 |

![Hình 5. (a) `t_first` và `t₅₀` làm trơn đối xứng theo `φ`, với CI bootstrap trajectory 95% (bỏ dải nơi cận trên chưa xác định). (b) Hai phần thời gian trung bình trên cùng các trajectory đạt `t₅₀` thô: trước và sau lần tới đầu tiên; nhãn ghi số seed đạt/tổng.](figures/fig5_two_room_replicates.png)

Bốn kết quả, đều từ `results/two_room_replicates_fits.json`:

1. **Độ rộng cầu và kết quả đạt ngưỡng.** `L` dao động 1,86–2,00 trên toàn lưới. Ở `N = 400`, trung vị Kaplan–Meier `t₅₀` giảm từ 1.334 xuống 172 bước khi cầu tăng từ 1 lên 16. Ở cầu 1, hai ô `N = 200` và `N = 600` chưa quan sát được trung vị vì lần lượt chỉ 43,75% và 47,5% seed đạt ngưỡng; không gán một con số hữu hạn cho hai ô này.
2. **Hồi quy trên các trung vị quan sát được.** `t_first ∝ φ^−0,73` (CI bootstrap trajectory 95% −0,78…−0,68; `R² = 0,978`, 15 ô). Với `t₅₀`, fit trên 13 ô có trung vị hữu hạn cho số mũ −0,69 (CI −0,77…−0,63; `R² = 0,978`). Ô có trung vị chưa xác định bị loại khỏi fit và có thể thay đổi giữa các mẫu bootstrap, nên CI của fit này cần đọc thận trọng. Hồi quy `t_first` với số cầu và `N` riêng rẽ có số mũ −0,74 và +0,68; chỉ dùng số cầu cho `R² = 0,83`. Chưa đo thông lượng hay kiểm tra giả định hazard không đổi, nên không suy số mũ thông lượng từ số mũ thời gian chờ (§6).
3. **Không suy cơ chế từ một fit `t₅₀`.** Với 13 ô hữu hạn, hồi quy theo số cầu và `N` cho các số mũ −0,71 và +0,61. Hai biến dự báo còn phụ thuộc thiết kế, và fit không đo riêng đóng góp thời gian lan trong buồng.
4. **Phân rã theo từng trajectory.** Trong mỗi ô, trung vị của tỉ số `t_first/t₅₀` thô trên các trajectory đạt cả hai sự kiện nằm trong **10,5–32,4%**. Tỉ lệ trajectory đủ điều kiện trong từng ô là **43,75–100%**; phần không đạt không có thời gian `t₅₀` quan sát được nên không đưa vào phép phân rã. Thời gian sau lần tới có thể gồm nhiều lần mất type và tái xâm nhập; số lần xuất hiện và biến mất không trực tiếp đo xác suất trụ của từng lần.

Các khoảng bootstrap nói trên chỉ phản ánh biến thiên trajectory trên mỗi topology clique cố định và điều kiện trên lưới cấu hình này. Với cầu 1, cả ba khoảng CI của trung vị `t₅₀` đều có cận trên chưa xác định; không ngoại suy các fit ra ngoài dải đã khảo sát.

### 5.6. Buồng thưa: tách `τ` khỏi độ dẫn của cầu

Ở §5.5, độ rộng cầu quyết định đồng thời `τ` và `φ`, nên không thể nói đại lượng nào mới là nguyên nhân. Để tách chúng, đồ án thay mỗi clique bằng một **đồ thị đều ngẫu nhiên bậc `d`** (`d = 8` và `d = 24`) trong khi giữ nguyên `N`, độ rộng cầu và quy trình gieo nguồn. Lưới: 3 kích thước × 5 độ rộng cầu × 2 bậc × 4 graph ngẫu nhiên × 20 seed không trùng = 2.400 trajectory mới, gộp với 1.200 trajectory clique ở §5.5 thành 45 ô. CI bootstrap lấy lại mẫu graph rồi lấy lại mẫu trajectory trong graph đã chọn; với ô clique tất định chỉ lấy lại mẫu trajectory. CI phụ thuộc vào họ đồ thị, tham số và các ô đang xét, chưa đo sai số dạng hồi quy hay khả năng ngoại suy. Mã: `scripts/check_two_room_sparse.py`; kết quả: `results/two_room_sparse*.csv`.

Ta so bốn đại lượng dự đoán `t_first` và `t₅₀` bằng hồi quy log-log: `φ = cầu/(N/2)`; độ dẫn cắt cổ điển `φ_vol = cầu/((N/2)·bậc)`; số cạnh cầu trên mỗi hàng xóm `φ_deg = cầu/bậc`; và `τ`. Tất cả 45 ô có trung vị `t_first` hữu hạn; hai ô clique không có trung vị `t₅₀` quan sát được nên fit `t₅₀` dùng 43 ô.

**Bảng 1d. Hồi quy log-log theo từng đại lượng dự đoán (45 ô với `t_first`; 43 ô có trung vị `t₅₀` hữu hạn).**

| Đại lượng | `t_first`: số mũ | `t_first`: R² | `t₅₀`: số mũ | `t₅₀`: R² |
|---|---|---|---|---|
| `φ = cầu/(N/2)` | −0,58 | 0,36 | −0,43 | 0,33 |
| `φ_vol` | −0,57 | 0,91 | −0,47 | 0,93 |
| `φ_deg = cầu/bậc` | −0,63 | **0,98** | −0,50 | 0,94 |
| `τ` | +0,59 | 0,90 | +0,48 | 0,93 |

**Bảng 1e. Ở `N = 400`, cùng độ rộng cầu, so sánh buồng clique và buồng thưa (`τ` trung bình trên graph; `t_first` là trung vị Kaplan–Meier của 80 trajectory).**

| Cầu | clique: `τ` | clique: `t_first` | `d = 24`: `τ` | `d = 24`: `t_first` | `d = 8`: `τ` | `d = 8`: `t_first` |
|---|---|---|---|---|---|---|
| 1 | 20.099 | 200 | 2.606 | 33 | 1.029 | 17 |
| 2 | 10.049 | 89 | 1.303 | 19 | 514 | 12 |
| 4 | 5.024 | 50 | 652 | 14 | 257 | 9 |
| 8 | 2.512 | 35 | 326 | 11 | 128 | 5 |
| 16 | 1.256 | 21 | 162 | 7 | 64 | 3 |

![Hình 6. `t_first` theo ba đại lượng dự đoán, cho clique (tròn) và buồng đều ngẫu nhiên `d = 8` (vuông), `d = 24` (tam giác); màu theo `N`. Các điểm trong lưới tập trung quanh đường hồi quy hơn khi dùng `cầu/bậc`.](figures/fig6_two_room_sparse.png)

Ba kết quả:

1. **Chỉ dùng `φ = cầu/(N/2)` mô tả kém hơn khi thêm buồng thưa.** Ở cùng `(N, cầu)`, `φ` như nhau nhưng trung vị `t_first` của buồng `d = 8` bằng 0,08–0,31 lần clique; với `d = 24` là 0,17–0,54. `R²` của fit theo `φ` là 0,36 khi gộp ba loại buồng, so với 0,98 cho clique. Đây là so sánh mô tả trên lưới, không phải kiểm định duy nhất cho mọi topology.
2. **`cầu/bậc` dự đoán tốt nhất trong các predictor và lưới đã khảo sát.** Fit `t_first ∝ φ_deg^−0,63` có `R² = 0,98` và CI bootstrap graph/trajectory của số mũ −0,66…−0,60. Khi đưa cả `φ_deg` và `τ` vào hồi quy, số mũ lần lượt là −0,64 (CI −0,72…−0,54) và −0,02 (CI −0,09…+0,09). CI chứa 0 nghĩa là chưa phát hiện đóng góp bổ sung của `τ` trong mô hình và lưới này; nó không chứng minh tác dụng bằng 0. Trong nhóm clique, `d = 8`, `d = 24`, fit riêng theo `φ_deg` cho `R²` lần lượt 0,978 / 0,975 / 0,965. Việc đầu đích có nhiều hàng xóm và ít trọng số dành cho cầu là một giả thuyết cơ học phù hợp, chưa phải xác suất hay thông lượng đo trực tiếp.
3. **Với `t₅₀`, nhiều predictor cùng mô tả tốt 43 ô hữu hạn.** `φ_vol`, `φ_deg` và `τ` có `R²` từ 0,93 đến 0,94. Trong hồi quy chung, số mũ của `φ_deg` là −0,29 (CI −0,36…−0,22) và của `τ` là +0,21 (CI +0,15…+0,30). Đây là liên hệ điều kiện trong một dạng mô hình; CI chưa giải quyết tương quan mạnh giữa các predictor, các ô bị censor, hay chứng minh cơ chế lan truyền.

Các giới hạn: chỉ hai bậc `d`, mỗi ô thưa dùng bốn mẫu graph và mỗi ô clique dùng một topology; `τ` với `φ_deg` tương quan mạnh (hệ số log-log −0,96 trên 45 ô). Bootstrap có lấy lại mẫu graph và trajectory nhưng vẫn điều kiện trên số mẫu, các họ, các ô và dạng hồi quy đã chọn. Với `t₅₀`, hai ô clique chưa có trung vị quan sát được và bị loại khỏi fit điểm; tập ô hữu hạn trong từng mẫu bootstrap cũng có thể thay đổi.

### 5.7. Buồng không đều: vị trí đầu cầu

Với clique và đồ thị đều, mọi đầu cầu có cùng bậc, nên `cầu/bậc` không phân biệt được "bậc trung bình của buồng" với "bậc của chính các đầu cầu". Để phân biệt, mỗi buồng là một đồ thị Barabási–Albert (`m = 3`, bậc trung bình ≈ 5,9, có vài hub; bậc của các đầu cầu hub là 27–39) và cầu được gắn vào **hub**, **lá** (nút bậc thấp nhất) hoặc nút **ngẫu nhiên** của *từng* buồng, theo cả 9 tổ hợp nguồn/đích, cộng một đối chứng đều bậc 6 có cùng bậc trung bình. Lưới: `N = 400`, cầu `{1, 2, 4, 8}`, 5 graph seed × 20 swarm seed mỗi ô, 4.000 lượt mô phỏng, 40 ô. Swarm seed 0–19 được tái sử dụng giữa các graph và các ô, vì vậy không coi tất cả các lượt này là độc lập. CI lấy lại mẫu graph trong từng ô, giữ nguyên các trajectory của graph đó; phép bootstrap chưa bảo toàn đầy đủ phụ thuộc giữa các ô và không phải CI phân tầng graph rồi trajectory. Mã: `src/hive/hetero.py`, `scripts/check_two_room_hetero.py`; kết quả: `results/two_room_hetero*.csv`.

Hai đại lượng dự đoán đối đầu, cùng một phần tử ở mỗi ô: `cầu/bậc TB` là số cạnh cầu chia bậc trung bình của buồng đích (không đổi khi đổi vị trí đầu cầu); `Σ 1/bậc(đầu cầu đích)` là tổng, trên các cạnh cầu, của nghịch đảo bậc nút đích (bậc tính cả cạnh cầu). Giả thuyết cơ học ở §5.6 cho rằng nút đích "nghe" nút bên kia cầu với trọng số ≈ `1/bậc`, nên đại lượng thứ hai phải thắng, và vị trí đầu cầu *phía nguồn* hầu như không quan trọng.

**Bảng 1f. Lần tới đầu tiên theo vị trí đầu cầu phía đích (nguồn ngẫu nhiên; `t_first` là trung vị 100 lần chạy, CI theo đồ thị; "tới / mất" là số lần type tốt xuất hiện / biến mất trung bình).**

| Cầu | Đầu cầu đích | bậc đầu cầu | Σ 1/bậc | `t_first` | `t₅₀` | tới / mất |
|---|---|---|---|---|---|---|
| 1 | hub | 39,2 | 0,026 | 68 [59–79] | 231 | 2,8 / 1,9 |
| 1 | ngẫu nhiên | 4,6 | 0,223 | 14 [12–16] | 238 | 18,6 / 17,6 |
| 1 | lá | 4,0 | 0,250 | 16 [12–17] | 255 | 22,2 / 21,3 |
| 1 | đều bậc 6 | 7,0 | 0,143 | 18 [14–20] | 167 | 5,9 / 5,0 |
| 2 | hub | 37,2 | 0,054 | 46 [39–49] | 158 | 2,3 / 1,3 |
| 2 | ngẫu nhiên | 8,1 | 0,351 | 11 [7–16] | 156 | 8,9 / 7,9 |
| 2 | lá | 4,0 | 0,500 | 9 [7–11] | 190 | 15,1 / 14,2 |
| 2 | đều bậc 6 | 7,0 | 0,286 | 12,0 [10,0–13,0] | 114 | 4,0 / 3,0 |
| 4 | hub | 32,8 | 0,127 | 21 [18–25] | 113 | 1,9 / 1,0 |
| 4 | ngẫu nhiên | 5,5 | 0,827 | 5 [3–6] | 122 | 5,4 / 4,5 |
| 4 | lá | 4,0 | 1,000 | 5 [4–5] | 112 | 5,5 / 4,6 |
| 4 | đều bậc 6 | 7,0 | 0,571 | 8,0 [7,0–9,0] | 91 | 3,0 / 2,0 |
| 8 | hub | 27,2 | 0,313 | 11 [10–12] | 85 | 1,7 / 0,7 |
| 8 | ngẫu nhiên | 7,3 | 1,378 | 4 [3–5] | 81 | 3,0 / 2,0 |
| 8 | lá | 4,0 | 2,000 | 3,0 [2,0–4,0] | 95 | 4,0 / 3,0 |
| 8 | đều bậc 6 | 7,0 | 1,143 | 6 [5–6] | 69 | 2,0 / 1,0 |

![Hình 7. (a) `t_first` theo độ rộng cầu cho 9 tổ hợp vị trí đầu cầu (màu = phía đích, nét = phía nguồn) và đối chứng đều bậc 6. (b) theo `Σ 1/bậc(đầu cầu đích)`: mọi ô nằm trên một đường. (c) theo `cầu/bậc TB` của buồng: các ô cùng giá trị x trải rộng theo y.](figures/fig7_two_room_hetero.png)

Bốn kết quả:

1. **Vị trí đầu cầu phía đích dự đoán `t_first` trong lưới này.** Khi đầu cầu nguồn ngẫu nhiên và giữ cùng độ rộng cầu, đặt đầu đích vào hub làm trung vị `t_first` lớn hơn 2,75–4,86 lần so với đầu đích ngẫu nhiên (trung vị các tỉ số 4,19); đặt vào lá cho tỉ số 0,75–1,14. `Σ 1/bậc(đầu cầu đích)` mô tả `t_first` với **`R² = 0,97`** (số mũ −0,68) trên 40 ô, so với **0,44** của `cầu/bậc TB`, 0,33 của `τ` và 0,14 của `Σ 1/bậc` phía nguồn. Chỉ xét phần biến thiên *trong* cùng một độ rộng cầu bằng cách loại trung bình log theo từng độ rộng, `R²` là 0,95 với số mũ −0,68.
2. **Các predictor bổ sung cải thiện ít trong các hồi quy đã xét.** Khi thêm `cầu/bậc TB` vào cùng hồi quy, số mũ của nó là 0,00 (CI −0,04…+0,05); thêm `τ`, số mũ là +0,02 (CI −0,03…+0,06); số mũ của `Σ 1/bậc(đầu cầu đích)` vẫn −0,68 đến −0,71. Cùng mô hình, hệ số phía nguồn có số mũ +0,06 (CI +0,04…+0,09), nhỏ nhưng CI không chứa 0; vì vậy không thể nói phía nguồn vô can. Các CI này có hạn chế về seed và bootstrap như nêu ở cuối mục. Một giả thuyết phù hợp là bậc của đầu đích làm thay đổi khả năng nghe phía nguồn; thí nghiệm chưa đo trực tiếp xác suất đó.
3. **Với `t₅₀`, thứ tự các mô hình thay đổi.** `Σ 1/bậc(đầu cầu đích)` có `R² = 0,32`; `τ` cho `R² = 0,92` và `cầu/bậc TB` cho 0,88. Khi đưa cả `Σ 1/bậc` và `τ`, số mũ của `Σ 1/bậc` là −0,01 (CI −0,03…+0,01) còn của `τ` là +0,44 (CI +0,41…+0,48). Trung vị theo độ rộng cầu của tỉ số `t₅₀` so với bố trí ngẫu nhiên nằm khoảng 0,87–1,12 trên các tổ hợp nguồn/đích. Bảng 1f cho thấy với hub, số lần tới và mất trung bình đều thấp hơn so với lá/ngẫu nhiên. Những số đếm này nhất quán với giả thuyết hub khó bị xâm nhập nhưng cũng có thể giúp type mới trụ lại; chúng chưa đo riêng xác suất trụ và chưa chứng minh hai cơ chế triệt tiêu nhau.
4. **Đối chứng đều bậc 6.** Cùng bậc trung bình nhưng không có hub, đầu cầu ngẫu nhiên của BA có trung vị `t_first` nhỏ hơn ở cả bốn độ rộng cầu (14 / 11 / 5 / 4 so với 18 / 12 / 8 / 6); nhiều CI chồng nhau. Bậc trung bình của đầu cầu BA không luôn thấp hơn 7 (4,6 / 8,1 / 5,5 / 7,3), nhưng `Σ 1/bậc` luôn lớn hơn đối chứng (0,22 / 0,35 / 0,83 / 1,38 so với 0,14 / 0,29 / 0,57 / 1,14). Đây là quan sát phù hợp với vai trò của bậc đầu cầu trong lưới, chưa tự nó xác định cơ chế nhân quả.

Giới hạn: chỉ một `N = 400`, chỉ BA với `m = 3`, cầu ≤ 8, 5 graph seed mỗi ô, và `Σ 1/bậc(đầu cầu đích)` có chứa chính số cạnh cầu nên một phần `R²` 0,97 là do độ rộng cầu (phân tích *trong* từng độ rộng cầu ở trên xem xét thêm khía cạnh này). Số mũ −0,68 thay vì −1 cho thấy quan hệ `1/bậc` đơn giản chưa mô tả đầy đủ dữ liệu. CI bootstrap các graph trong từng ô và phụ thuộc vào 20 trajectory đã có cho mỗi graph; seed được tái sử dụng giữa các ô và bootstrap chưa bảo toàn cấu trúc ghép cặp giữa các ô. Không thể từ CI này suy ra bất định đầy đủ của so sánh vị trí; cần chạy lại với thiết kế seed và bootstrap ghép cặp được xác định rõ.

### 5.8. Độ nhạy theo định nghĩa thời điểm đạt ngưỡng

`t₉₀` mặc định đọc lần vượt ngưỡng của trung bình trượt *đối xứng*, vốn dùng cả dữ liệu tương lai nên có thể báo sớm hơn thời điểm tín hiệu thực xuất hiện: với `[0,0,0,1,1,1]`, ngưỡng 0,1, cửa sổ 5, phép đối xứng trả bước 1 dù giá trị khác 0 đầu tiên chỉ ở bước 3. Để định lượng mức nhạy, `scripts/check_threshold_sensitivity.py` chạy lại 16 cấu hình của quy luật `L` (`N = 500`, 400 bước, 100 seed) và thí nghiệm gieo nguồn (`N = 300`, 600 bước, 30 seed), rồi đo lại ngưỡng trên **cùng** quỹ đạo theo ba định nghĩa: đối xứng (headline), nhân quả (trung bình trượt đuôi, không nhìn trước) và thô (không làm trơn). Kết quả ở Hình 9 và `results/threshold_sensitivity_*`.

**Quy luật `L`.** Trên mẫu đang xét, cả ba định nghĩa cho đường khớp có độ dốc dương (Bảng 1g). Phép nhân quả tịnh tiến tung độ gốc **+2,0 bước** và giữ nguyên độ dốc lẫn `R²`; phép thô hạ độ dốc còn 3,18, cho thấy giá trị cụ thể có nhạy với định nghĩa ngưỡng. Kết quả này chỉ áp dụng cho các cấu hình và cửa sổ quan sát đã kiểm tra.

Bảng 1g. Quy luật `t₉₀ ~ L` khớp lại trên cùng quỹ đạo theo ba định nghĩa vượt ngưỡng (16 cấu hình, `N = 500`, 100 seed).

| Định nghĩa | Tung độ gốc | Độ dốc | R² |
|---|---|---|---|
| Đối xứng (headline) | 44,55 | 3,45 | 0,909 |
| Nhân quả | 46,55 | 3,45 | 0,909 |
| Thô | 43,92 | 3,18 | 0,904 |

**Hiệu ứng cầu.** RMST₅₀ buồng đích giảm đơn điệu theo độ rộng cầu dưới cả ba định nghĩa: cầu 1 → 12 cho **550,6 → 185,8** (đối xứng), **551,1 → 187,6** (nhân quả), **550,5 → 184,3** (thô). Tỉ lệ đạt ngưỡng bằng nhau giữa ba định nghĩa trong mẫu này (0,27 / 0,63 / 0,93 / 0,93 cho cầu 1 / 3 / 6 / 12). Trung vị có điều kiện `t₅₀` cầu 1 → 12 là 425,5 → 132,5 (đối xứng), 427,5 → 134,5 (nhân quả), 424,0 → 131,5 (thô). Tuy nhiên không phải mọi thời điểm chỉ lệch hai bước: ở cầu 3, trung vị nhóm đạt là 248 bước theo đối xứng và 233 bước theo thô, lệch 15 bước; RMST tương ứng là 392,1 và 390,6. Trung vị có điều kiện không mô tả các seed bị censor.

![Hình 9. Độ nhạy theo định nghĩa ngưỡng. Trái: quy luật L khớp theo ba định nghĩa trên cùng quỹ đạo. Phải: RMST₅₀ buồng đích theo độ rộng cầu; ba đường gần trùng nhau.](figures/fig9_threshold_sensitivity.png)

**Kết luận.** Trong lưới đã chạy, cả ba phép đo đều cho độ dốc `L` dương và RMST₅₀ giảm theo độ rộng cầu; độ dốc và một số trung vị có điều kiện vẫn thay đổi đáng kể. Thời điểm đối xứng có thể xuất hiện trước tín hiệu gốc. Khi cần lần đầu chính tín hiệu gốc đạt ngưỡng, dùng `method="raw"`; phương pháp `"causal"` không dùng dữ liệu tương lai nhưng vẫn đo đường đã làm trơn.

## 6. Thảo luận

**Kết quả được đo.** Trên mẫu WS/RGG, `L` mô tả `t₉₀` tốt hơn `τ`, đồng thời còn phụ thuộc họ đồ thị và tương quan với bậc (§5.3). Trong thí nghiệm gieo nguồn, tăng độ rộng cầu làm RMST₅₀ giảm từ 550,6 xuống 185,8 bước khi `L` chỉ đổi trong khoảng 1,93–2,00. Các thí nghiệm buồng thưa và BA cho thấy bậc đầu cầu bên đích giúp dự đoán lần tới đầu tiên. Các kết quả cho `t₅₀` cho thấy cần xét thêm động lực lan sau khi phương án tốt đã tới.

**Giả thuyết mặt trận, chưa được kiểm chứng.** Weighted voter thiên lệch theo phần thưởng, khác với bước đi ngẫu nhiên trung tính dùng để định nghĩa spectral gap. Một cách giải thích là chọn lọc tạo lan truyền có hướng, khiến khoảng cách trở thành thang thời gian quan trọng. Các phương trình Fisher–KPP [7, 8] có nghiệm sóng chạy trong những điều kiện thích hợp và cung cấp một phép tương tự cho giả thuyết này. Tuy nhiên, đồ án chưa suy ra giới hạn liên tục/Fisher–KPP từ luật cập nhật đồng bộ trên đồ thị, chưa định nghĩa và đo vị trí mặt trận hay vận tốc của nó. Tung độ gốc gần thời gian well-mixed và đường học lattice kéo dài chỉ nhất quán với giả thuyết, không đủ để kết luận vận tốc không đổi hoặc `t₉₀ ∝ L` là cơ chế đã chứng minh.

**Giả thuyết nút cổ chai và sự trụ lại.** Phân rã từ các quỹ đạo đạt cả hai mốc cho thấy phần lớn thời gian đến 50% nằm sau lần tới đầu tiên ở thí nghiệm clique (§5.5). Khoảng này gồm cả những lần phương án tốt biến mất và quay lại. Khả năng trụ lại là một giả thuyết giải thích mẫu hình quan sát được. Muốn kiểm tra giả thuyết, cần theo dõi từng lần chuyển qua cầu và xác suất duy trì sau mỗi lần. Số mũ của thời gian chờ chỉ chuyển thành số mũ thông lượng dưới những giả định bổ sung, chẳng hạn tốc độ lần tới không đổi.

**Cách đọc phù hợp.** Khoảng cách và khả năng truyền qua lát cắt là hai hướng mô tả hữu ích, nhưng hình dung “điện trở nối tiếp” chỉ là phép tương tự: thời gian sau lần tới đầu tiên không phải thuần lan trong buồng, vì còn tái xâm nhập và mất type. Bước tiếp theo là đo thông lượng theo thời gian, xác suất trụ có điều kiện theo trạng thái lúc tới, và vận tốc lan trên topology có hình học rõ. Những phép đo đó mới phân biệt được giả thuyết mặt trận, giới hạn ở cầu và động lực học thiết lập quần thể.

## 7. Hạn chế

1. **Censoring.** Hai cấu hình chậm nhất của Bảng 1 có 4–10% seed không đạt 0,9 trong 400 bước. Ở thí nghiệm gieo nguồn, censoring nặng hơn nhiều: với cầu 1, 73% seed không đạt 0,5 trong 600 bước, nên các trung vị `t₅₀`, `t₉₀` chỉ mô tả nhóm seed nhanh và phải đọc cùng tỉ lệ seed đạt. Đã báo cáo riêng tỉ lệ đạt thay vì che đi, và bổ sung RMST (Bảng 1b′) — thống kê sống sót tính trọn thời gian quan sát của seed bị censored để so sánh được giữa các cầu; `t₁₀` không dùng được vì nhiều cấu hình chạm trần 600 bước ngay từ ngưỡng 10%. RMST vẫn bị chặn trên bởi trần mô phỏng: khi phần lớn seed không đạt ngưỡng, RMST tiến sát trần và chỉ là cận dưới của thời gian thật.
2. **Cùng một kịch bản chất lượng.** Thí nghiệm đồ thị chỉ dùng `q ∈ [0,4; 0,7]`. Với dải chất lượng thấp hơn, nhiễu tương đối lớn hơn và đàn dễ kẹt ở phương án tầm trung.
3. **Mạng bậc nặng đuôi mới chỉ khảo sát sơ bộ.** Bảng 2 dùng ba mạng kiểm tra ngoài mẫu hiệu chuẩn (BA `m = 2`, BA `m = 5`, ER bậc trung bình 6) với 100 seed mỗi mạng. Cần thêm mẫu đồ thị và quét hệ số biến thiên bậc để đánh giá khả năng ngoại suy.
4. **Không có cơ chế khám phá.** Như bài báo gốc, khi một phương án biến mất khỏi quần thể thì nó không thể quay lại. Trong môi trường thay đổi theo thời gian, đàn sẽ không thích nghi được.
5. **`t₉₀` phụ thuộc ngưỡng và cửa sổ làm trơn.** Đã cố định 0,9 và 5 bước cho mọi cấu hình, nhưng các ngưỡng khác có thể cho hệ số hồi quy khác. Thêm nữa, trung bình trượt *đối xứng* dùng dữ liệu tương lai nên `t₉₀` là thời điểm vượt ngưỡng của đường đã làm trơn, có thể sớm hơn thời điểm vật lý đầu tiên; §5.8 đo lại ngưỡng trên cùng quỹ đạo theo phép làm trơn nhân quả và dữ liệu thô để định lượng mức nhạy này.
6. **Độ nhạy của quan hệ với `L`.** Bootstrap theo điểm cấu hình và bootstrap phân tầng trong tập cấu hình cố định đo hai nguồn bất định khác nhau (§5.3). Một cấu hình vòng lattice có ảnh hưởng lớn tới độ dốc; phân tích theo họ và theo bậc chưa tách được nguyên nhân hay khả năng ngoại suy.
7. **Thí nghiệm gieo nguồn.** Pilot dùng 30 seed mỗi độ rộng cầu. Bản clique dùng một topology và 80 seed độc lập mỗi ô. CI của clique phản ánh biến thiên quỹ đạo, còn buồng thưa lấy mẫu lại cả đồ thị và quỹ đạo trong từng đồ thị. Buồng thưa chỉ có hai bậc và bốn đồ thị mỗi ô, nên các biến dự đoán còn tương quan mạnh. Thí nghiệm BA chỉ dùng một kích thước, một tham số `m` và tối đa 8 cạnh cầu. Phân rã thời gian trong Hình 5 điều kiện theo hai mốc quan sát được; cần xem thêm tỷ lệ đạt và RMST khi đánh giá toàn bộ lượt chạy.
8. **Phạm vi tái lập và provenance.** Đồ án đối chiếu mã gốc và chạy lại các thí nghiệm của dự án này. Phần đối chiếu upstream chưa dùng chung mẫu ngẫu nhiên. Metadata ghi cấu hình, seed, môi trường và hash đầu vào/đầu ra. Gói `--strict` kiểm tra đủ đầu ra và liên kết hash; đối chiếu các số trong phần diễn giải vẫn cần so với bảng kết quả chạy.

## 8. Kết luận

Đồ án tái tạo quan hệ cốt lõi của Soma và cộng sự trong cấu hình well-mixed đã chọn: đường trung bình P-MCL gần đường MRD với sai lệch lớn nhất 0,018 từ bước 10; đường bầy đàn cũng gần tham chiếu. Đây không phải bản chạy lại toàn bộ upstream hay chứng minh cho mọi tham số.

Trên 16 cấu hình hiệu chuẩn, `L` có quan hệ gần tuyến tính với trung vị Kaplan–Meier của `t₉₀`. Kết quả giữ cùng chiều qua 142 mẫu đồ thị và 4.800 quỹ đạo, cũng như qua ba cách xác định lần vượt ngưỡng (§5.3 và §5.8). Hiệu ứng họ và bậc hạn chế việc ngoại suy hệ số sang mạng khác.

Khi gieo phương án tốt chỉ vào buồng nguồn, tăng cầu từ 1 lên 12 cạnh làm RMST₅₀ giảm từ 550,6 xuống 185,8 bước và tỷ lệ đạt tăng từ 0,27 lên 0,93, trong khi `L` gần như giữ nguyên. Bản nhân rộng đo lần tới đầu tiên và lần đạt 50% trong buồng đích. Các thí nghiệm buồng thưa và BA cho thấy mức độ phù hợp của thước đo phụ thuộc vào mốc đang dự đoán.

Cần đo trực tiếp thông lượng qua cầu và xác suất trụ của từng lần xâm nhập trước khi kết luận về cơ chế. Việc mở rộng sang nhiều kích thước, bậc và họ đồ thị có thể kiểm tra khả năng áp dụng của kết quả ngoài lưới đã khảo sát.

## Tài liệu tham khảo

1. Soma, K., Bouteiller, Y., Hamann, H., Beltrame, G. (2026). The hive mind is a single reinforcement learning agent. *Swarm Intelligence*, 20. https://doi.org/10.1007/s11721-026-00264-7
2. Nowak, M. A. (2006). *Evolutionary Dynamics: Exploring the Equations of Life*. Harvard University Press.
3. Ohtsuki, H., Hauert, C., Lieberman, E., Nowak, M. A. (2006). A simple rule for the evolution of cooperation on graphs and social networks. *Nature*, 441, 502–505.
4. Watts, D. J., Strogatz, S. H. (1998). Collective dynamics of small-world networks. *Nature*, 393, 440–442.
5. Penrose, M. (2003). *Random Geometric Graphs*. Oxford University Press.
6. Sutton, R. S., Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
7. Fisher, R. A. (1937). The wave of advance of advantageous genes. *Annals of Eugenics*, 7(4), 355–369.
8. Kolmogorov, A. N., Petrovsky, I. G., Piskunov, N. S. (1937). Étude de l'équation de la diffusion avec croissance de la quantité de matière et son application à un problème biologique. *Moscow University Bulletin of Mathematics*, 1, 1–25.
