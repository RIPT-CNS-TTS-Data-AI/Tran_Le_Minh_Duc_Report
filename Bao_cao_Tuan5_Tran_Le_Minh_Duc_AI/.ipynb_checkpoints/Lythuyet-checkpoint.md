# I.Ensemble Learning
## Khái niệm

**Ensemble Learning** là một kỹ thuật trong học máy (machine learning) nhằm kết hợp nhiều mô hình dự đoán đơn lẻ — thường được gọi là **"weak learners"** hoặc mô hình yếu — để tạo thành một mô hình tổng hợp mạnh mẽ hơn, được gọi là **"strong learner"**. 

Mô hình yếu là những mô hình chỉ hoạt động tốt hơn một chút so với việc đoán ngẫu nhiên, ví dụ như cây quyết định độ sâu nông (shallow decision tree). Khi được kết hợp đúng cách, những mô hình yếu này có thể bổ trợ lẫn nhau để cải thiện độ chính xác, tăng độ ổn định, và giảm khả năng overfitting hoặc underfitting.

Mục tiêu chính của Ensemble Learning là tận dụng điểm mạnh và giảm thiểu điểm yếu của từng mô hình thành phần. Thay vì cố gắng phát triển một mô hình đơn lẻ thật phức tạp, chúng ta huấn luyện nhiều mô hình đơn giản, mỗi mô hình giải quyết vấn đề theo cách riêng, sau đó tổng hợp kết quả đầu ra để đưa ra dự đoán cuối cùng.

Việc kết hợp này có thể được thực hiện theo nhiều cách như:
- Trung bình hoá đầu ra (averaging),
- Bỏ phiếu đa số (majority voting),
- Sử dụng một mô hình khác (meta-model) để học cách kết hợp đầu ra từ các mô hình con.

Khi được áp dụng đúng, Ensemble Learning giúp:
- **Tăng độ chính xác** của mô hình tổng thể.
- **Giảm phương sai (variance)** — giúp mô hình tổng hợp ít bị ảnh hưởng bởi dữ liệu nhiễu.
- **Giảm độ chệch (bias)** — bằng cách kết hợp nhiều góc nhìn khác nhau từ các mô hình yếu.
- **Tăng độ ổn định** — mô hình tổng hợp ít bị dao động khi dữ liệu đầu vào thay đổi nhẹ.

Ensemble Learning đặc biệt hiệu quả trong các bài toán phức tạp, nơi một mô hình đơn lẻ khó có thể đạt hiệu suất cao do hạn chế về kiến trúc hoặc dữ liệu. Đây là nền tảng của nhiều hệ thống hiện đại trong thực tế như:
- Nhận diện hình ảnh,
- Phân loại văn bản,
- Hệ thống gợi ý (recommendation systems),
- Các bài toán dự báo trong tài chính, y tế, v.v.
## Lý do kết hợp

Kết hợp nhiều mô hình trong Ensemble Learning không chỉ nhằm cải thiện hiệu suất mà còn giúp xây dựng các hệ thống học máy có tính ổn định và khả năng tổng quát cao hơn. Dưới đây là các lý do chính khiến kỹ thuật này trở nên hiệu quả:

- **Giảm bias hoặc variance thông qua đánh đổi bias-variance**: Một mô hình đơn lẻ có thể có xu hướng thiên lệch (bias) cao nếu nó quá đơn giản hoặc phương sai (variance) cao nếu nó quá phức tạp và nhạy cảm với nhiễu trong dữ liệu huấn luyện. Ensemble Learning cho phép cân bằng hai yếu tố này bằng cách kết hợp nhiều mô hình khác nhau — một số có thể bù đắp cho thiên lệch của các mô hình khác và ngược lại, giúp giảm thiểu lỗi tổng thể.

- **Tận dụng ưu điểm của từng mô hình đơn lẻ**: Mỗi mô hình học máy có điểm mạnh riêng. Ví dụ, cây quyết định có thể dễ diễn giải nhưng dễ overfit, trong khi hồi quy tuyến tính có khả năng tổng quát tốt nhưng không xử lý tốt mối quan hệ phi tuyến. Bằng cách kết hợp chúng, ta có thể tận dụng ưu điểm của từng loại và hạn chế nhược điểm, từ đó tạo ra mô hình toàn diện hơn.

- **Tăng tính ổn định và độ tin cậy**: Trong các tình huống thực tế, dữ liệu đầu vào có thể thay đổi hoặc chứa nhiễu. Một mô hình đơn lẻ có thể không phản ứng tốt với những thay đổi nhỏ, trong khi mô hình tổng hợp có khả năng "làm mượt" phản ứng đó, mang lại kết quả nhất quán hơn.

- **Chống overfitting và underfitting**: Nhờ vào việc sử dụng nhiều mô hình và kỹ thuật như bagging hoặc boosting, các mô hình ensemble có khả năng giảm nguy cơ học thuộc dữ liệu huấn luyện hoặc bỏ sót các mẫu quan trọng.

---

## Kỹ thuật phổ biến

Ensemble Learning có thể được thực hiện bằng nhiều chiến lược khác nhau, tùy vào cách huấn luyện và kết hợp các mô hình con. Dưới đây là các phương pháp phổ biến:

| **Phương pháp** | **Đặc điểm** | **Ví dụ điển hình** |
|-----------------|--------------|----------------------|
| **Bagging** (Bootstrap Aggregating) | Huấn luyện song song nhiều mô hình trên các tập con của dữ liệu được chọn ngẫu nhiên (có hoàn lại - bootstrap), sau đó kết hợp đầu ra thông qua trung bình (regression) hoặc bỏ phiếu đa số (classification). | Random Forest |
| **Boosting** | Huấn luyện các mô hình theo tuần tự. Mỗi mô hình mới cố gắng sửa lỗi của mô hình trước đó bằng cách tập trung hơn vào các mẫu khó. Đây là phương pháp rất hiệu quả nhưng nhạy cảm với nhiễu. | AdaBoost, XGBoost, LightGBM |
| **Stacking** (Stacked Generalization) | Kết hợp đầu ra (predictions) của nhiều mô hình con bằng cách sử dụng một mô hình "meta" học từ các dự đoán đó. Đây là kỹ thuật mạnh mẽ thường dùng trong các cuộc thi như Kaggle. | Không cố định (thường kết hợp nhiều thuật toán) |
| **Voting** | Sử dụng nhiều mô hình huấn luyện riêng biệt và kết hợp dự đoán của chúng thông qua đa số phiếu (majority voting) hoặc trung bình có trọng số (weighted average). Phương pháp này đơn giản và dễ triển khai. | VotingClassifier (sklearn) |

Mỗi kỹ thuật đều có ưu và nhược điểm riêng, và việc lựa chọn kỹ thuật phù hợp phụ thuộc vào bài toán cụ thể, độ phức tạp của dữ liệu, và yêu cầu về hiệu suất cũng như khả năng giải thích.

---

# Semi-Supervised Learning

## Khái niệm

**Semi-Supervised Learning (SSL)** là một nhánh của học máy nằm giữa học có giám sát (supervised learning) và học không giám sát (unsupervised learning). Trong SSL, mô hình được huấn luyện bằng cách sử dụng một lượng nhỏ dữ liệu **có nhãn** (labeled data) kết hợp với một lượng lớn dữ liệu **không có nhãn** (unlabeled data).

Ý tưởng chính đằng sau SSL là dữ liệu không nhãn, mặc dù không trực tiếp cung cấp thông tin phân loại, vẫn chứa các đặc trưng và cấu trúc có thể khai thác để cải thiện hiệu suất của mô hình. Điều này đặc biệt quan trọng trong thực tế, khi việc gán nhãn dữ liệu tốn nhiều thời gian, chi phí hoặc cần chuyên gia (ví dụ như trong y tế, xử lý ngôn ngữ tự nhiên, hoặc thị giác máy tính).

Semi-Supervised Learning giúp cải thiện khả năng tổng quát của mô hình mà không cần phụ thuộc hoàn toàn vào dữ liệu có nhãn.

---

## Ứng dụng

Semi-Supervised Learning được sử dụng phổ biến trong các lĩnh vực mà:
- **Chi phí gán nhãn cao**: Ví dụ, chẩn đoán hình ảnh y khoa cần chuyên gia.
- **Dữ liệu không nhãn dồi dào**: Như các bài đăng trên mạng xã hội, hình ảnh từ camera, hoặc các đoạn văn bản thu thập từ web.
- **Học tập liên tục và trong thế giới thực**: Nơi dữ liệu được thu thập liên tục nhưng không thể gán nhãn kịp thời.

Một số ví dụ ứng dụng:
- Phân loại ảnh y tế với chỉ vài trăm ảnh được bác sĩ gán nhãn.
- Phân tích cảm xúc từ bình luận mạng xã hội.
- Nhận dạng chữ viết tay hoặc giọng nói với một bộ dữ liệu nhãn nhỏ.

---

## Phương pháp cơ bản

### 1. **Pseudo Labeling**
- Mô hình ban đầu được huấn luyện trên dữ liệu có nhãn.
- Sau đó, mô hình được dùng để **gán nhãn giả** (pseudo-labels) cho dữ liệu không nhãn.
- Dữ liệu được gán nhãn này sẽ được đưa trở lại mô hình để huấn luyện tiếp, thường kết hợp với dữ liệu có nhãn ban đầu.
- Phương pháp này đơn giản và hiệu quả, nhưng phụ thuộc nhiều vào chất lượng của các nhãn giả — nếu sai, mô hình có thể học sai.

### 2. **Consistency Regularization**
- Dựa trên giả định rằng mô hình nên đưa ra **kết quả đầu ra nhất quán** khi dữ liệu được làm nhiễu nhẹ.
- Ví dụ: khi ảnh bị xoay, làm mờ hoặc thay đổi ánh sáng thì dự đoán vẫn không nên thay đổi.
- Một số phương pháp nổi bật sử dụng nguyên tắc này: **Π-model**, **Mean Teacher**, **MixMatch**, **FixMatch**.
- Đây là một kỹ thuật mạnh trong SSL hiện đại, đặc biệt là trong thị giác máy tính.

---

## Giả thuyết nền tảng

Semi-Supervised Learning dựa trên một số giả định cơ bản để có thể tận dụng dữ liệu không nhãn một cách hiệu quả:

- **Smoothness Assumption (Giả định trơn mượt)**:
  - Nếu hai điểm dữ liệu gần nhau trong không gian đầu vào, chúng nên có nhãn giống nhau.
  
- **Cluster Assumption (Giả định cụm)**:
  - Dữ liệu thuộc cùng một lớp có xu hướng tạo thành các cụm (clusters) riêng biệt. Đường ranh giới giữa các lớp nên nằm ở những vùng mật độ dữ liệu thấp.

- **Low-Density Separation**:
  - Đường ranh giới phân loại tối ưu nên đi qua vùng có mật độ dữ liệu thấp, tránh cắt ngang những vùng tập trung dữ liệu.

Các giả thuyết này giúp định hình kiến trúc và loss function trong các mô hình SSL hiện đại.
# Probabilistic Graphical Models (PGM)

## Khái niệm

**Probabilistic Graphical Models (PGMs)** là một phương pháp mạnh mẽ trong học máy và thống kê nhằm biểu diễn và suy luận các mối quan hệ phụ thuộc xác suất giữa các biến thông qua **cấu trúc đồ thị**.

PGM cung cấp một cách tiếp cận trực quan và hiệu quả để mô hình hóa các hệ thống phức tạp có chứa **sự không chắc chắn** (uncertainty). Thay vì phải định nghĩa rõ toàn bộ phân phối xác suất, chúng ta sử dụng cấu trúc đồ thị để **mã hóa các điều kiện phụ thuộc (conditional dependencies)** và **giảm thiểu số lượng tham số cần học**.

Mỗi đỉnh (node) trong đồ thị biểu diễn một biến ngẫu nhiên, và các cạnh (edge) thể hiện mối quan hệ giữa các biến. Mối quan hệ này có thể là nhân quả hoặc tương quan, tùy thuộc vào loại đồ thị sử dụng.

PGM có hai nhánh chính:
- **Bayesian Networks** (Mạng Bayes)
- **Markov Networks** (Mạng Markov)

---

## Bayesian Networks (Mạng Bayes)

### Đặc điểm:
- Là đồ thị **có hướng** và **không có chu trình** (DAG - Directed Acyclic Graph).
- Mỗi đỉnh trong đồ thị đại diện cho một biến ngẫu nhiên.
- Mỗi cung (cạnh có hướng) đại diện cho mối quan hệ nhân quả hoặc điều kiện phụ thuộc xác suất giữa các biến.
- Phân phối xác suất chung được phân rã thành tích của các phân phối có điều kiện:
  
  \[
  P(X_1, X_2, ..., X_n) = \prod_{i=1}^n P(X_i \mid \text{Parents}(X_i))
  \]

### Ứng dụng:
- Chẩn đoán y tế: mô hình hóa mối quan hệ giữa bệnh và triệu chứng.
- Hệ thống khuyến nghị có tính nhân quả.
- Mô hình hóa chuỗi sự kiện trong xử lý ngôn ngữ tự nhiên hoặc phân tích rủi ro.

### Ví dụ:
- Nếu biến A ảnh hưởng đến biến B, thì trong mạng Bayes sẽ có một mũi tên từ A → B, và \( P(B \mid A) \) sẽ được định nghĩa.

---

## Markov Networks (Mạng Markov)

### Đặc điểm:
- Là đồ thị **vô hướng**.
- Thay vì mô hình hóa quan hệ nhân quả, Markov Networks biểu diễn **mối quan hệ tương quan** giữa các biến.
- Phân phối xác suất chung được mô hình hóa thông qua **hàm tiềm năng (potential functions)** trên các cliques trong đồ thị:

  \[
  P(X_1, ..., X_n) = \frac{1}{Z} \prod_{C \in \text{Cliques}} \phi_C(X_C)
  \]
  
  Trong đó:
  - \( \phi_C \) là hàm tiềm năng trên clique C.
  - \( Z \) là hệ số chuẩn hóa (partition function).

### Ứng dụng:
- Thị giác máy tính: mô hình hóa mối quan hệ giữa các pixel hoặc vùng ảnh.
- Phân tích mạng xã hội: biểu diễn tương quan giữa người dùng.
- Xử lý ngôn ngữ tự nhiên: mô hình Markov ngầm (HMM) là ví dụ đơn giản của PGM.

### Ví dụ:
- Nếu ba biến X, Y, Z có quan hệ tương quan và không có hướng cụ thể, chúng sẽ được kết nối bởi các cạnh vô hướng như X—Y—Z.

---

## So sánh nhanh

| Đặc điểm | Bayesian Networks | Markov Networks |
|----------|-------------------|-----------------|
| Kiểu đồ thị | Có hướng (DAG) | Vô hướng |
| Mối quan hệ | Nhân quả / Điều kiện | Tương quan |
| Mô hình hóa | \( P(X_i \mid \text{Parents}(X_i)) \) | \( \phi_C(X_C) \) với clique C |
| Ưu điểm | Diễn giải tốt, dễ hiểu mối quan hệ nhân quả | Mạnh trong mô hình hóa các mối liên kết tổng quát |
| Ứng dụng chính | Y tế, chuỗi sự kiện, NLP | Thị giác máy tính, mạng xã hội |

---
## Ứng dụng

Probabilistic Graphical Models (PGMs) có nhiều ứng dụng thực tế trong các lĩnh vực cần mô hình hóa sự không chắc chắn, đặc biệt là những lĩnh vực có dữ liệu phức tạp hoặc mang tính quan hệ cao. Dưới đây là một số ứng dụng tiêu biểu:

###  Chẩn đoán y tế
- **Bayesian Networks** được sử dụng để mô hình hóa mối quan hệ giữa các triệu chứng, nguyên nhân, và bệnh lý.
- Ví dụ: Xác suất một bệnh nhân mắc bệnh viêm phổi có thể được tính toán dựa trên các yếu tố như sốt, ho, tiền sử hút thuốc, và các xét nghiệm lâm sàng.
- Mô hình có thể được dùng để hỗ trợ ra quyết định lâm sàng, đặc biệt trong các hệ thống hỗ trợ bác sĩ.

###  Xử lý ngôn ngữ tự nhiên (NLP)
- Các mô hình như **Hidden Markov Models (HMM)** và **Conditional Random Fields (CRF)** là các dạng đặc biệt của PGMs, được dùng phổ biến trong:
  - Phân tích từ loại (POS tagging)
  - Nhận dạng thực thể (NER)
  - Dự đoán từ tiếp theo (ngôn ngữ xác suất)
- Giúp máy tính hiểu và xử lý văn bản dựa trên mối quan hệ xác suất giữa các từ/ngữ cảnh.

###  Thị giác máy tính
- Trong phân đoạn ảnh, mô hình như **Markov Random Fields** (MRFs) được sử dụng để mô hình hóa sự tương quan không gian giữa các pixel, nhằm đưa ra phân vùng ảnh chính xác hơn.
- Ứng dụng trong nhận diện khuôn mặt, phát hiện vật thể, phục hồi ảnh nhiễu.

###  Hệ thống khuyến nghị và phân tích hành vi
- Mô hình hóa mối quan hệ giữa người dùng và sản phẩm (hoặc hành vi tiêu dùng) bằng mạng Bayes để dự đoán sở thích, khả năng tương tác, hoặc nguy cơ rời bỏ dịch vụ.

###  Mạng xã hội và phân tích mạng
- Phân tích mối quan hệ và ảnh hưởng giữa các cá nhân trong mạng xã hội.
- Mô hình Markov có thể mô tả sự lan truyền thông tin hoặc hành vi giữa các nút trong mạng.

---

PGM là một trong những công cụ nền tảng để xử lý các bài toán có yếu tố không chắc chắn trong dữ liệu, đặc biệt hữu ích trong các hệ thống cần suy luận hoặc ra quyết định tự động.

## Tổng kết

Probabilistic Graphical Models cung cấp một nền tảng mạnh mẽ để biểu diễn các hệ thống phức tạp với tính không chắc chắn cao. Chúng giúp giảm số lượng tham số cần học, cải thiện khả năng suy diễn, và dễ dàng mở rộng trong các hệ thống lớn.

- **Bayesian Networks** phù hợp khi cần mô hình hóa các quan hệ có hướng, nguyên nhân - kết quả.
- **Markov Networks** mạnh hơn trong mô hình hóa các tương quan phi hướng và môi trường tương tác phức tạp.

# Recommendation Systems

## Khái niệm

**Hệ thống gợi ý (Recommendation Systems)** là một lĩnh vực quan trọng trong trí tuệ nhân tạo và học máy, với mục tiêu dự đoán và gợi ý các mục (items) mà người dùng có thể quan tâm. Chúng được ứng dụng rộng rãi trong thương mại điện tử, truyền thông, mạng xã hội, và nhiều lĩnh vực khác nhằm tăng sự tương tác, thời gian sử dụng, và doanh thu.

---

## Phương pháp cơ bản

Có hai phương pháp chính thường được sử dụng trong hệ thống gợi ý:

| **Loại**              | **Cơ chế**                                                                 | **Ví dụ điển hình**      |
|------------------------|---------------------------------------------------------------------------|---------------------------|
| **Content-Based**      | Dựa trên **đặc trưng nội dung** của sản phẩm hoặc dịch vụ. Hệ thống phân tích các thuộc tính (metadata) như thể loại, từ khóa, mô tả... để đề xuất các mục tương tự với những gì người dùng từng tương tác. | YouTube, Spotify          |
| **Collaborative Filtering** | Dựa trên **hành vi người dùng tương tự**. Hệ thống gợi ý những mục mà người dùng giống bạn đã thích, dựa trên mô hình ma trận tương tác giữa người dùng và sản phẩm. | Netflix, Amazon, TikTok  |

Ngoài ra, có các phương pháp mở rộng như:
- **Hybrid Recommender Systems**: Kết hợp cả hai phương pháp trên để tận dụng điểm mạnh của mỗi cái.
- **Knowledge-Based**: Gợi ý dựa vào các quy tắc, yêu cầu rõ ràng từ người dùng (ít dùng cho sản phẩm có dữ liệu hành vi lớn).
- **Context-Aware**: Tính đến ngữ cảnh như thời gian, vị trí, thiết bị...

---

## Ứng dụng thực tế

###  Netflix
- Netflix sử dụng mô hình **hybrid**:
  - **Collaborative Filtering** để phát hiện xu hướng xem giữa các người dùng có hành vi tương tự.
  - **Content-Based Filtering** để phân tích các đặc trưng như diễn viên, đạo diễn, thể loại phim.
- Hệ thống này giúp tạo ra các danh sách đề xuất cá nhân hóa như "Vì bạn đã xem...".

###  Amazon
- Gợi ý sản phẩm dựa trên **lịch sử mua hàng**, **sản phẩm đã xem**, và **hành vi của người dùng tương tự**.
- Hệ thống cũng gợi ý **"Frequently Bought Together"** và **"Customers who bought this also bought"**, kết hợp cả mô hình content và collaborative.

###  YouTube
- Sử dụng content-based filtering để đề xuất video tương tự dựa trên tiêu đề, từ khóa, mô tả.
- Kết hợp với dữ liệu hành vi (watch history, likes, watch time) để tối ưu hóa feed người dùng.

###  Spotify
- Dựa trên đặc trưng âm thanh (genre, tempo, energy) và hành vi nghe nhạc để tạo playlist như "Discover Weekly" hoặc "Daily Mix".

---

## Các kỹ thuật phổ biến trong Recommendation Systems

Hệ thống gợi ý hiện đại sử dụng nhiều kỹ thuật học máy và học sâu để tăng độ chính xác và khả năng cá nhân hóa. Dưới đây là một số kỹ thuật tiêu biểu:

---

###  1. Matrix Factorization (Phân rã ma trận)

**Ý tưởng**: Biểu diễn ma trận tương tác người dùng - sản phẩm dưới dạng tích của hai ma trận hạng thấp:
- Một ma trận biểu diễn đặc trưng người dùng
- Một ma trận biểu diễn đặc trưng sản phẩm

Sau khi học xong, tích giữa hai vector sẽ biểu diễn "mức độ yêu thích" mà người dùng dành cho sản phẩm đó.

**Công thức**:
\[
R \approx U \cdot V^T
\]
Trong đó:
- \( R \): ma trận đánh giá gốc (user × item)
- \( U \): ma trận đặc trưng người dùng
- \( V \): ma trận đặc trưng sản phẩm

**Ưu điểm**:
- Giảm chiều hiệu quả
- Có thể khái quát cho các mục chưa từng tương tác

**Thuật toán phổ biến**:
- **SVD (Singular Value Decomposition)**
- **ALS (Alternating Least Squares)**
- **FunkSVD** (không cần ma trận đầy đủ)

---

###  2. Deep Learning-Based Recommendation

Học sâu giúp xử lý tốt dữ liệu phi cấu trúc (văn bản, ảnh, âm thanh), cũng như các mối quan hệ phi tuyến tính phức tạp. Một số kiến trúc đáng chú ý:

#### a. Neural Collaborative Filtering (NCF)
- Mô hình hóa tương tác giữa người dùng và sản phẩm qua mạng nơ-ron nhiều tầng.
- Tổng quát hóa Collaborative Filtering bằng cách học biểu diễn phi tuyến tính.

#### b. Autoencoders
- Dùng để xây dựng mô hình tái tạo tương tác người dùng, học biểu diễn nén (latent) của họ.
- **Denoising Autoencoders** đặc biệt hữu ích khi dữ liệu thưa.

#### c. Sequence-Aware Recommendation (RNN/LSTM)
- Dành cho bài toán **gợi ý theo chuỗi hành vi thời gian** (ví dụ: gợi ý video tiếp theo).
- RNN hoặc LSTM giúp mô hình nhớ được lịch sử sử dụng của người dùng.

#### d. Transformers for Recommendations
- Áp dụng kiến trúc như **BERT** để học ngữ cảnh hành vi người dùng sâu hơn.
- Ví dụ: **SASRec**, **BERT4Rec**

---

###  3. Graph-Based Recommendation

Sử dụng **đồ thị** để biểu diễn mối quan hệ giữa người dùng – sản phẩm – đặc trưng. Các mô hình dựa trên **Graph Neural Networks (GNN)** có thể khai thác cấu trúc đồ thị hiệu quả.

- Ví dụ: **PinSage** (Pinterest), **GraphSAGE**, **LightGCN**
- Mô hình hóa ảnh hưởng lan truyền (propagation) từ sản phẩm tới người dùng và ngược lại.

---

###  4. Context-Aware & Reinforcement Learning

- **Context-Aware**: Mô hình xem xét các yếu tố như vị trí, thời gian, thiết bị, tâm trạng...
- **Reinforcement Learning**: Gợi ý theo kiểu thử–sai, tối ưu trải nghiệm dài hạn (ví dụ: gợi ý video để giữ người dùng càng lâu càng tốt).

---

## Tổng kết kỹ thuật

| Phương pháp | Đặc điểm chính | Mạnh ở đâu |
|-------------|----------------|------------|
| Matrix Factorization | Học biểu diễn người dùng & sản phẩm tuyến tính | Tốt với dữ liệu đánh giá thưa |
| Neural Collaborative Filtering | Học tương tác phi tuyến giữa người dùng & sản phẩm | Tùy biến cao, chính xác hơn |
| Autoencoders | Học nén và tái tạo hành vi | Phù hợp hệ thống sparse |
| RNN/LSTM | Gợi ý theo chuỗi hành vi | Ứng dụng thời gian thực |
| Transformers | Hiểu sâu hành vi và ngữ cảnh | Gợi ý chính xác theo phiên |
| GNNs | Mô hình hóa quan hệ mạng lưới | Mạnh trong ứng dụng quy mô lớn |

---

Việc chọn mô hình nào phụ thuộc vào:
- Dữ liệu sẵn có (tĩnh, chuỗi, phi cấu trúc…)
- Tính chất hệ thống (cần thời gian thực, cần cá nhân hóa sâu…)
- Tài nguyên tính toán (deep learning tốn nhiều GPU)


## Tổng kết

Hệ thống gợi ý là thành phần cốt lõi trong trải nghiệm người dùng hiện đại. Việc kết hợp dữ liệu người dùng, đặc trưng sản phẩm, và ngữ cảnh sử dụng cho phép xây dựng các hệ thống cá nhân hóa thông minh, tăng tương tác và tối ưu hóa doanh thu.

Việc lựa chọn phương pháp phụ thuộc vào:
- Quy mô dữ liệu
- Mức độ sẵn có của dữ liệu hành vi
- Tính đa dạng của sản phẩm
- Yêu cầu về thời gian thực và tính cá nhân hóa

# MLOps – Machine Learning Operations

## Khái Niệm
MLOps (Machine Learning Operations) là tập hợp các quy trình, công cụ và nguyên tắc nhằm tự động hóa toàn bộ vòng đời của mô hình học máy. Điều này bao gồm từ các giai đoạn phát triển (development), triển khai (deployment), cho đến việc duy trì và theo dõi mô hình (monitoring & retraining). MLOps đóng vai trò là cầu nối giữa các nhóm **Data Science** và **DevOps**, giúp đưa mô hình vào môi trường sản xuất một cách ổn định, an toàn và có khả năng mở rộng.

Giống như **DevOps** trong phần mềm truyền thống, MLOps đảm bảo sự lặp lại, theo dõi và cải tiến liên tục trong quá trình vận hành mô hình học máy (ML). Điều này cực kỳ quan trọng để duy trì hiệu suất và tính ổn định của mô hình sau khi đã triển khai vào sản xuất.

## Tại Sao Cần MLOps?

Dưới đây là một số lý do chính tại sao MLOps lại trở nên quan trọng đối với các mô hình học máy:

1. **Tỉ Lệ Thành Công Thấp**: Chỉ khoảng 20% các mô hình học máy (ML) được triển khai thành công vào môi trường thực tế. Nhiều mô hình chỉ dừng lại ở giai đoạn nghiên cứu mà không bao giờ được triển khai.
   
2. **Không Có Quy Trình Rõ Ràng**: Việc chuyển giao mô hình từ môi trường phát triển (ví dụ như Jupyter Notebook) vào môi trường sản xuất là một thách thức lớn. Mặc dù mô hình có thể hoạt động tốt trong môi trường phát triển, nhưng khi chuyển sang sản xuất, nó có thể gặp phải các vấn đề về hiệu suất và khả năng mở rộng.

3. **Data Drift và Versioning**: Nếu không kiểm soát được sự thay đổi dữ liệu (data drift) và không quản lý được phiên bản của mô hình, mô hình có thể nhanh chóng mất hiệu quả. Điều này đặc biệt quan trọng trong các ứng dụng thực tế, nơi dữ liệu có thể thay đổi theo thời gian.

4. **Thách Thức Scaling**: Việc mở rộng mô hình học máy để phục vụ cho nhiều người dùng hoặc nhiều thiết bị có thể là một thách thức kỹ thuật lớn. Hệ thống cần phải có khả năng mở rộng linh hoạt và đáp ứng nhanh chóng trong các tình huống với khối lượng công việc lớn.

### MLOps Giải Quyết Những Vấn Đề Gì?

MLOps giúp giải quyết những vấn đề này thông qua:

- **Tự động hóa việc huấn luyện và triển khai**: Việc huấn luyện mô hình và triển khai mô hình vào môi trường thực tế được tự động hóa, giúp giảm thiểu sự can thiệp thủ công và tăng tốc độ triển khai.
  
- **Quản lý mô hình và phiên bản**: MLOps giúp theo dõi và quản lý các phiên bản mô hình, đảm bảo rằng mô hình luôn được cập nhật và duy trì trong môi trường sản xuất.

- **Theo dõi hiệu suất mô hình**: MLOps cung cấp công cụ để theo dõi hiệu suất của mô hình theo thời gian, phát hiện kịp thời các vấn đề như data drift hoặc concept drift.

- **Cập nhật mô hình khi có dữ liệu mới**: Khi có dữ liệu mới, hệ thống MLOps có thể tự động kích hoạt việc huấn luyện lại mô hình để đảm bảo mô hình luôn cập nhật với dữ liệu mới nhất.

## Thành Phần Chính Của MLOps Pipeline

Một pipeline MLOps thường bao gồm các bước sau:

### 1. **Model Training**

MLOps giúp tự động hóa quá trình huấn luyện mô hình từ dữ liệu đến việc huấn luyện lại. Các bước trong pipeline dữ liệu được tự động hóa, giúp tối ưu hóa việc quản lý mô hình và dữ liệu. MLOps cũng hỗ trợ logging, tracking và reproducibility, đảm bảo rằng mô hình có thể được tái tạo một cách chính xác.

**Công cụ phổ biến**:
- **TFX** (TensorFlow Extended): Dành cho pipeline TensorFlow.
- **MLflow**: Quản lý và theo dõi lifecycle của mô hình.
- **Metaflow**, **Weights & Biases** (W&B): Cung cấp các công cụ tracking và quản lý cho các mô hình học máy.

### 2. **Deployment**

Việc triển khai mô hình vào môi trường sản xuất là một trong những bước quan trọng trong MLOps. Mô hình cần được đóng gói thành các API hoặc container để có thể triển khai vào các hệ thống backend, đồng thời đảm bảo khả năng mở rộng.

**Hình thức triển khai**:
- **API Endpoint**: Flask, FastAPI, gRPC.
- **Container hóa**: Docker, Kubernetes.
- **Model Serving**: TensorFlow Serving, TorchServe, KServe.

### 3. **Monitoring**

Giám sát hiệu suất mô hình là một yếu tố quan trọng trong MLOps. Quá trình giám sát giúp phát hiện các vấn đề như data drift và concept drift, đồng thời theo dõi các chỉ số quan trọng như accuracy, latency và số lượng dự đoán lỗi.

**Công cụ phổ biến**:
- **Prometheus + Grafana**: Theo dõi các metric thời gian thực.
- **Evidently AI**: Phát hiện drift và sinh báo cáo tự động.
- **WhyLabs**, **Seldon Alibi**, **Arize AI**: Các công cụ chuyên biệt để theo dõi và giám sát mô hình.

### 4. **Retraining**

Khi dữ liệu mới đến hoặc mô hình không còn hiệu quả, hệ thống MLOps có thể kích hoạt quá trình huấn luyện lại mô hình tự động. Việc huấn luyện lại này giúp cải thiện hiệu suất của mô hình và đảm bảo rằng mô hình luôn được cập nhật với dữ liệu mới.

**Công cụ phổ biến**:
- **MLflow**, **Neptune.ai**, **Weights & Biases** (W&B): Cung cấp khả năng theo dõi và quản lý việc huấn luyện lại mô hình.

## Các Công Cụ Phổ Biến Trong MLOps

Dưới đây là các công cụ phổ biến trong từng thành phần của MLOps:

### CI/CD:
- **Jenkins**, **GitLab CI/CD**, **GitHub Actions**: Tự động hóa pipeline từ mã nguồn đến triển khai.

### Orchestration:
- **Kubeflow**, **Airflow**, **Prefect**: Điều phối các bước trong workflow của học máy.

### Tracking:
- **MLflow**, **Neptune.ai**, **Weights & Biases**: Theo dõi các thông số huấn luyện và log mô hình.

### Serving:
- **TensorFlow Serving**, **TorchServe**, **KServe**: Triển khai mô hình dưới dạng service có thể gọi từ các hệ thống khác.

### Monitoring:
- **Prometheus**, **Grafana**, **Evidently**, **Arize**: Giám sát mô hình và phát hiện drift dữ liệu.

### Feature Store:
- **Feast**, **Tecton**: Quản lý và chia sẻ các đặc trưng đầu vào của mô hình.

## Tổng Kết

MLOps không chỉ là một tập hợp các công cụ, mà là một văn hóa và quy trình chuẩn hóa giúp đảm bảo các mô hình học máy không chỉ dừng lại ở các nguyên mẫu, mà có thể được triển khai và duy trì trong môi trường sản xuất một cách hiệu quả và bền vững. Áp dụng MLOps giúp:

- Tăng tốc thời gian đưa mô hình vào thị trường.
- Giảm thiểu rủi ro liên quan đến dữ liệu và drift.
- Cải thiện khả năng cộng tác giữa các nhóm khoa học dữ liệu và kỹ thuật.

Như vậy, MLOps là yếu tố quan trọng giúp phát triển và duy trì các mô hình học máy trong môi trường sản xuất, đảm bảo tính ổn định, mở rộng và hiệu quả.
