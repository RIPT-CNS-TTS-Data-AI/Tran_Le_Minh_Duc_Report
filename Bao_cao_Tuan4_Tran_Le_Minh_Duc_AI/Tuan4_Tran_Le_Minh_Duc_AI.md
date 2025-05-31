# **I. Supervised Learning là gì?**
Supervised learning là việc sử dụng những quan sát có dán nhãn (labeled data) để “dạy” các chương trình machine learning đưa ra dự báo chính xác. Nói cách khác, supervised learning là thuật toán dự đoán đầu ra (outcome) của một dữ liệu mới (new input) dựa trên các cặp (input, outcome) đã biết từ trước. Cặp dữ liệu này còn được gọi là quan sát có dán nhãn (data, label), tức (dữ liệu, nhãn).
Nguồn: https://knowledge.sapp.edu.vn/knowledge/t%E1%BB%95ng-h%E1%BB%A3p-c%C3%A1c-ki%E1%BA%BFn-th%E1%BB%A9c-c%C6%A1-b%E1%BA%A3n-module-6-machine-learning

**Nguyên lý hoạt động của học có giám sát:**
Hiểu đơn giản, Supervised Learning học từ một tập dữ liệu huấn luyện có gán nhãn, trong đó mỗi dữ liệu đầu vào đều có đầu ra tương ứng. Qua quá trình huấn luyện, mô hình dần nhận diện mối quan hệ giữa chúng. Sau đó, mô hình được kiểm tra và tối ưu hóa để có thể đưa ra dự đoán chính xác trên dữ liệu mới. Dưới đây là cách hoạt động của học có giám sát:

- Trước khi bắt đầu đào tạo, các nhà khoa học dữ liệu tạo ra tập dữ liệu huấn luyện, trong đó mỗi dữ liệu đầu vào đều đi kèm với nhãn chính xác. Ví dụ, nếu muốn dạy mô hình nhận diện mèo và chó, ta sẽ cung cấp một tập hợp hình ảnh của cả hai loài, kèm theo nhãn xác định từng hình là "mèo" hay "chó". Mô hình sẽ học cách phân biệt đặc điểm của hai loài này.

- Trong quá trình huấn luyện, thuật toán của mô hình xử lý một lượng lớn dữ liệu để tìm ra mối quan hệ tiềm ẩn giữa đầu vào và đầu ra. Sau đó, hiệu suất của mô hình được đánh giá bằng tập dữ liệu kiểm tra để xác định xem mô hình đã được huấn luyện thành công hay chưa. Xác thực chéo (Cross-validation) là quá trình kiểm tra mô hình bằng một phần khác của tập dữ liệu để đảm bảo độ chính xác và khả năng tổng quát hóa.
Nguồn: https://vnptai.io/vi/blog/detail/supervised-learning-la-gi
**Nguyên lý hoạt động của học có giám sát**
*Mô hình về nguyên lý hoạt động của Supervised Learning*
- Để mô hình hoạt động tốt hơn, cần có các thuật toán tối ưu hóa. Trong đó, nhóm thuật toán tối ưu lặp Gradient Descent hay còn gọi là Thuật toán giảm độ dốc, bao gồm cả biến thể Stochastic Gradient Descent - SGD, là những thuật toán tối ưu hóa phổ biến nhất khi huấn luyện mạng nơ-ron và các mô hình học máy khác.

- Thuật toán tối ưu hóa đánh giá độ chính xác thông qua hàm mất mát (Loss Function) – một phương trình đo lường sự khác biệt giữa dự đoán của Supervised Learning và giá trị thực tế. Độ dốc của hàm mất mát là chỉ số quan trọng để đánh giá hiệu suất của mô hình. Thuật toán tối ưu hóa sẽ giảm dần độ dốc để tối thiểu hóa sai số, đồng thời liên tục cập nhật các tham số để cải thiện mô hình trong suốt quá trình huấn luyện. Nhờ quá trình huấn luyện và tối ưu hóa không ngừng, các mô hình ngày càng thông minh và hiệu quả hơn.
Nguồn: https://vnptai.io/vi/blog/detail/supervised-learning-la-gi
## **1.Mục tiêu và quy trình tổng quát**
### **a.Mục tiêu** 
 - Là tìm ra một hàm ánh xạ từ đầu vào đến đầu ra để có thể dự đoán chính xác nhãn cho các ví dụ chưa thấy.
 Nguồn: https://www.linkedin.com/pulse/supervised-learning-ph%C6%B0%C6%A1ng-ph%C3%A1p-h%E1%BB%8Dc-gi%C3%A1m-s%C3%A1t-tien-thanh-nguyen-zsruc
### **b.Quy trình tổng quát**
1. Thu thập và chuẩn bị dữ liệu: Tập hợp dữ liệu có nhãn từ các nguồn khác nhau và chuẩn bị chúng cho quá trình huấn luyện.

2. Chia dữ liệu: Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra. Tập huấn luyện được sử dụng để đào tạo mô hình, còn tập kiểm tra để đánh giá hiệu suất của mô hình.

3. Lựa chọn và huấn luyện mô hình: Chọn một thuật toán học máy phù hợp (như hồi quy logistic, cây quyết định, v.v.) và huấn luyện mô hình trên tập huấn luyện.

4. Đánh giá mô hình: Kiểm tra mô hình trên tập kiểm tra để đo lường độ chính xác và khả năng tổng quát hóa của mô hình.

5. Triển khai và cải tiến: Triển khai mô hình đã được huấn luyện vào thực tế và liên tục cải tiến dựa trên phản hồi và hiệu suất thực tế.

Nguồn: https://www.linkedin.com/pulse/supervised-learning-ph%C6%B0%C6%A1ng-ph%C3%A1p-h%E1%BB%8Dc-gi%C3%A1m-s%C3%A1t-tien-thanh-nguyen-zsruc

### **c.Phân biệt Classification vs Regression**
#### ***Classification (Phân Loại) là gì?***
- Classification (phân loại) là một bài toán supervised learning (học có giám sát) trong machine learning, với mục tiêu dự đoán nhãn lớp (class label) rời rạc của một đối tượng dữ liệu dựa trên các đặc trưng (features) của đối tượng đó. Nói một cách đơn giản, đó là việc gán một đối tượng vào một trong các nhóm (lớp) đã được xác định trước.

- Ví dụ, trong bài toán phân loại email, chúng ta muốn xác định xem một email là “spam” hay “không spam” (hai lớp). Các đặc trưng có thể là từ ngữ xuất hiện trong email, địa chỉ người gửi, tiêu đề email,… Mô hình classification sẽ học từ dữ liệu huấn luyện (các email đã được gán nhãn) để tìm ra mối quan hệ giữa các đặc trưng và nhãn lớp, từ đó dự đoán nhãn cho các email mới.
**Các loại Classification**
- *Binary Classification (Phân loại nhị phân)*
Binary Classification là bài toán phân loại trong đó đối tượng dữ liệu chỉ có thể thuộc về một trong hai lớp. Hai lớp này thường được gọi là “positive” (lớp khẳng định) và “negative” (lớp phủ định), hoặc đơn giản là 0 và 1.
Đây là dạng phân loại phổ biến và cơ bản nhất. Nhiều bài toán thực tế có thể được đưa về dạng binary classification. Đôi khi, tôi thấy việc đơn giản hóa bài toán về dạng nhị phân lại mang lại hiệu quả bất ngờ, giúp mô hình tập trung vào việc phân biệt hai lớp quan trọng nhất.
- *Multiclass Classification (Phân loại đa lớp)*
Multiclass Classification là bài toán phân loại trong đó đối tượng dữ liệu có thể thuộc về một trong nhiều hơn hai lớp. Các lớp này là loại trừ lẫn nhau (mutually exclusive), nghĩa là mỗi đối tượng chỉ thuộc về một lớp duy nhất.
Khác với binary classification chỉ có hai lựa chọn, multiclass classification mở ra một không gian rộng lớn hơn với nhiều khả năng hơn. Điều này đòi hỏi các thuật toán phải có khả năng phân biệt giữa nhiều lớp khác nhau, thường là phức tạp hơn.
- *Multi-label Classification (Phân loại đa nhãn)*
Multi-label Classification là bài toán phân loại trong đó một đối tượng dữ liệu có thể thuộc về nhiều lớp cùng một lúc. Các lớp không loại trừ lẫn nhau.
Đây là dạng phân loại phức tạp nhất, phản ánh đúng thực tế khi một đối tượng thường có nhiều thuộc tính, nhiều khía cạnh khác nhau. Nó đòi hỏi các thuật toán phải có khả năng “nhìn” đối tượng từ nhiều góc độ.

Nguồn: https://interdata.vn/blog/classification-la-gi/

#### ***Regression (Hồi quy) là gì?***
- Hồi quy là một phương pháp được sử dụng trong Machine Learning nhằm mục đích dự đoán giá trị liên tục. Ví dụ, nó có thể được sử dụng để dự đoán giá nhà, nhiệt độ, doanh thu bán hàng, hoặc bất kỳ giá trị nào khác có thể đo lường được.

- *Các loại hồi quy mà chúng ta có thể kể đến như:*
Hồi quy tuyến tính (Linear Regression): Mô hình đơn giản nhất, trong đó mối quan hệ giữa biến độc lập và biến phụ thuộc được biểu diễn bằng một đường thẳng.
Hồi quy phi tuyến (Non-linear Regression): Khi mối quan hệ giữa các biến không thể biểu diễn bằng một đường thẳng.
Hồi quy logistic (Logistic Regression): Dù có tên là hồi quy, trên thực tế nó được dùng cho các bài toán phân loại.
- *Các mô hình hồi quy được sử dụng nhiều hiện nay*

- Trong hồi quy, mô hình sẽ tìm kiếm những mối quan hệ giữa các dữ liệu đầu vào (biến độc lập) và giá trị đầu ra (biến phụ thuộc) bằng cách tối ưu hóa một hàm mất mát, thường là hàm bình phương của sai số giữa giá trị dự đoán và giá trị thực tế.
#### ***So sánh Regression và Classification trong Machine Learning***
So sánh hai phương pháp Hồi quy và Phân loại theo các tiêu chí được đề cập trong bảng dưới đây:


| Tiêu chí                  | Regression (Hồi quy)                                               | Classification (Phân loại)                                    |
|----------------------------|--------------------------------------------------------------------|---------------------------------------------------------------|
| **Mục tiêu**               | Dự đoán một giá trị liên tục                                       | Dự đoán một nhãn thuộc về một lớp rời rạc                     |
| **Đầu ra**                 | Giá trị liên tục (số thực, chẳng hạn như giá cả, nhiệt độ)         | Nhãn thuộc về một hoặc nhiều lớp (ví dụ: Có/Không, Loại 1/Loại 2/Loại 3) |
| **Ví dụ**                  | Dự đoán giá nhà                                                   | Phân loại email thành thư rác hoặc không phải thư rác         |
| **Đặc trưng đầu vào**      | Thường là biến số liên tục                                          | Có thể là biến số liên tục hoặc rời rạc                      |
| **Các thuật toán**         | Hồi quy tuyến tính, hồi quy logistic (được dùng cả trong hồi quy và phân loại) | Cây quyết định, máy vector hỗ trợ (SVM), mạng nơ-ron         |
| **Phạm vi ứng dụng**       | Dự báo tài chính, khí tượng học                                   | An ninh mạng, nhận dạng hình ảnh, phân loại văn bản           |
| **Phân loại cụ thể**       | Hồi quy với nhiều biến số được gọi là hồi quy đa biến             | Phân loại nhị phân (2 lớp), phân loại đa lớp (nhiều hơn 2 lớp) |

---
- Classification: Đầu ra là các nhãn rời rạc, ví dụ như phân loại email thành "spam" hoặc "không spam", nhận diện hình ảnh là "chó", "mèo" hoặc "chim".
- Regression: Đầu ra là giá trị liên tục, ví dụ như dự đoán giá nhà, dự báo doanh thu, dự đoán nhiệt độ dựa trên các đặc trưng đầu vào.
**Về cơ bản**, Regression và Classification là hai phương pháp có các thuật toán và input khác nhau, vì thế output và ứng dụng thực tế cũng sẽ khác nhau.
Nguồn: https://statio.vn/blog/regression-vs-classification-la-gi-so-sanh-giua-hai-phuong-phap-trong-machine-learning-cach-lua-chon-va-ung-dung-thuc-te
#### ***Cách lựa chọn giữa Regression (Hồi quy) và Classification (Phân loại)***
Theo dõi cách lựa chọn giữa Hồi quy và Phân loại chính xác, cụ thể nhất dưới đây:

**Xác định tính chất của dữ liệu**
Việc lựa chọn giữa hồi quy và phân loại phụ thuộc vào tính chất của dữ liệu mà bạn đang làm việc. Nếu bạn đang làm việc với giá trị liên tục, hồi quy là lựa chọn chính xác. Ngược lại, nếu bạn cần phân loại dữ liệu thành các nhóm khác nhau, bạn nên sử dụng phân loại.

**Mục tiêu dự đoán**
Cần xác định mục tiêu cụ thể mà bạn muốn dự đoán. Nếu mục tiêu là dự đoán một số cụ thể, hãy chọn hồi quy. Nếu bạn muốn phân loại dữ liệu vào các nhóm khác nhau, hãy chọn phân loại.

**Đánh giá kết quả**
Mỗi phương pháp sẽ có các chỉ số đánh giá riêng biệt. Hồi quy thường sử dụng các chỉ số như RMSE (Root Mean Squared Error) hoặc MAE (Mean Absolute Error), trong khi phân loại dùng Accuracy, Precision, Recall, và F1 Score.
#### Ứng dụng thực tế của Regression vs Classification
Trong các lĩnh vực thực tiễn, Regression (Hồi quy) và Classification (Phân loại) được ứng dụng tương đối nhiều. Một số ví dụ điển hình có thể kể đến:

- Ứng dụng của hồi quy: Dự đoán giá nhà (Sử dụng thông tin như diện tích, số phòng ngủ, vị trí để dự đoán giá nhà); Phân tích doanh thu (Dự đoán doanh số dựa trên các yếu tố như quảng cáo, mùa vụ, và giá cả).
- Ứng dụng của phân loại: Phân loại email (Theo thư rác và thư quan trọng); Nhận dạng hình ảnh (Phân loại hình ảnh thành các nhóm như động vật, đồ vật, người).

# **Unsupervised Learning là gì?**
Unsupervised Learning (Học không giám sát) là một lĩnh vực của học máy, trong đó các thuật toán tự động phân tích và tìm kiếm cấu trúc, mẫu hoặc mối quan hệ tiềm ẩn trong dữ liệu mà không cần đến nhãn hoặc hướng dẫn trước từ con người. Dữ liệu đầu vào là dữ liệu thô, chưa được gán nhãn, và mục tiêu là để mô hình tự phát hiện ra các đặc điểm nổi bật hoặc nhóm dữ liệu tương đồng.

## **1.Mục tiêu và Ứng dụng của Unsupervised Learning**
### **a.Mục tiêu:**

- Khám phá các cấu trúc ẩn, mẫu hoặc mối quan hệ trong dữ liệu chưa được gán nhãn.

- Nhóm các điểm dữ liệu tương tự lại với nhau hoặc giảm số chiều của dữ liệu để thuận tiện cho việc lưu trữ, tính toán, trực quan hóa hoặc tiền xử lý cho các bài toán khác.

### **b.Ứng dụng phổ biến:**

Phân khúc khách hàng (Customer Segmentation): Nhóm khách hàng theo hành vi, sở thích để tối ưu hóa chiến lược marketing.

Phát hiện bất thường (Anomaly Detection): Xác định các giao dịch gian lận, lỗi hệ thống hoặc hành vi bất thường trong dữ liệu.

Hệ thống gợi ý (Recommendation Systems): Đề xuất sản phẩm, bài hát, phim dựa trên nhóm người dùng hoặc sản phẩm tương tự.

Xử lý ngôn ngữ tự nhiên (NLP): Phân tích chủ đề, tóm tắt văn bản, phân nhóm tài liệu.

Xử lý ảnh và thị giác máy tính: Phân đoạn ảnh, nhận diện đối tượng, nén ảnh.

Tiền xử lý dữ liệu: Giảm chiều dữ liệu (ví dụ: PCA) để tăng tốc độ và hiệu quả cho các mô hình học máy khác.

Nguồn :https://meeyland.com/tin-tuc/unsupervised-learning-vi-du-va-so-sanh-voi-supervised-learning-148378175567

## **2.Phân biệt Clustering và Dimensionality Reduction**
### a.Clustering là gì?

**Clustering (Phân cụm)** trong khoa học dữ liệu là quá trình tổ chức các đối tượng thành các nhóm (cụm) sao cho các đối tượng trong cùng cụm có mức độ tương đồng cao.

Ví dụ thực tế: Giống như việc trẻ em tự phân loại mảnh ghép đồ chơi theo hình dạng hoặc màu sắc mà không cần hướng dẫn.

#### Đặc điểm của Clustering
- **Số lượng cụm không cố định**: Thường không biết trước số cụm.
- **Phương pháp đa dạng**: Nhiều kỹ thuật khác nhau, như K-means, DBSCAN, Hierarchical Clustering.
- **Kết quả biến đổi**: Các thuật toán khác nhau có thể cho ra các nhóm khác nhau.

#### Yêu cầu khi xây dựng hệ thống Clustering
- **Chọn thuật toán phù hợp**: Dựa trên loại dữ liệu và mục tiêu.
- **Tính sẵn sàng (High Availability)**: Hệ thống phải hoạt động liên tục, có sao lưu và phục hồi.
- **Độ tin cậy (Reliability)**: Chống chịu lỗi tốt, dự phòng nút và xử lý lỗi.
- **Tiền xử lý dữ liệu**: Chuẩn hóa dữ liệu, xử lý giá trị thiếu và ngoại lai.
- **Phân tích kết quả**: Đánh giá chất lượng phân cụm, tích hợp kết quả vào hệ thống hiện có, đảm bảo an toàn dữ liệu.

#### Ưu và Nhược điểm của Clustering

##### Ưu điểm
- **Khám phá mô hình dữ liệu**: Phát hiện các cấu trúc và quan hệ ẩn trong dữ liệu.
- **Khám phá dữ liệu**: Giúp phân tích các tập dữ liệu lớn trực quan và dễ hiểu hơn.
- **Học không giám sát**: Không yêu cầu dữ liệu gán nhãn, linh hoạt với nhiều loại dữ liệu.
- **Ưu điểm Server Cluster**:
  - **Độ tin cậy cao**: Tự động tiếp quản khi có máy chủ lỗi.
  - **Khả năng mở rộng**: Thêm/bớt máy chủ dễ dàng.
  - **Hiệu năng cao**: Phân phối tải, xử lý nhiều tác vụ song song.
  - **Bảo mật tốt hơn**: Phân tán dữ liệu, giảm rủi ro tấn công.
  - **Quản lý linh hoạt**: Dễ tối ưu hóa vận hành hệ thống.
  - **Tối ưu chi phí lâu dài**: Hiệu quả tài nguyên, tiết kiệm chi phí vận hành.

##### Nhược điểm
- **Tính chủ quan khi diễn giải**: Phụ thuộc thuật toán và cách chọn tham số.
- **Nhạy cảm với tham số**: Sai tham số có thể gây phân cụm sai lệch.
- **Vấn đề mở rộng quy mô**: Một số thuật toán gặp khó khăn khi xử lý dữ liệu lớn hoặc nhiều chiều.

> Mặc dù Clustering rất hữu ích trong việc khám phá và phân tích dữ liệu, cần lưu ý kỹ các hạn chế khi áp dụng thực tế.

Nguồn: https://interdata.vn/blog/clustering-la-gi/
### Dimensionality Reduction (Giảm chiều dữ liệu)

#### 1. Định nghĩa
Giảm chiều dữ liệu là quá trình chuyển dữ liệu từ không gian nhiều chiều sang không gian ít chiều hơn, vẫn giữ lại thông tin quan trọng. Đây là kỹ thuật then chốt trong học máy và phân tích dữ liệu lớn, phức tạp.

#### 2. Mục tiêu và Lợi ích
- **Đơn giản hóa mô hình**: Dễ huấn luyện, tránh overfitting.
- **Trực quan hóa dữ liệu**: Dễ biểu diễn dữ liệu đa chiều trên 2D, 3D.
- **Giảm nhiễu**: Loại bỏ đặc trưng dư thừa, nhiễu.
- **Tiết kiệm tài nguyên**: Giảm chi phí tính toán và lưu trữ.
- **Khắc phục lời nguyền chiều dữ liệu**: Cải thiện hiệu quả học máy.

#### 3. Các phương pháp Giảm chiều
- **Feature Selection (Chọn lọc đặc trưng)**: Chọn các đặc trưng quan trọng (ví dụ: Filter, Wrapper, Embedded Methods).
- **Feature Extraction/Projection (Trích xuất đặc trưng)**: Tạo đặc trưng mới từ dữ liệu gốc (ví dụ: PCA, LDA, t-SNE, UMAP, Autoencoder).

#### 4. Một số kỹ thuật phổ biến
- **PCA (Principal Component Analysis)**: Chiếu dữ liệu lên các trục có phương sai lớn nhất.
- **LDA (Linear Discriminant Analysis)**: Tối ưu hóa phân tách giữa các lớp, dùng trong phân loại.
- **t-SNE, UMAP**: Phi tuyến, trực quan hóa dữ liệu phức tạp.
- **Autoencoder**: Mạng nơ-ron học biểu diễn nén, phù hợp cho dữ liệu phi tuyến.

#### 5. Ứng dụng thực tiễn
- Trực quan hóa dữ liệu.
- Tiền xử lý cho học máy.
- Nén dữ liệu.
- Phát hiện bất thường, phân cụm, xử lý ảnh và NLP.

#### 6. Tóm tắt

| Phương pháp        | Đặc điểm chính                           | Ứng dụng tiêu biểu                     |
|--------------------|------------------------------------------|----------------------------------------|
| Feature Selection  | Chọn đặc trưng quan trọng, giảm nhiễu    | Tiền xử lý, đơn giản hóa dữ liệu       |
| PCA                | Chiếu lên trục phương sai lớn nhất       | Nén dữ liệu, trực quan hóa             |
| t-SNE, UMAP        | Kỹ thuật phi tuyến, mạnh về trực quan hóa| Biểu diễn dữ liệu phức tạp             |
| Autoencoder        | Nén dữ liệu bằng mạng nơ-ron             | Xử lý dữ liệu phi tuyến                |

---

**Kết luận**:  
Giảm chiều dữ liệu giúp đơn giản hóa, tối ưu hóa và trực quan hóa dữ liệu, nâng cao hiệu quả mô hình học máy và phân tích dữ liệu.



| Tiêu chí         | Clustering (Phân cụm)                                                                                                                                  | Dimensionality Reduction (Giảm chiều dữ liệu)                                                                                         |
|------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| **Mục tiêu**      | Nhóm các điểm dữ liệu thành các cụm dựa trên sự tương đồng nội tại giữa chúng                                                                           | Giảm số lượng đặc trưng (feature) của dữ liệu mà vẫn giữ thông tin quan trọng                                                         |
| **Phương pháp**   | Phân tích sự tương đồng để chia dữ liệu thành các nhóm (ví dụ: K-Means, DBSCAN)                                                                         | Biến đổi dữ liệu sang không gian mới có ít chiều hơn (ví dụ: PCA, t-SNE)                                                              |
| **Đầu ra**        | Nhãn cụm cho mỗi điểm dữ liệu (mỗi điểm thuộc một nhóm cụ thể)                                                                                          | Dữ liệu mới với số chiều thấp hơn, mỗi chiều là tổ hợp tuyến tính của các chiều gốc                                                   |
| **Ứng dụng**      | Phân khúc khách hàng, phát hiện bất thường, phân nhóm hình ảnh, tài liệu                                                                               | Trực quan hóa dữ liệu, nén dữ liệu, tiền xử lý cho mô hình học máy khác                                                              |
| **Ví dụ minh họa**| Nhóm khách hàng thành các phân khúc dựa trên hành vi mua sắm                                                                                           | Giảm số chiều của dữ liệu gen từ 10.000 xuống 2 hoặc 3 để trực quan hóa                                                              |

**Tóm lại:**

- Clustering tập trung vào việc nhóm các điểm dữ liệu lại với nhau dựa trên sự giống nhau.

- Dimensionality Reduction tập trung vào việc giảm số chiều của dữ liệu, giữ lại thông tin quan trọng nhất để thuận tiện cho lưu trữ, tính toán và trực quan hóa.

Nguồn: https://www.datacamp.com/tutorial/understanding-dimensionality-reduction  


-------

# **III.So sánh giữa Supervised và Unsupervised Learning**
# So sánh Supervised Learning và Unsupervised Learning

| Tiêu chí                | Supervised Learning                                    | Unsupervised Learning                                           |
|--------------------------|--------------------------------------------------------|-----------------------------------------------------------------|
| **Dữ liệu đầu vào**       | Dữ liệu có nhãn (ví dụ: ảnh gán nhãn "chó", "mèo")     | Dữ liệu không nhãn (ví dụ: giao dịch, văn bản thô)              |
| **Mục tiêu chính**        | Dự đoán đầu ra chính xác từ dữ liệu mới                | Khám phá cấu trúc ẩn, nhóm dữ liệu hoặc giảm chiều              |
| **Loại bài toán**         | Phân loại (Classification), Hồi quy (Regression)      | Phân cụm (Clustering), Giảm chiều (Dimensionality Reduction), Phát hiện luật (Association) |
| **Ứng dụng điển hình**    | Nhận diện khuôn mặt, dự báo thời tiết, chẩn đoán bệnh  | Phân khúc khách hàng, nén ảnh, hệ thống gợi ý, phát hiện gian lận |
| **Thuật toán tiêu biểu**  | Linear Regression, Decision Tree, SVM, Neural Networks| K-Means, DBSCAN, PCA, Apriori, Autoencoder                     |
| **Đánh giá kết quả**      | Accuracy, Precision, Recall, F1-Score, MSE, R²         | Silhouette Score, Davies-Bouldin Index, đánh giá định tính      |
| **Độ phức tạp dữ liệu**   | Yêu cầu dữ liệu sạch, nhãn chính xác; dễ bị overfitting| Xử lý được dữ liệu nhiễu, nhưng khó đánh giá chính xác kết quả |
| **Yêu cầu về nhãn**       | Bắt buộc có nhãn (tốn chi phí thu thập và gán nhãn)    | Không cần nhãn (tiết kiệm thời gian và chi phí)                 |
| **Khả năng giải thích**   | Một số mô hình dễ giải thích (ví dụ: Decision Tree)    | Kết quả khó giải thích, cần phân tích chuyên sâu                |
| **Xử lý nhiễu/ngoại lệ**  | Nhạy cảm với dữ liệu nhiễu                            | Một số thuật toán xử lý tốt nhiễu (ví dụ: DBSCAN)               |
| **Chi phí tính toán**     | Cao hơn do tối ưu hóa hàm mất mát                      | Thấp hơn, nhưng phụ thuộc vào kích thước và thuật toán          |
| **Tính linh hoạt**        | Kém linh hoạt, chỉ học từ nhãn đã biết                 | Linh hoạt, tự khám phá cấu trúc mới                             |
| **Xử lý dữ liệu mất cân bằng** | Cần kỹ thuật cân bằng (Oversampling, Undersampling) | Ít bị ảnh hưởng do không phụ thuộc vào phân phối nhãn          |
| **Khả năng mở rộng**      | Khó mở rộng sang miền dữ liệu khác                    | Dễ mở rộng cho dữ liệu lớn, đa dạng                             |
| **Giai đoạn tiền xử lý**  | Chuẩn hóa dữ liệu, xử lý missing values, mã hóa nhãn   | Chuẩn hóa, xử lý nhiễu, chọn số cụm hoặc số chiều giảm           |
| **Ví dụ cụ thể**          | Dự đoán giá nhà dựa trên diện tích, vị trí            | Nhóm các bài báo khoa học theo chủ đề                           |

---

> **Tóm lại**:  
> - **Supervised Learning** phù hợp cho các bài toán cần dự đoán kết quả chính xác, nhưng đòi hỏi dữ liệu gán nhãn đầy đủ.  
> - **Unsupervised Learning** phù hợp để khám phá, phân tích dữ liệu chưa biết trước nhãn, với khả năng linh hoạt cao nhưng khó đánh giá và giải thích kết quả.

Nguồn : https://tuhoclaptrinhsite.wordpress.com/2021/08/15/hieu-ro-ve-hoc-co-giam-sat-supervised-learning-va-hoc-khong-giam-sat-unsupervised-learning/

# **Giới thiệu các thuật toán tiêu biểu**
## 1.Supervised learning
### a. Linear Regression

- **Mục tiêu**:  
  Dự đoán giá trị liên tục (regression) dựa trên mối quan hệ tuyến tính giữa biến đầu vào (independent variables) và biến đầu ra (dependent variable).

- **Cách hoạt động**:  
  Tìm đường thẳng (hoặc siêu phẳng trong không gian nhiều chiều) tốt nhất để mô tả mối liên hệ giữa các biến, bằng cách tối thiểu hóa tổng bình phương sai số giữa giá trị dự đoán và giá trị thực tế.  
  Phương trình tổng quát của Linear Regression:
  
  \[
  y = w_0 + w_1x_1 + w_2x_2 + \dots + w_nx_n
  \]

  Trong đó:
  - \( y \) là biến mục tiêu (target variable).
  - \( x_1, x_2, ..., x_n \) là các biến đầu vào (features).
  - \( w_0 \) là hệ số tự do (intercept), và \( w_1, w_2, ..., w_n \) là các hệ số hồi quy (weights).

- **Ưu điểm**:
  - Đơn giản, dễ hiểu, dễ triển khai.
  - Tính toán nhanh, phù hợp với dữ liệu tuyến tính.

- **Nhược điểm**:
  - Hiệu quả kém khi dữ liệu có quan hệ phi tuyến.
  - Nhạy cảm với ngoại lai (outliers).

- **Ứng dụng**:
  - Dự báo giá nhà, doanh thu, xu hướng thị trường, lượng tiêu thụ sản phẩm.

 Nguồn:https://pro.arcgis.com/en/pro-app/latest/tool-reference/geoai/how-linear-regression-works.htm

---

### b. Decision Tree

- **Mục tiêu**:  
  Sử dụng cho cả hai nhiệm vụ: phân loại (classification) và hồi quy (regression).

- **Cách hoạt động**:  
  Dữ liệu được phân chia theo các thuộc tính tại các nút quyết định, tạo thành cấu trúc dạng cây. Mỗi nhánh của cây tương ứng với một giá trị thuộc tính, dẫn tới nút lá chứa kết quả cuối cùng.  
  Quy trình chia tách tiếp tục cho đến khi:
  - Các nhóm đạt mức đồng nhất nhất định.
  - Hoặc đạt điều kiện dừng do giới hạn chiều sâu, số lượng mẫu tối thiểu, hoặc mức độ thuần khiết.

- **Ưu điểm**:
  - Trực quan, dễ hiểu, dễ giải thích mô hình.
  - Không yêu cầu nhiều tiền xử lý dữ liệu (không cần chuẩn hóa đặc trưng).

- **Nhược điểm**:
  - Dễ bị **overfitting** (quá khớp) nếu cây quá sâu.
  - Nhạy cảm với thay đổi nhỏ trong dữ liệu (dễ thay đổi cấu trúc cây).

- **Ứng dụng**:
  - Phân loại khách hàng, phân tích rủi ro tài chính, chẩn đoán bệnh, ra quyết định tự động trong hệ thống.

Nguồn:https://trituenhantao.io/kien-thuc/decision-tree/

---

### c. Logistic Regression

- **Mục tiêu**:  
  Phân loại nhị phân (binary classification), dự đoán xác suất một sự kiện xảy ra.

- **Cách hoạt động**:  
  Sử dụng hàm sigmoid để chuyển đổi giá trị đầu ra về khoảng (0,1), mô hình hóa xác suất của một lớp.  
  Phương trình Logistic Regression:

  \[
  P(y=1|x) = \frac{1}{1+e^{-(\beta_0 + \beta_1x_1 + \dots + \beta_kx_k)}}
  \]

  Trong đó:
  - \( P(y=1|x) \) là xác suất dự đoán của lớp 1.
  - \( \beta_0, \beta_1, ..., \beta_k \) là các hệ số mô hình.

- **Ưu điểm**:
  - Hiệu quả với bài toán phân loại đơn giản.
  - Đầu ra có thể được hiểu là xác suất.

- **Nhược điểm**:
  - Không xử lý tốt quan hệ phi tuyến phức tạp.
  - Dễ bị ảnh hưởng bởi đa cộng tuyến (multicollinearity).

- **Ứng dụng**:
  - Phân loại email spam, dự đoán khách hàng rời bỏ, chẩn đoán bệnh.

---

> **Ghi chú**:  
> - Đối với bài toán phức tạp, Decision Tree thường được cải tiến thành các mô hình như Random Forest hoặc Gradient Boosting để khắc phục nhược điểm quá khớp.
> - Linear Regression có thể được mở rộng thành các biến thể như Ridge Regression, Lasso Regression để cải thiện hiệu suất.
> - Logistic Regression mở rộng được cho phân loại đa lớp bằng kỹ thuật như One-vs-Rest hoặc Softmax Regression.

---
## 2.Thuật toán Unsupervised: K-Means, DBSCAN, PCA

### a. K-Means Clustering

#### Định nghĩa
K-Means là một thuật toán phân cụm không giám sát phổ biến, dùng để nhóm các điểm dữ liệu chưa gán nhãn thành **K cụm** dựa trên sự tương đồng.  
Mỗi cụm được đại diện bởi một **tâm cụm (centroid)** – trung bình cộng của các điểm trong cụm đó.

#### Cách hoạt động
1. Chọn ngẫu nhiên **K tâm cụm** ban đầu.
2. Gán mỗi điểm dữ liệu vào cụm có tâm gần nhất (dựa trên khoảng cách Euclidean).
3. Cập nhật lại tâm cụm bằng trung bình cộng của các điểm trong cụm.
4. Lặp lại bước 2 và 3 cho đến khi:
   - Các tâm cụm ổn định (không thay đổi đáng kể),
   - Hoặc đạt số lần lặp tối đa.

#### Đặc điểm
- **Ưu điểm**:
  - Đơn giản, dễ hiểu, dễ triển khai.
  - Tính toán nhanh, hiệu quả với dữ liệu lớn.
- **Nhược điểm**:
  - Phải xác định trước số lượng cụm **K**.
  - Nhạy cảm với giá trị khởi tạo tâm cụm ban đầu và điểm ngoại lai.
  - Khó phân cụm khi các cụm không có dạng hình cầu hoặc khác kích thước.

#### Ứng dụng
- Phân khúc khách hàng.
- Phân nhóm tài liệu.
- Phân đoạn hình ảnh trong xử lý ảnh.

---

### b. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

#### Định nghĩa
DBSCAN là thuật toán phân cụm dựa trên mật độ điểm, có thể phát hiện các cụm có hình dạng bất kỳ và tự động nhận diện **điểm nhiễu (outliers)**.

#### Cách hoạt động
- Dựa trên hai tham số chính:
  - \( \varepsilon \) (**epsilon**): bán kính lân cận.
  - **MinPts**: số điểm tối thiểu để được coi là một cụm dày đặc.

- Phân loại các điểm:
  - **Core Point**: Có ít nhất **MinPts** điểm lân cận trong bán kính \( \varepsilon \).
  - **Border Point**: Nằm gần điểm Core nhưng không đủ MinPts để thành Core.
  - **Noise Point**: Không thuộc cụm nào.

- Quá trình:
  - Bắt đầu từ một Core Point, mở rộng cụm bằng cách kết nối các điểm lân cận.
  - Các Border Points kết nối với Core Points.
  - Bỏ qua các Noise Points.

#### Đặc điểm
- **Ưu điểm**:
  - Không cần xác định trước số cụm.
  - Xử lý tốt dữ liệu nhiễu và các cụm có hình dạng khác nhau.
- **Nhược điểm**:
  - Nhạy cảm với lựa chọn tham số \( \varepsilon \) và MinPts.
  - Khó xử lý dữ liệu có mật độ thay đổi.

#### Ứng dụng
- Phát hiện bất thường trong giao dịch, mạng máy tính.
- Phân tích dữ liệu không gian địa lý (GIS).
- Phân nhóm dữ liệu trong nghiên cứu sinh học, thiên văn học.

---

### c. PCA (Principal Component Analysis)

#### Định nghĩa
PCA là kỹ thuật **giảm chiều dữ liệu tuyến tính** nhằm tối ưu hóa việc biểu diễn dữ liệu bằng cách giữ lại phương sai lớn nhất, đồng thời giảm số chiều.

#### Cách hoạt động
1. **Chuẩn hóa** dữ liệu để đưa các biến về cùng thang đo.
2. **Tính ma trận hiệp phương sai** để tìm sự liên hệ giữa các biến.
3. **Tính toán các eigenvectors và eigenvalues** của ma trận hiệp phương sai.
4. **Chọn principal components** tương ứng với eigenvalues lớn nhất.
5. **Chiếu dữ liệu** lên không gian mới bởi các thành phần chính này.

#### Đặc điểm
- **Ưu điểm**:
  - Giảm số lượng đặc trưng, loại bỏ nhiễu.
  - Tăng tốc quá trình huấn luyện mô hình học máy.
  - Hỗ trợ trực quan hóa dữ liệu phức tạp trong không gian 2D/3D.
- **Nhược điểm**:
  - Một phần thông tin có thể bị mất.
  - Các thành phần mới khó diễn giải ý nghĩa thực tế.

#### Ứng dụng
- Tiền xử lý dữ liệu cho mô hình học máy.
- Nén dữ liệu hình ảnh, video.
- Phát hiện bất thường trong dữ liệu cao chiều.

---

> **Tóm lại**:  
> - **K-Means**: Nhanh và hiệu quả, nhưng cần chọn số cụm trước và nhạy cảm với điểm khởi tạo.  
> - **DBSCAN**: Tự động xác định số cụm, phát hiện nhiễu tốt, phù hợp với dữ liệu phức tạp.  
> - **PCA**: Không phải thuật toán phân cụm, nhưng hỗ trợ **giảm chiều** trước khi phân cụm hoặc trực quan hóa dữ liệu.



