# Giới thiệu về AI
## AI là gì?
Trí tuệ nhân tạo (AI) là công nghệ cho phép máy móc, đặc biệt là máy tính, "học hỏi" và "suy nghĩ" như con người. Trí tuệ nhân tạo khác với việc lập trình logic trong các ngôn ngữ lập trình là ở việc ứng dụng các hệ thống học máy (machine learning) để mô phỏng trí tuệ của con người trong các xử lý mà con người làm tốt hơn máy tính.

Cụ thể, trí tuệ nhân tạo giúp máy tính có được những trí tuệ của con người như: biết suy nghĩ và lập luận để giải quyết vấn đề, biết giao tiếp do hiểu ngôn ngữ, tiếng nói, biết học và tự thích nghi,…

## Các lĩnh vực của AI
- Machine Learning (ML)
- Deep Learning (DL)
- Computer Vision (CV)
- Natural Language Processing (NLP)
- Robotics

## Ứng dụng AI trong thực tế
- Trợ lý ảo (Siri, Google Assistant)
- Xe tự lái
- Dự đoán tài chính
- Hệ thống gợi ý (Netflix, Spotify)
- Chẩn đoán y tế



---

# Dữ liệu trong AI
## Định nghĩa dữ liệu

Dữ liệu là tập hợp các thông tin, số liệu hoặc quan sát được thu thập và lưu trữ dưới nhiều dạng khác nhau. Trong AI, dữ liệu đóng vai trò cốt lõi, là nền tảng để huấn luyện mô hình, giúp AI nhận diện mẫu, dự đoán và đưa ra quyết định.

## Phân loại dữ liệu trong AI

### Big Data (Dữ liệu lớn)
Big Data là tập dữ liệu có dung lượng khổng lồ, đa dạng và được tạo ra với tốc độ cao. Nó thường được mô tả theo mô hình "3V":
Volume (Dung lượng lớn): Lượng dữ liệu khổng lồ, có thể lên đến petabyte hoặc exabyte.
Variety (Đa dạng): Bao gồm dữ liệu có cấu trúc (bảng, cơ sở dữ liệu), phi cấu trúc (văn bản, hình ảnh, video) và bán cấu trúc (JSON, XML, log files).
Velocity (Tốc độ cao): Dữ liệu được tạo ra liên tục từ nhiều nguồn như mạng xã hội, cảm biến IoT, giao dịch tài chính...

Ứng dụng trong AI:
Phân tích hành vi người dùng trên mạng xã hội
Dự đoán xu hướng tiêu dùng từ dữ liệu mua sắm
Nhận diện hình ảnh và video trong các hệ thống giám sát

### Fast Data (Dữ liệu tốc độ cao)
Fast Data là dữ liệu được xử lý theo thời gian thực hoặc gần thời gian thực, giúp AI phản ứng nhanh với các sự kiện xảy ra.

Đặc điểm:
Tốc độ xử lý nhanh: Yêu cầu AI phân tích và đưa ra quyết định ngay lập tức.
Dung lượng vừa phải: Không quá lớn như Big Data nhưng phải được xử lý liên tục.
Ứng dụng trong các hệ thống thời gian thực.

Ứng dụng trong AI:
Hệ thống phát hiện gian lận trong ngân hàng
AI dự báo kẹt xe trên Google Maps
Cảm biến IoT giám sát máy móc trong sản xuất

---

## Ba loại bài toán trong Machine Learning
| Loại ML                | Dữ liệu đầu vào | Mục tiêu chính     | Ứng dụng                | Thuật toán phổ biến                |
|------------------------|---------------|------------------|----------------------|----------------------------------|
| **Supervised Learning**  | Có nhãn        | Dự đoán output   | Phân loại, hồi quy   | Linear Regression, Decision Trees, Neural Networks |
| **Unsupervised Learning** | Không có nhãn  | Tìm mẫu, cụm     | Phân nhóm, giảm chiều | K-Means, PCA, Hierarchical Clustering |
| **Reinforcement Learning** | Dữ liệu động  | Học qua thưởng/phạt | Tối ưu hành vi | Q-Learning, Deep Q Networks (DQN), Policy Gradient |
## Thuật ngữ và ký hiệu trong AI
### 1.Thuật ngữ (Terminology) trong AI

| Thuật ngữ | Định nghĩa |
|-----------|-----------|
| **Artificial Intelligence (AI)** | Trí tuệ nhân tạo, mô phỏng khả năng suy nghĩ và học hỏi của con người bằng máy tính. |
| **Machine Learning (ML)** | Học máy, một nhánh của AI giúp máy tính học từ dữ liệu mà không cần lập trình cụ thể. |
| **Deep Learning (DL)** | Học sâu, một nhánh của ML sử dụng mạng nơ-ron nhiều lớp để học biểu diễn dữ liệu phức tạp. |
| **Neural Network (NN)** | Mạng nơ-ron nhân tạo, mô hình học dựa trên cách hoạt động của nơ-ron sinh học. |
| **Gradient Descent** | Thuật toán tối ưu hóa dùng để tìm cực tiểu của một hàm số. |
| **Overfitting** | Hiện tượng mô hình học quá kỹ dữ liệu huấn luyện, dẫn đến kém tổng quát hóa cho dữ liệu mới. |
| **Underfitting** | Hiện tượng mô hình quá đơn giản, không học được đặc trưng quan trọng từ dữ liệu. |
| **Feature Engineering** | Quá trình chọn lọc, biến đổi và tạo mới các đặc trưng của dữ liệu để cải thiện hiệu suất mô hình. |
| **Backpropagation** | Phương pháp điều chỉnh trọng số trong mạng nơ-ron để giảm sai số. |
| **Activation Function** | Hàm kích hoạt trong mạng nơ-ron giúp mô hình học được các quan hệ phi tuyến. |

### 2. Ký hiệu (Notions) trong AI

| Ký hiệu | Ý nghĩa |
|---------|--------|
| **\( X \)** | Dữ liệu đầu vào (Input Data) |
| **\( Y \)** | Nhãn đầu ra (Target Output) |
| **\( \hat{Y} \)** | Dự đoán đầu ra của mô hình |
| **\( w \)** | Trọng số (Weights) trong mô hình ML |
| **\( b \)** | Hệ số chặn (Bias) trong mô hình ML |
| **\( f(x) \)** | Hàm dự đoán của mô hình |
| **\( L \)** | Hàm mất mát (Loss Function) để đánh giá độ chính xác của mô hình |
| **\( \alpha \)** | Tốc độ học (Learning Rate) trong Gradient Descent |
| **\( \sigma(x) \)** | Hàm kích hoạt Sigmoid |
| **\( ReLU(x) \)** | Hàm kích hoạt ReLU (Rectified Linear Unit) |
| **\( \sum \)** | Tổng (Summation) của các giá trị |
| **\( \nabla L \)** | Đạo hàm của hàm mất mát theo các tham số mô hình (Gradient of Loss Function) |

# Thư viện phổ biến trong AI

## NumPy
- **Khái niệm**: Thư viện xử lý mảng số học
- **Ra đời**: 2005 bởi Travis Oliphant
- **Ưu điểm**: Tốc độ nhanh, hỗ trợ nhiều phép toán
- **Nhược điểm**: Không hỗ trợ visualization

## Pandas
- **Khái niệm**: Xử lý dữ liệu dạng bảng
- **Ra đời**: 2008 bởi Wes McKinney
- **Ưu điểm**: Hỗ trợ dữ liệu dạng bảng, dễ thao tác
- **Nhược điểm**: Không tối ưu cho dữ liệu lớn

## Matplotlib
- **Khái niệm**: Thư viện vẽ đồ thị
- **Ra đời**: 2003 bởi John Hunter
- **Ưu điểm**: Linh hoạt, mạnh mẽ
- **Nhược điểm**: Cú pháp phức tạp

## PyTorch
- **Khái niệm**: Framework deep learning
- **Ra đời**: 2016 bởi Facebook
- **Ưu điểm**: Dễ dùng, hỗ trợ GPU
- **Nhược điểm**: Chưa tối ưu so với TensorFlow

---

# Notebook và nguồn dữ liệu

## Jupyter Notebooks, Google Colab, Kaggle
---

### Jupyter Notebooks

Jupyter Notebook là một môi trường lập trình tương tác dựa trên web, được sử dụng rộng rãi trong khoa học dữ liệu, AI và Machine Learning. Nó hỗ trợ nhiều ngôn ngữ lập trình, nhưng phổ biến nhất là Python.

Tính năng chính:
Chia mã thành từng ô (cell), có thể chạy từng phần riêng lẻ.
Hỗ trợ Markdown để viết tài liệu kèm với mã nguồn.
Hỗ trợ vẽ biểu đồ trực tiếp bằng Matplotlib, Seaborn…
Tích hợp tốt với các thư viện xử lý dữ liệu như Pandas, NumPy.
Có thể chạy trên máy cá nhân (local) hoặc trên cloud với JupyterHub.
Ứng dụng:
Phân tích dữ liệu, trực quan hóa dữ liệu.
Viết tài liệu hướng dẫn AI/ML.
Chạy thử nghiệm mô hình học máy.



---

### Google Colab

Google Colab (Collaboratory) là một dịch vụ miễn phí của Google cho phép chạy Jupyter Notebook trên nền tảng cloud. Nó giúp người dùng lập trình AI/ML mà không cần cài đặt môi trường trên máy tính.

Tính năng chính:
Miễn phí CPU, GPU, TPU trên cloud.
Hỗ trợ chạy code trực tiếp trên trình duyệt mà không cần cài đặt.
Kết nối với Google Drive để lưu trữ file .ipynb.
Tích hợp sẵn nhiều thư viện như TensorFlow, PyTorch, Pandas…
Hỗ trợ cộng tác dễ dàng bằng cách chia sẻ notebook qua link.

Ứng dụng:
Huấn luyện mô hình AI/ML với GPU miễn phí.
Chạy thử nghiệm nhanh trên dữ liệu lớn.
Học và thực hành các thuật toán AI.



---

### Kaggle Notebooks

Kaggle Notebooks là một công cụ trực tuyến của Kaggle, cho phép chạy mã nguồn Python trên cloud, đặc biệt là trong lĩnh vực khoa học dữ liệu và AI. Nó được tích hợp sẵn với kho dữ liệu Kaggle Datasets và cộng đồng Kaggle.

Tính năng chính:
Miễn phí CPU, GPU, TPU (giới hạn tài nguyên).
Truy cập trực tiếp dữ liệu từ Kaggle Datasets.
Tích hợp sẵn thư viện AI/ML phổ biến.

Ứng dụng:
Phân tích dữ liệu Kaggle mà không cần tải về máy.
Tham gia thi đấu AI trên Kaggle.
Tìm hiểu và tham khảo notebook của người khác.
#### Một Số Dataset Phổ Biến Trên Kaggle:
- **Titanic Dataset**: Dự đoán khả năng sống sót của hành khách trên tàu Titanic.
- **MNIST Dataset**: Bộ dữ liệu gồm các chữ số viết tay, dùng cho bài toán nhận dạng hình ảnh.
- **House Prices - Advanced Regression**: Dự đoán giá nhà dựa trên các đặc trưng như diện tích, số phòng, vị trí,...
- **COVID-19 Dataset**: Thống kê và phân tích dữ liệu về đại dịch COVID-19 trên toàn cầu.
- **IMDB Movie Reviews**: Bộ dữ liệu về đánh giá phim, thường được dùng cho phân tích cảm xúc (Sentiment Analysis).
### Hugging Face Là Gì?
**Hugging Face** là một nền tảng AI chuyên về **NLP (Xử lý ngôn ngữ tự nhiên)** và **Machine Learning**, cung cấp:
- **Transformers Library**: Thư viện mã nguồn mở hỗ trợ các mô hình AI tiên tiến như BERT, GPT, T5, BLOOM.
- **Model Hub**: Kho lưu trữ các mô hình học sâu (Deep Learning) được huấn luyện sẵn.
- **Datasets**: Hàng nghìn bộ dữ liệu miễn phí phục vụ nghiên cứu và huấn luyện mô hình.
- **Spaces**: Dịch vụ triển khai ứng dụng AI trực tiếp trên nền tảng web.

###  Một Số Datasets Phổ Biến Trên Hugging Face:
- **SQuAD**: Bộ dữ liệu câu hỏi - trả lời dành cho bài toán hiểu ngữ cảnh.
- **GLUE Benchmark**: Tập hợp các bài kiểm tra để đánh giá hiệu suất mô hình NLP.
- **Common Voice**: Bộ dữ liệu giọng nói đa ngôn ngữ từ Mozilla, phục vụ huấn luyện mô hình nhận diện giọng nói.
- **Wikitext**: Dữ liệu văn bản từ Wikipedia dùng để huấn luyện mô hình sinh ngôn ngữ.
- **MS MARCO**: Bộ dữ liệu truy vấn tìm kiếm web, thường dùng trong AI tìm kiếm thông tin.

---

## Nguồn dữ liệu
- **Kaggle**: Nền tảng chia sẻ dataset, nổi tiếng với Titanic Dataset, MNIST
- **Hugging Face**: Nền tảng dành cho NLP, có mô hình pretrained như BERT, GPT

---

# Phân tích dữ liệu sơ bộ (EDA)

## Hiểu dữ liệu (Data Understanding)
- Kiểm tra kích thước, loại dữ liệu
- Xác định giá trị thiếu

## Phân tích dữ liệu (Data Analysis)
- Tính toán các thống kê cơ bản
- Tìm hiểu mối quan hệ giữa các biến

## Trực quan hóa dữ liệu (Data Visualization)
- Dùng biểu đồ (histogram, scatter plot, box plot) để hiểu dữ liệu

---
