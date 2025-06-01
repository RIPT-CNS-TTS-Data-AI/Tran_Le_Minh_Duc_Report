# Transformer và Mô hình Sinh (Generative Models)

## Giới thiệu về Transformer

### Transformer là gì?

Transformer là một kiến trúc mạng nữ thần kinh sâu (deep neural network) cách mạng được Google giới thiệu vào năm 2017 trong bài báo nổi tiếng "Attention Is All You Need" của Vaswani và cộng sự. Đây được coi là một bước đột phá quan trọng trong lĩnh vực trí tuệ nhân tạo, đặc biệt là xử lý ngôn ngữ tự nhiên (NLP).

Khác biệt lớn nhất của Transformer so với các mô hình truyền thống như RNN (Recurrent Neural Networks) và LSTM (Long Short-Term Memory) là việc loại bỏ hoàn toàn tính chất tuần tự (sequential) trong xử lý dữ liệu. Thay vào đó, Transformer dựa hoàn toàn vào cơ chế **Attention**, đặc biệt là **Multi-head Self-Attention**, cho phép mô hình xử lý toàn bộ chuỗi dữ liệu một cách song song.

### Bối cảnh ra đời

Trước khi Transformer xuất hiện, các mô hình NLP chủ yếu dựa vào RNN hoặc LSTM để xử lý dữ liệu tuần tự. Tuy nhiên, những mô hình này gặp phải nhiều hạn chế:

- **Bottleneck tính toán**: Phải xử lý tuần tự từng phần tử, không thể song song hóa
- **Vanishing gradient**: Khó học được các quan hệ xa trong chuỗi dài
- **Tốc độ huấn luyện chậm**: Do không thể tối ưu hóa song song

Transformer đã giải quyết được những vấn đề này và mở ra kỷ nguyên mới cho AI.

## Kiến trúc Transformer Chi tiết

### Tổng quan Kiến trúc

Kiến trúc Transformer gốc gồm hai thành phần chính:
- **Encoder**: Mã hóa thông tin đầu vào
- **Decoder**: Giải mã và sinh ra đầu ra

Cả hai đều được xây dựng từ các khối (blocks) giống nhau, với mỗi khối chứa các lớp xử lý khác nhau.

### Encoder - Bộ Mã hóa

#### Cấu trúc Encoder

Encoder của Transformer gồm N lớp giống nhau (trong mô hình gốc N=6), mỗi lớp bao gồm:

1. **Multi-Head Self-Attention Layer**
2. **Position-wise Feed-Forward Network (FFN)**
3. **Residual Connections** và **Layer Normalization**

#### Chi tiết từng thành phần:

**Multi-Head Self-Attention:**
- Cho phép mỗi vị trí trong chuỗi "chú ý" đến tất cả các vị trí khác
- Tính toán song song cho toàn bộ chuỗi
- Sử dụng nhiều "head" để nắm bắt các khía cạnh ngữ cảnh khác nhau

**Feed-Forward Network:**
- Hai lớp linear với hàm kích hoạt ReLU ở giữa
- Áp dụng độc lập cho từng vị trí
- Giúp mô hình học các biến đổi phi tuyến phức tạp

**Residual Connection + Layer Norm:**
- Giúp huấn luyện mô hình sâu ổn định hơn
- Tránh vấn đề vanishing gradient

### Decoder - Bộ Giải mã

#### Cấu trúc Decoder

Decoder cũng gồm N lớp, mỗi lớp có:

1. **Masked Multi-Head Self-Attention**
2. **Multi-Head Cross-Attention** (với output của Encoder)
3. **Position-wise Feed-Forward Network**
4. **Residual Connections** và **Layer Normalization**

#### Đặc điểm quan trọng:

**Masked Self-Attention:**
- Đảm bảo vị trí thứ i chỉ có thể "nhìn thấy" các vị trí trước đó (j < i)
- Quan trọng cho việc sinh văn bản autoregressive

**Cross-Attention:**
- Kết nối Decoder với output của Encoder
- Cho phép Decoder "chú ý" đến thông tin đầu vào khi sinh output

### Cơ chế Attention Chi tiết

#### Attention là gì?

Attention là cơ chế cho phép mô hình tập trung vào các phần quan trọng của đầu vào khi xử lý một phần tử cụ thể. Về mặt toán học:

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

Trong đó:
- **Q (Query)**: "Câu hỏi" - thông tin cần tìm
- **K (Key)**: "Chìa khóa" - thông tin để so khớp
- **V (Value)**: "Giá trị" - thông tin thực tế được trả về

#### Self-Attention

Trong Self-Attention, Q, K, V đều được tạo từ cùng một input:
- Mỗi từ trong câu vừa là Query, vừa là Key và Value
- Cho phép mỗi từ "chú ý" đến tất cả các từ khác trong câu
- Học được quan hệ ngữ cảnh giữa các từ

#### Multi-Head Attention

Multi-Head Attention mở rộng khái niệm Attention bằng cách:

1. **Chia thành nhiều "head"**: Thay vì một attention, sử dụng h heads song song
2. **Mỗi head học khía cạnh khác nhau**: Có thể là cú pháp, ngữ nghĩa, pragmatics...
3. **Kết hợp kết quả**: Concatenate và project để tạo output cuối

```
MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W^O
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### Positional Encoding

Do Transformer không có tính chất tuần tự tự nhiên, cần thêm thông tin vị trí:

**Sinusoidal Positional Encoding:**
```math
PE(pos, 2i) = sin(pos/10000^(2i/d_model))

PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
```

Điều này giúp mô hình phân biệt được thứ tự của các từ trong câu.

## Ưu điểm của Transformer

### So sánh với RNN/LSTM

| Khía cạnh | RNN/LSTM | Transformer |
|-----------|----------|-------------|
| **Xử lý song song** | Không thể | Hoàn toàn song song |
| **Quan hệ xa** | Khó học | Excellent với Attention |
| **Tốc độ huấn luyện** | Chậm | Nhanh hơn nhiều |
| **Khả năng mở rộng** | Hạn chế | Rất tốt |
| **Bộ nhớ** | Ít hơn | Nhiều hơn (O(n²)) |

### Những ưu điểm chi tiết

1. **Parallelization (Song song hóa)**
   - Có thể xử lý toàn bộ chuỗi cùng lúc
   - Tận dụng được GPU/TPU hiệu quả
   - Giảm thời gian huấn luyện đáng kể

2. **Long-range Dependencies**
   - Attention trực tiếp kết nối mọi cặp vị trí
   - Không bị giới hạn bởi "trí nhớ" của hidden state
   - Có thể học quan hệ phức tạp trong văn bản dài

3. **Interpretability (Khả năng giải thích)**
   - Attention weights có thể được visualize
   - Hiểu được mô hình "chú ý" vào đâu
   - Dễ debug và phân tích hơn

4. **Scalability (Khả năng mở rộng)**
   - Có thể xây dựng mô hình rất lớn (billions parameters)
   - Hiệu suất tăng theo kích thước model và dữ liệu
   - Nền tảng cho Large Language Models
## Các Biến thể và Phát triển
### Encoder-only Models
**BERT (Bidirectional Encoder Representations from Transformers)**
- Chỉ sử dụng Encoder
- Bidirectional context (nhìn cả hai hướng)
- Excellent cho classification, NER, QA
**RoBERTa, ALBERT, DeBERTa**
- Cải tiến của BERT với các kỹ thuật khác nhau
- Tối ưu hóa hiệu suất và tốc độ
### Decoder-only Models
**GPT (Generative Pretrained Transformer)**
- Chỉ sử dụng Decoder với masked attention
- Autoregressive generation
- Excellent cho text generation
**GPT-2, GPT-3, GPT-4**
- Scale up với billions/trillions parameters
- Emergent abilities ở scale lớn
### Encoder-Decoder Models
**T5 (Text-to-Text Transfer Transformer)**
- Mọi task đều được format thành text-to-text
- Unified framework cho nhiều task
**BART, mT5**
- Variants cho các ứng dụng cụ thể
## Ứng dụng của Transformer
### Xử lý Ngôn ngữ Tự nhiên (NLP)
#### Text Generation
- **Chatbots và AI Assistants**: GPT-3.5, GPT-4, Claude
- **Creative Writing**: Sinh thơ, truyện, kịch bản
- **Code Generation**: GitHub Copilot, CodeT5
#### Machine Translation
- **Google Translate**: Cải thiện đáng kể chất lượng dịch
- **Multilingual Models**: mBERT, XLM-R
- **Zero-shot Translation**: Dịch giữa các ngôn ngữ chưa thấy
#### Text Understanding
- **Question Answering**: BERT cho SQuAD, natural QA
- **Sentiment Analysis**: Phân tích cảm xúc trong reviews, social media
- **Named Entity Recognition**: Nhận diện thực thể trong văn bản
- **Text Summarization**: Tóm tắt tự động các tài liệu dài
#### Information Retrieval
- **Semantic Search**: Tìm kiếm dựa trên ý nghĩa thay vì từ khóa
- **Document Ranking**: Xếp hạng relevance của tài liệu
- **Cross-lingual IR**: Tìm kiếm xuyên ngôn ngữ
### Computer Vision
#### Vision Transformer (ViT)
- Áp dụng Transformer trực tiếp cho image patches
- Vượt qua CNN trong nhiều benchmark
- Đặc biệt hiệu quả với large-scale datasets
#### Object Detection
- **DETR (Detection Transformer)**: End-to-end object detection
- **Deformable DETR**: Cải thiện efficiency
#### Image Generation
- **DALL-E**: Text-to-image generation
- **Imagen, Parti**: High-quality image synthesis
### Multi-modal Applications
#### Vision-Language Models
- **CLIP**: Contrastive learning cho image-text pairs
- **BLIP**: Bootstrapping vision-language understanding
- **Flamingo**: Few-shot learning trên vision-language task
#### Speech Processing
- **Whisper**: Automatic speech recognition
- **SpeechT5**: Speech synthesis và recognition
### Khoa học và Nghiên cứu
#### Protein Folding
- **AlphaFold**: Dự đoán cấu trúc protein
- **ESM**: Evolutionary Scale Modeling cho proteins
#### Drug Discovery
- **MolGPT**: Molecular generation
- **ChemBERTa**: Chemical understanding
#### Climate Modeling
- **WeatherBench**: Weather prediction
- **Climate Transformer**: Long-term climate modeling
### Robotics và Control
#### Robot Control
- **Transformer-based policies**: Học control policies từ demonstration
- **Decision Transformer**: Reinforcement learning as sequence modeling
#### Game Playing
- **MuZero**: Planning trong games phức tạp
- **OpenAI Five**: Dota 2 gameplay
## Thách thức và Hạn chế
### Computational Complexity
**Quadratic Complexity:**
- Attention có độ phức tạp O(n²) với length của sequence
- Trở thành bottleneck với very long sequences
- Giải pháp: Sparse attention, linear attention variants
------
# Generative Models - Mô hình sinh trong Machine Learning

## 1. Generative Models là gì?

Mô hình sinh (Generative Models) là một lớp các mô hình học máy có khả năng tạo ra dữ liệu mới dựa trên việc học các đặc trưng và phân phối xác suất của dữ liệu huấn luyện. Khác với các mô hình phân biệt (discriminative models) chỉ tập trung vào việc phân loại hoặc dự đoán, mô hình sinh có thể tạo ra các mẫu dữ liệu hoàn toàn mới nhưng vẫn giữ được các đặc trưng tương tự như dữ liệu gốc.

### Đặc điểm chính của Generative Models:

- **Học phân phối dữ liệu**: Mô hình sinh học cách mô phỏng phân phối xác suất P(X) của dữ liệu đầu vào
- **Khả năng sinh dữ liệu**: Có thể tạo ra các mẫu dữ liệu mới từ phân phối đã học
- **Hiểu cấu trúc dữ liệu**: Nắm bắt được các mối quan hệ phức tạp và cấu trúc ẩn trong dữ liệu
- **Ứng dụng đa dạng**: Từ sinh ảnh, văn bản, âm thanh đến tạo dữ liệu tổng hợp

## 2. So sánh Discriminative vs. Generative Models

| **Đặc điểm** | **Generative Models** | **Discriminative Models** |
|--------------|----------------------|---------------------------|
| **Mục tiêu học tập** | Học phân phối p(x,y) hoặc p(x) | Học phân phối p(y\|x) |
| **Chức năng chính** | Sinh dữ liệu mới, mô phỏng phân phối | Phân biệt, phân loại các lớp |
| **Ví dụ điển hình** | Naive Bayes, GAN, VAE, GPT | Logistic Regression, SVM, Random Forest |
| **Ưu điểm** | - Tốt cho unsupervised learning<br>- Hiệu quả với dữ liệu nhỏ<br>- Có thể sinh dữ liệu mới<br>- Xử lý được missing data | - Hiệu quả cho supervised learning<br>- Tốc độ training và inference nhanh<br>- Độ chính xác cao cho classification |
| **Nhược điểm** | - Tính toán phức tạp hơn<br>- Thời gian training lâu<br>- Đòi hỏi nhiều tài nguyên | - Không thể sinh dữ liệu mới<br>- Phụ thuộc vào labeled data<br>- Kém linh hoạt với missing data |
| **Ứng dụng** | Data augmentation, Content generation, Anomaly detection | Classification, Regression, Pattern recognition |

## 3. Các loại Generative Models chính

### 3.1 Traditional Generative Models
- **Naive Bayes**: Sử dụng định lý Bayes với giả định độc lập
- **Hidden Markov Models (HMM)**: Mô hình chuỗi với trạng thái ẩn
- **Gaussian Mixture Models (GMM)**: Kết hợp nhiều phân phối Gaussian

### 3.2 Deep Generative Models
- **Generative Adversarial Networks (GANs)**
- **Variational Autoencoders (VAEs)**
- **Autoregressive Models** (GPT, PixelRNN)
- **Flow-based Models** (Normalizing Flows)
- **Diffusion Models** (DDPM, Stable Diffusion)

## 4. Generative Adversarial Networks (GANs)

### 4.1 Giới thiệu GAN

GAN là một kiến trúc mô hình sinh được đề xuất bởi Ian Goodfellow năm 2014. GAN bao gồm hai mạng neural network cạnh tranh với nhau trong một "trò chơi minimax":

- **Generator (G)**: Học cách tạo ra dữ liệu giả từ nhiễu ngẫu nhiên
- **Discriminator (D)**: Học cách phân biệt giữa dữ liệu thật và dữ liệu giả

### 4.2 Cấu trúc và Hoạt động

```
Noise z → Generator G(z) → Fake Data
Real Data ↘
           Discriminator D → Real/Fake Probability
Fake Data ↗
```

#### Quá trình training:

1. **Generator**: Nhận đầu vào là vector nhiễu z từ phân phối đơn giản (thường là Gaussian), biến đổi thành dữ liệu giả G(z)
2. **Discriminator**: Nhận cả dữ liệu thật và giả, đưa ra xác suất đó là dữ liệu thật
3. **Adversarial Training**: Generator cố gắng "lừa" Discriminator, còn Discriminator cố gắng không bị lừa

#### Hàm mất mát:
```math
min_G max_D V(D,G) = E[log D(x)] + E[log(1 - D(G(z)))]
```

### 4.3 Các biến thể GAN phổ biến

- **DCGAN**: Sử dụng Convolutional layers
- **WGAN**: Sử dụng Wasserstein distance
- **StyleGAN**: Tạo ảnh chất lượng cao với style transfer
- **CycleGAN**: Image-to-image translation không cần paired data
- **BigGAN**: Tạo ảnh độ phân giải cao với class conditioning

### 4.4 Ưu nhược điểm của GAN

#### Ưu điểm:
- Tạo ra dữ liệu chất lượng cao, sắc nét
- Không cần giả định về phân phối dữ liệu
- Đa dạng trong các biến thể và ứng dụng
- Hiệu suất tốt với dữ liệu phức tạp

#### Nhược điểm:
- **Mode collapse**: Generator tạo ra ít diversity
- **Training instability**: Khó cân bằng giữa G và D
- **Vanishing gradient**: Discriminator quá mạnh làm Generator không học được
- Khó đánh giá chất lượng mô hình

### 4.5 Ứng dụng của GAN

- **Sinh ảnh**: Tạo ảnh người, động vật, cảnh quan
- **Style transfer**: Chuyển đổi phong cách nghệ thuật
- **Data augmentation**: Tăng cường dữ liệu training
- **Super-resolution**: Nâng cao độ phân giải ảnh
- **Deepfake**: Tạo video giả (có vấn đề đạo đức)
- **Medical imaging**: Tạo dữ liệu y tế tổng hợp
- **Game development**: Tạo texture, character tự động

## 5. Variational Autoencoders (VAEs)

### 5.1 Giới thiệu VAE

VAE là một mô hình sinh kết hợp giữa Autoencoder và Bayesian inference, được đề xuất bởi Kingma và Welling năm 2013. VAE học cách mã hóa dữ liệu vào một không gian tiềm ẩn (latent space) có cấu trúc xác suất rõ ràng.

### 5.2 Cấu trúc VAE

```math
Input x → Encoder → μ(x), σ(x) → Sampling z → Decoder → Reconstructed x'
                        ↓
                   Latent Space z
```

#### Các thành phần:

1. **Encoder (Recognition Network)**: 
   - Mã hóa input x thành các tham số μ và σ của phân phối Gaussian
   - q(z|x) ≈ N(μ(x), σ²(x))

2. **Latent Space**: 
   - Không gian tiềm ẩn z với phân phối prior p(z) = N(0, I)
   - Cho phép interpolation và sampling

3. **Decoder (Generative Network)**:
   - Giải mã z thành dữ liệu tái tạo x'
   - p(x|z) được parameterized bởi decoder

### 5.3 Hàm mất mát VAE

VAE tối ưu hóa **Evidence Lower Bound (ELBO)**:

```math
L(θ,φ;x) = E[log p_θ(x|z)] - KL(q_φ(z|x) || p(z))
```

Bao gồm hai thành phần:
- **Reconstruction Loss**: Đo độ chính xác tái tạo dữ liệu
- **KL Divergence**: Điều chuẩn latent space về phân phối prior

### 5.4 Reparameterization Trick

Để có thể backpropagation qua sampling operation:
```math
z = μ + σ ⊙ ε, where ε ~ N(0, I)
```

### 5.5 Ưu nhược điểm của VAE

#### Ưu điểm:
- **Stable training**: Không có adversarial training
- **Principled latent space**: Có thể interpolate smoothly
- **Probabilistic framework**: Có cơ sở lý thuyết vững chắc
- **Controllable generation**: Có thể điều khiển quá trình sinh

#### Nhược điểm:
- **Blurry outputs**: Ảnh sinh ra thường mờ hơn GAN
- **Posterior collapse**: Latent variables có thể bị ignore
- **Limited expressiveness**: Giả định Gaussian có thể hạn chế

### 5.6 Các biến thể VAE

- **β-VAE**: Điều chỉnh trọng số KL term
- **WAE**: Wasserstein Autoencoder
- **VQ-VAE**: Vector Quantized VAE
- **VAE-GAN**: Kết hợp VAE và GAN

### 5.7 Ứng dụng của VAE

- **Image generation**: Tạo ảnh với khả năng điều khiển
- **Data compression**: Nén dữ liệu với lossy compression
- **Anomaly detection**: Phát hiện outliers dựa vào reconstruction error
- **Drug discovery**: Tạo phân tử mới trong dược phẩm
- **Recommender systems**: Collaborative filtering
- **Semi-supervised learning**: Kết hợp labeled và unlabeled data

## 6. So sánh GAN vs VAE

| **Khía cạnh** | **GAN** | **VAE** |
|---------------|---------|---------|
| **Chất lượng sinh** | Cao, sắc nét | Trung bình, hơi mờ |
| **Độ ổn định training** | Khó, không ổn định | Dễ, ổn định |
| **Latent space** | Không có cấu trúc rõ ràng | Có cấu trúc xác suất |
| **Diversity** | Có thể bị mode collapse | Tốt hơn |
| **Controllability** | Hạn chế | Tốt |
| **Theoretical foundation** | Yếu hơn | Mạnh (Bayesian) |

## 7. Các xu hướng hiện tại

### 7.1 Diffusion Models
- Stable Diffusion, DALL-E 2, Midjourney
- Chất lượng sinh cao, training ổn định
- Ứng dụng trong text-to-image generation

### 7.2 Large Language Models
- GPT-3/4, ChatGPT, Claude
- Autoregressive text generation
- Multimodal capabilities

### 7.3 Neural Radiance Fields (NeRF)
- 3D scene generation
- Novel view synthesis

## 8. Thách thức và hướng phát triển

### 8.1 Thách thức hiện tại:
- **Evaluation metrics**: Khó đánh giá chất lượng sinh
- **Mode collapse**: Thiếu diversity trong output
- **Training instability**: Đặc biệt với GAN
- **Computational cost**: Đòi hỏi tài nguyên lớn
- **Ethical concerns**: Deepfake, misinformation

### 8.2 Hướng phát triển:
- **Multimodal generation**: Kết hợp text, image, audio
- **Few-shot generation**: Học từ ít dữ liệu
- **Controllable generation**: Điều khiển chi tiết hơn
- **Efficient architectures**: Giảm computational cost
- **Federated learning**: Training distributed

# Transfer Learning và Fine-tuning: Hướng dẫn toàn diện

## 1. Transfer Learning là gì?

Transfer Learning (học chuyển giao) là một kỹ thuật trong machine learning và deep learning, nơi một mô hình đã được huấn luyện trên một tác vụ lớn (source task) được tái sử dụng và điều chỉnh để giải quyết một tác vụ mới nhưng có liên quan (target task). Thay vì huấn luyện mô hình từ đầu, chúng ta tận dụng tri thức và đặc trưng đã học được từ tác vụ gốc.

### Nguyên lý cơ bản

Ý tưởng cốt lõi của Transfer Learning dựa trên quan sát rằng các mô hình deep learning, đặc biệt là trong các layer đầu, thường học được những đặc trưng tổng quát có thể áp dụng cho nhiều tác vụ khác nhau. Ví dụ:

- **Trong Computer Vision**: Các layer đầu học cách nhận diện các đường viền, góc cạnh, và texture cơ bản
- **Trong NLP**: Các layer đầu học cách hiểu cú pháp, ngữ nghĩa cơ bản của ngôn ngữ
- **Trong Audio Processing**: Các layer đầu học cách nhận diện các pattern âm thanh cơ bản

## 2. Tại sao sử dụng Transfer Learning?

### 2.1 Tiết kiệm tài nguyên

Transfer Learning mang lại những lợi ích to lớn về mặt tài nguyên:

- **Thời gian huấn luyện**: Giảm từ nhiều tuần xuống còn vài giờ hoặc vài ngày
- **Chi phí tính toán**: Không cần GPU/TPU mạnh mẽ trong thời gian dài
- **Năng lượng**: Giảm đáng kể lượng điện năng tiêu thụ
- **Nhân lực**: Ít cần chuyên gia để thiết kế và debug mô hình phức tạp

### 2.2 Giải quyết vấn đề dữ liệu hạn chế

Trong thực tế, nhiều bài toán gặp phải tình trạng thiếu dữ liệu:

- **Dữ liệu y tế**: Khó thu thập do vấn đề bảo mật và đạo đức
- **Dữ liệu chuyên ngành**: Cần chuyên gia để gán nhãn, chi phí cao
- **Dữ liệu hiếm**: Các trường hợp edge case hoặc sự kiện hiếm

Transfer Learning cho phép tận dụng tri thức từ tập dữ liệu lớn để cải thiện hiệu suất trên tập dữ liệu nhỏ.

### 2.3 Cải thiện hiệu suất

Các mô hình pre-trained thường được huấn luyện trên tập dữ liệu khổng lồ với tài nguyên tính toán lớn, tạo ra những representation chất lượng cao mà các mô hình nhỏ khó có thể đạt được.

## 3. Các loại Transfer Learning

### 3.1 Domain Adaptation

Khi source domain và target domain khác nhau nhưng task giống nhau:
- **Ví dụ**: Mô hình phân loại ảnh được huấn luyện trên ảnh tự nhiên, áp dụng cho ảnh y tế

### 3.2 Task Transfer

Khi domain giống nhau nhưng task khác nhau:
- **Ví dụ**: Từ image classification chuyển sang object detection

### 3.3 Cross-domain Transfer

Khi cả domain và task đều khác nhau:
- **Ví dụ**: Từ xử lý văn bản chuyển sang xử lý âm thanh (qua embedding)

## 4. Các mô hình phổ biến trong Transfer Learning

### 4.1 Computer Vision

#### ResNet (Residual Networks)
- **Đặc điểm**: Giải quyết vấn đề vanishing gradient với residual connections
- **Variants**: ResNet-50, ResNet-101, ResNet-152
- **Ứng dụng**: Image classification, object detection, semantic segmentation

#### VGG (Visual Geometry Group)
- **Đặc điểm**: Kiến trúc đơn giản với các convolutional layer 3x3
- **Variants**: VGG-16, VGG-19
- **Ưu điểm**: Dễ hiểu, stable performance

#### EfficientNet
- **Đặc điểm**: Cân bằng giữa depth, width, và resolution
- **Ưu điểm**: Hiệu suất cao với ít tham số hơn
- **Variants**: EfficientNet-B0 đến B7

#### Vision Transformer (ViT)
- **Đặc điểm**: Áp dụng attention mechanism cho computer vision
- **Ưu điểm**: Hiệu suất vượt trội trên large datasets
- **Nhược điểm**: Cần nhiều dữ liệu để huấn luyện hiệu quả

### 4.2 Natural Language Processing

#### BERT (Bidirectional Encoder Representations from Transformers)
- **Đặc điểm**: Học bidirectional context từ unlabeled text
- **Variants**: BERT-base, BERT-large, RoBERTa, DistilBERT
- **Ứng dụng**: Text classification, NER, question answering

#### GPT (Generative Pre-trained Transformer)
- **Đặc điểm**: Autoregressive language model
- **Variants**: GPT-2, GPT-3, GPT-4
- **Ứng dụng**: Text generation, completion, few-shot learning

#### T5 (Text-to-Text Transfer Transformer)
- **Đặc điểm**: Chuyển đổi mọi NLP task thành text-to-text format
- **Ưu điểm**: Unified approach cho multiple tasks
- **Ứng dụng**: Translation, summarization, question answering

### 4.3 Audio Processing

#### Wav2Vec
- **Đặc điểm**: Self-supervised learning trên raw audio
- **Ứng dụng**: Speech recognition, audio classification

#### OpenAI Whisper
- **Đặc điểm**: Robust speech recognition model
- **Ưu điểm**: Multilingual support, noise robust

## 5. Fine-tuning: Khái niệm và Phương pháp

### 5.1 Fine-tuning là gì?

Fine-tuning là quá trình điều chỉnh một mô hình đã được pre-trained để thích nghi với tác vụ cụ thể. Quá trình này bao gồm:

1. **Khởi tạo**: Bắt đầu với pre-trained weights
2. **Điều chỉnh kiến trúc**: Thay đổi output layer cho task mới
3. **Huấn luyện**: Cập nhật weights với learning rate thấp
4. **Đánh giá**: Kiểm tra performance trên validation set

### 5.2 Các chiến lược Fine-tuning

#### 5.2.1 Feature Extraction

**Cách thức hoạt động:**
- Đóng băng (freeze) tất cả pre-trained layers
- Chỉ huấn luyện classifier layer mới
- Sử dụng pre-trained model như feature extractor

**Ưu điểm:**
- Huấn luyện nhanh chóng
- Ít risk overfitting
- Cần ít dữ liệu
- Ít tài nguyên tính toán

**Nhược điểm:**
- Khả năng thích nghi hạn chế
- Có thể không tối ưu cho task mới

**Khi nào sử dụng:**
- Dữ liệu target task ít (< 1000 samples)
- Task mới tương tự với pre-trained task
- Tài nguyên tính toán hạn chế

#### 5.2.2 Fine-tuning toàn bộ mô hình

**Cách thức hoạt động:**
- Mở khóa (unfreeze) tất cả layers
- Huấn luyện toàn bộ mô hình với learning rate thấp
- Điều chỉnh tất cả weights

**Ưu điểm:**
- Hiệu suất cao nhất
- Thích nghi tốt với task mới
- Linh hoạt trong việc học pattern mới

**Nhược điểm:**
- Cần nhiều dữ liệu
- Risk overfitting cao
- Tốn tài nguyên tính toán
- Có thể "quên" tri thức gốc (catastrophic forgetting)

**Khi nào sử dụng:**
- Dữ liệu target task nhiều (> 10,000 samples)
- Task mới khác biệt với pre-trained task
- Có đủ tài nguyên tính toán

#### 5.2.3 Gradual Fine-tuning

**Cách thức hoạt động:**
1. Bắt đầu với feature extraction
2. Dần dần unfreeze các layer từ trên xuống
3. Fine-tune từng phần một cách tuần tự

**Ưu điểm:**
- Cân bằng giữa stability và adaptability
- Giảm risk catastrophic forgetting
- Có thể điều chỉnh linh hoạt theo hiệu suất

### 5.3 Layer freezing strategies

#### Bottom-up approach
- Freeze các layer thấp (feature extraction layers)
- Fine-tune các layer cao (task-specific layers)
- Phù hợp khi task mới tương tự task gốc

#### Top-down approach
- Freeze các layer cao
- Fine-tune các layer thấp
- Ít phổ biến, dùng trong trường hợp đặc biệt

#### Selective freezing
- Freeze một số layer cụ thể dựa trên analysis
- Cần hiểu biết sâu về kiến trúc mô hình

## 6. Chiến lược Fine-tuning hiệu quả

### 6.1 Learning Rate Scheduling

#### Discriminative Learning Rates
- Sử dụng learning rate khác nhau cho các layer khác nhau
- Layer thấp: learning rate rất thấp (1e-5)
- Layer cao: learning rate cao hơn (1e-3)

#### Gradual Unfreezing
- Bắt đầu với learning rate thấp
- Tăng dần learning rate khi unfreeze nhiều layer hơn

#### Cosine Annealing
- Giảm learning rate theo hàm cosine
- Giúp model converge smoothly

### 6.2 Data Augmentation

#### Standard Augmentation
- Rotation, flipping, scaling cho images
- Synonym replacement, back-translation cho text

#### Advanced Augmentation
- Mixup, CutMix cho computer vision
- Word dropout, sentence shuffling cho NLP

### 6.3 Regularization Techniques

#### Dropout
- Thêm dropout layers để prevent overfitting
- Điều chỉnh dropout rate dựa trên dataset size

#### Weight Decay
- L2 regularization để giữ weights nhỏ
- Prevent overfitting, đặc biệt quan trọng với small datasets

#### Early Stopping
- Monitor validation loss
- Stop training khi validation loss không cải thiện

### 6.4 Advanced Techniques

#### Progressive Resizing
- Bắt đầu với image size nhỏ
- Dần tăng size trong quá trình training
- Giúp model học coarse-to-fine features

#### Test Time Augmentation (TTA)
- Apply augmentation lúc inference
- Average predictions từ multiple augmented versions

#### Model Ensembling
- Kết hợp predictions từ multiple fine-tuned models
- Cải thiện robustness và accuracy

## 7. Evaluation và Monitoring

### 7.1 Metrics Selection

#### Classification Tasks
- Accuracy, Precision, Recall, F1-score
- ROC-AUC cho binary classification
- Macro/Micro averaged metrics cho multi-class

#### Regression Tasks
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- R² (Coefficient of Determination)

#### Domain-specific Metrics
- BLEU score cho machine translation
- ROUGE score cho text summarization
- mIoU cho semantic segmentation

### 7.2 Validation Strategies

#### Hold-out Validation
- Simple train/validation/test split
- Phù hợp với large datasets

#### K-fold Cross Validation
- Chia data thành k folds
- Train k times, mỗi lần dùng 1 fold làm validation
- Robust evaluation cho small datasets

#### Stratified Sampling
- Đảm bảo distribution của classes giống nhau across splits
- Quan trọng với imbalanced datasets

### 7.3 Monitoring Training Process

#### Learning Curves
- Plot training/validation loss theo epoch
- Identify overfitting, underfitting

#### Gradient Monitoring
- Track gradient norms
- Detect vanishing/exploding gradients

#### Weight Analysis
- Visualize weight distributions
- Monitor weight changes during training
