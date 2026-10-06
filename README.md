# 🤟 Sign Language AI - Nhận Diện Ngôn Ngữ Ký Hiệu Tiếng Việt (VSL)

[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-00C0A3.svg?style=flat&logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10+-5C3EE8.svg?style=flat&logo=opencv&logoColor=white)](https://opencv.org/)

Hệ thống AI nhận diện ngôn ngữ ký hiệu tiếng Việt (Vietnamese Sign Language - VSL) thời gian thực thông qua Webcam bằng mô hình **Deep Learning (Transformer Encoder + CTC Loss)** kết hợp trích xuất đặc trưng khớp xương từ **MediaPipe Holistic**.

---

## 📌 Mục lục (Table of Contents)
- [Giới thiệu dự án (Overview)](#-giới-thiệu-dự-án-overview)
- [Tính năng nổi bật (Features)](#-tính-năng-nổi-bật-features)
- [Kiến trúc hệ thống (Architecture)](#-kiến-trúc-hệ-thống-architecture)
- [Yêu cầu hệ thống (Requirements)](#-yêu-cầu-hệ-thống-requirements)
- [Cài đặt nhanh (Quick Start)](#-cài-đặt-nhanh-quick-start)
- [Hướng dẫn sử dụng (Usage)](#-hướng-dẫn-sử-dụng-usage)
  - [1. Chạy nhận diện trực tiếp qua Webcam](#1-chạy-nhận-diện-trực-tiếp-qua-webcam)
  - [2. Chuẩn bị dữ liệu](#2-chuẩn-bị-dữ-liệu)
  - [3. Huấn luyện mô hình (Training)](#3-huấn-luyện-mô-hình-training)
  - [4. Đánh giá mô hình (Evaluation)](#4-đánh-giá-mô-hình-evaluation)
- [Cấu trúc thư mục (Project Structure)](#-cấu-trúc-thư-mục-project-structure)
- [Hiệu năng mô hình (Performance)](#-hiệu-năng-mô-hình-performance)
- [Đóng góp & Bản quyền (License)](#-đóng-góp--bản-quyền-license)

---

## 🌟 Giới thiệu dự án (Overview)

Dự án phát triển nguyên mẫu hỗ trợ giao tiếp cho người khiếm thính tại Việt Nam:
- Thu nhận luồng hình ảnh trực tiếp từ **Webcam**.
- Trích xuất tọa độ các khớp bàn tay, thân người bằng **MediaPipe Holistic**.
- Chuẩn hóa tọa độ và đưa qua mạng nơ-ron **Transformer Encoder** kết hợp hàm mất mát **CTC (Connectionist Temporal Classification)** để nhận diện từ/cụm từ VSL.
- Hiển thị phụ đề tiếng Việt có dấu Unicode trực tiếp trên màn hình OpenCV và **phát âm thanh tiếng Việt** (Text-to-Speech) qua `pyttsx3`.

Mô hình hiện tại được huấn luyện trên bộ dữ liệu **VSL 100 lớp từ đơn (100-class isolated words)**.

---

## 🚀 Tính năng nổi bật (Features)

- **Trích xuất đặc trưng chính xác:** Sử dụng MediaPipe Holistic trích xuất 21 điểm mỗi bàn tay (trái/phải) và 33 điểm tư thế cơ thể (pose).
- **Chuẩn hóa nâng cao:** Tọa độ khớp được chuẩn hóa theo từng nhóm (per-group landmark normalization) kèm kênh kiểm tra tính hợp lệ (validity channel).
- **Mô hình Transformer hiện đại:** Xử lý chuỗi thời gian (temporal sequence) với cơ chế Self-Attention mạnh mẽ, giải mã CTC không cần gán nhãn từng khung hình.
- **Hỗ trợ thời gian thực (Real-time):** Cơ chế cửa sổ trượt (overlapping sliding window), làm mịn nhãn theo thời gian, lọc độ tin cậy và tự động phát hiện khoảng lặng (silence timing).
- **Phát âm Text-to-Speech (TTS):** Tự động đọc to từ/cụm từ vừa nhận diện bằng giọng đọc tiếng Việt trên Windows.
- **Tự động tối ưu phần cứng:** Tự động nhận diện GPU NVIDIA CUDA (ví dụ RTX 5050 / RTX 30/40 series) hoặc dự phòng chuyển về CPU mượt mà.

---

## 🏗 Kiến trúc hệ thống (Architecture)

```mermaid
flowchart LR
  A[Webcam Video] --> B[OpenCV Frame Capture]
  B --> C[MediaPipe Holistic]
  C --> D[Normalize Landmarks & Masks]
  D --> E[Temporal Sequence Buffer]
  E --> F[Transformer Encoder]
  F --> G[CTC Decoder]
  G --> H[Sign Token Sequence]
  H --> I[Vietnamese Text Formatting]
  I --> J[OpenCV Desktop Display & Local TTS]
```

---

## 💻 Yêu cầu hệ thống (Requirements)

- **Hệ điều hành:** Windows 10/11 (64-bit).
- **Python:** Python 3.10 (khuyến nghị `3.10.x` để tương thích tốt nhất với PyTorch & MediaPipe).
- **Phần cứng:**
  - Webcam kết nối với máy tính.
  - Card đồ họa rời NVIDIA hỗ trợ CUDA (khuyến nghị, tùy chọn CPU vẫn chạy được).

---

## ⚡ Cài đặt nhanh (Quick Start)

### 1. Clone repository về máy tính
```powershell
git clone https://github.com/TaiNguyen567/SIGN-LANGUAGE-DEEPLEARNING-PROJECT.git
cd "SIGN-LANGUAGE-DEEPLEARNING-PROJECT"
```

### 2. Thiết lập môi trường tự động
Chạy file batch đi kèm để tự động tạo môi trường ảo `.venv` và cài đặt các thư viện cần thiết:
```powershell
.\setup_windows.bat
```
Hoặc kích hoạt môi trường và cài đặt thủ công:
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

---

## 📖 Hướng dẫn sử dụng (Usage)

### 1. Chạy nhận diện trực tiếp qua Webcam

Nhấp đúp chuột vào file `run_camera.bat` hoặc chạy lệnh:
```powershell
.\.venv\Scripts\python.exe app.py --camera 0 --device auto
```

**Phím tắt điều khiển trong giao diện camera:**
- `Q` hoặc `Esc`: Thoát ứng dụng.
- `R`: Xóa chuỗi từ đang nhận diện hiện tại (Reset).
- `S`: Đọc lại từ/câu vừa nhận diện (Repeat TTS).

---

### 2. Chuẩn bị dữ liệu

Bộ dữ liệu chuẩn bị theo cấu trúc VSL 100 lớp. Để phân chia tập Train / Val / Test và trích xuất đặc trưng MediaPipe:
```powershell
.\.venv\Scripts\python.exe scripts\prepare_dataset.py --metadata dataset\metadata\vsl100.csv --config config.yaml --stratify-by token_text
```

---

### 3. Huấn luyện mô hình (Training)

Chạy quá trình huấn luyện mạng Transformer với hàm mất mát CTC:
```powershell
.\.venv\Scripts\python.exe train.py --config config.yaml --device auto
```
- Các tham số cấu hình (epochs, learning rate, batch size, v.v.) được tùy chỉnh trong file `config.yaml`.
- Mô hình tốt nhất sẽ tự động được lưu tại `checkpoints/best_model.pt`.
- Lịch sử huấn luyện được ghi lại trong `logs/training.csv`.

---

### 4. Đánh giá mô hình (Evaluation)

Đánh giá độ chính xác của checkpoint trên tập kiểm thử (Test split):
```powershell
.\.venv\Scripts\python.exe evaluate.py --checkpoint checkpoints\best_model.pt --device auto
```
Kết quả báo cáo và biểu đồ sẽ được xuất ra thư mục `reports/`.

---

## 📂 Cấu trúc thư mục (Project Structure)

```text
├── app.py                     # Entry point khởi chạy ứng dụng nhận diện qua Webcam
├── inference.py               # Xử lý suy luận thời gian thực và giao diện OpenCV
├── train.py                   # Script huấn luyện mô hình Transformer + CTC
├── evaluate.py                # Đánh giá checkpoint mô hình trên tập test
├── config.yaml                # File cấu hình toàn bộ hệ thống
├── requirements.txt           # Danh sách các thư viện phụ thuộc
├── run_camera.bat             # Phím tắt chạy nhanh camera trên Windows
├── setup_windows.bat          # Script tự động khởi tạo môi trường Python 3.10
├── scripts/
│   ├── import_vsl100.py       # Import & tiền xử lý video VSL100 gốc
│   └── prepare_dataset.py     # Trích xuất landmarks và phân chia tập dữ liệu
├── src/
│   ├── data/                  # Xử lý dataset, dataloader và tokenize
│   ├── features/              # MediaPipe Holistic landmark extraction & normalization
│   ├── models/                # Kiến trúc Transformer Encoder & CTC head
│   ├── training/              # Vòng lặp huấn luyện, early stopping, metrics
│   ├── inference/             # Buffer trượt và bộ giải mã CTC
│   ├── translation/           # Xử lý hậu kỳ văn bản tiếng Việt
│   ├── tts/                   # Tích hợp giọng nói tiếng Việt qua pyttsx3
│   └── utils/                 # Các hàm tiện ích hỗ trợ
└── tests/                     # Unit test kiểm thử hệ thống
```

---

## 📊 Hiệu năng mô hình (Performance)

Trên tập kiểm thử gồm **409 mẫu VSL100** được tổ chức tách biệt (held-out split):
- **Độ chính xác từ (Exact Word Accuracy):** `87.53%`
- **Tỉ lệ lỗi từ (WER - Word Error Rate):** `14.95%`
- **Tỉ lệ lỗi ký tự (CER - Character Error Rate):** `13.34%`
- **Tốc độ xử lý:** Đạt trên 30 FPS với tăng tốc phần cứng CUDA trên GPU NVIDIA RTX.

---

## ⚠️ Lưu ý & Giới hạn (Limitations)

- Mô hình hiện tại tập trung vào nhận diện **100 từ vựng VSL đơn lẻ** (isolated sign words).
- Nhận diện tối ưu khi người thực hiện ký hiệu ngồi/đứng chính diện webcam trong điều kiện ánh sáng tốt và rõ cả hai tay.
- Giọng đọc phụ thuộc vào engine TTS tiếng Việt được cài đặt sẵn trên hệ điều hành Windows.

---

## 📜 Giấy phép & Bản quyền (License)

Dự án được xây dựng cho mục đích học tập, nghiên cứu và hỗ trợ cộng đồng.
- Mã nguồn dự án: Tham khảo hoặc sử dụng cho mục đích phi thương mại.
- Bản quyền hình ảnh/video dữ liệu ký hiệu thuộc về các đơn vị tác giả bộ dữ liệu gốc.
