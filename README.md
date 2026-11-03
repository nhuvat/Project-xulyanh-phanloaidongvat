# 🐾 Phân loại động vật từ ảnh — Animals-10

Đồ án môn **Xử lý ảnh**: phân loại ảnh động vật thành 10 lớp sử dụng deep learning và transfer learning. Repository gồm pipeline huấn luyện, đánh giá, giải thích mô hình Grad-CAM và giao diện web Gradio.

## Công nghệ & phương pháp

- **PyTorch / Torchvision**: xây dựng và huấn luyện mô hình.
- **CNN transfer learning**: ResNet18, ResNet50, EfficientNet-B0; trọng số khởi tạo từ ImageNet (nếu không dùng `--scratch`).
- **Image preprocessing**: Resize, CenterCrop, ToTensor, Normalize.
- **Data augmentation**: RandomResizedCrop, HorizontalFlip, Rotation, ColorJitter (chỉ ở train).
- **Dataset split**: phân tầng 70% train / 15% validation / 15% test, seed cố định.
- **Training**: train classifier head trước, fine-tune toàn mạng sau; AdamW, cosine LR scheduler, mixed precision khi CUDA.
- **Evaluation**: accuracy, precision/recall/F1, confusion matrix và danh sách lỗi.
- **Explainability**: Grad-CAM làm nổi bật vùng ảnh đóng góp vào dự đoán.
- **Web demo**: Gradio upload ảnh, trả về Top-3 nhãn cùng score.

> Repo chưa kèm dataset và checkpoint đã huấn luyện. Không công bố accuracy cụ thể nếu chưa chạy `evaluate.py` trên test set.

## Cấu trúc

| File | Vai trò |
|---|---|
| `config.py` | Cấu hình dữ liệu, kích thước ảnh, seed, tên lớp |
| `data.py` | Dataset, preprocessing, augmentation, chia tập |
| `nets.py` | ResNet18/50, EfficientNet-B0 và target layer Grad-CAM |
| `train.py` | Huấn luyện head và fine-tuning |
| `evaluate.py` | Metrics, confusion matrix, file ảnh dự đoán sai |
| `gradcam.py` | Trực quan hóa vùng mô hình chú ý |
| `app.py` | Web demo Gradio |
| `requirements.txt` | Thư viện Python |

## Chạy local

Khuyến nghị Python 3.10 hoặc 3.11.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Dataset cần theo cấu trúc `data/raw-img/<class_name>/<image_file>`. Dataset Animals-10 có 10 thư mục lớp tên tiếng Ý. Xem phần Google Colab bên dưới để tải và chuẩn bị.

### Huấn luyện

```bash
python data.py
python train.py --model efficientnet_b0
```

Mô hình checkpoint tốt nhất theo validation accuracy được lưu tại `outputs/efficientnet_b0.pt`.

### Đánh giá & Grad-CAM

```bash
python evaluate.py --ckpt outputs/efficientnet_b0.pt
python gradcam.py --ckpt outputs/efficientnet_b0.pt --n 8
python gradcam.py --ckpt outputs/efficientnet_b0.pt --n 8 --wrong
```

Các báo cáo/ảnh trực quan nằm trong `outputs/`.

### Chạy website demo

```bash
python app.py --ckpt outputs/efficientnet_b0.pt
```

Mở URL local do Gradio hiển thị (thường `http://127.0.0.1:7860`). Trên máy host, app bind `0.0.0.0`, port mặc định 7860 hoặc lấy từ biến môi trường `PORT`. Không dùng `share=True` trong môi trường deploy; hosting sẽ cấp URL công khai.

Nếu checkpoint được đặt ở đường dẫn khác, đặt biến môi trường `MODEL_PATH` hoặc truyền `--ckpt`.

## Huấn luyện trên Google Colab (GPU)

1. Tạo notebook Colab và bật GPU (Runtime → Change runtime type → T4 GPU, nếu khả dụng).
2. Clone repo và cài requirements.
3. Tải dataset Animals-10 từ Kaggle; không đưa `kaggle.json` lên GitHub.
4. Giải nén sao cho `data/raw-img/` có đủ 10 folder lớp.
5. Chạy train/evaluate như lệnh phía trên; sao lưu checkpoint outputs ra Drive.

Ví dụ:

```bash
git clone https://github.com/nhuvat/Project-xulyanh-phanloaidongvat.git
cd Project-xulyanh-phanloaidongvat
pip install -r requirements.txt
```

## Deploy thành website public

Hugging Face Spaces hỗ trợ Gradio. Tham khảo `README_HF_SPACE.md` để thiết lập Space và checkpoint. **Chỉ push source code thì website chưa có khả năng dự đoán**, vì file model `.pt` và dataset hiện không nằm trong repo.

## Push cập nhật lên GitHub

```bash
git status
git add README.md README_HF_SPACE.md app.py
git commit -m "Improve animal classification web demo and documentation"
git push origin main
```

Nếu nhánh repo của bạn tên `master`, thay `main` bằng `master`.

## Giới hạn & hướng phát triển

- Mô hình chỉ phân loại trong 10 lớp Animals-10, không phải detector tổng quát cho mọi loài.
- Ảnh chứa nhiều động vật, ảnh ngoài phân phối dataset hoặc ảnh chất lượng thấp có thể dự đoán sai.
- Score không đồng nghĩa xác suất hiệu chỉnh chuẩn.
- Có thể mở rộng: thêm dataset/lớp, cân bằng dữ liệu, so sánh model, Grad-CAM trên ảnh upload, tối ưu inference, và đóng gói API.

## Dataset & quyền sử dụng

Kiểm tra giấy phép và điều khoản của dataset Animals-10 cũng như pretrained weights trước khi phân phối ứng dụng hoặc dữ liệu.
