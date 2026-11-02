# Phân loại động vật từ ảnh (Animals-10, transfer learning)

Project đồ án: fine-tune mô hình pretrained (EfficientNet-B0 / ResNet) để phân loại 10 loài động vật,
có đánh giá chi tiết, Grad-CAM và demo web. Thiết kế để làm xong trong khoảng 2 ngày.

## Cấu trúc

| File | Vai trò |
|---|---|
| `config.py` | Đường dẫn, hằng số, bảng đổi tên lớp (Ý -> Anh) |
| `data.py` | Đọc dữ liệu, augmentation, chia train/val/test 70/15/15 có phân tầng |
| `nets.py` | Dựng mô hình (resnet18, resnet50, efficientnet_b0) |
| `train.py` | Huấn luyện 2 giai đoạn (train head -> fine-tune toàn bộ) |
| `evaluate.py` | Accuracy, F1, confusion matrix, danh sách ảnh sai |
| `gradcam.py` | Grad-CAM trực quan hóa vùng mô hình chú ý |
| `app.py` | Demo Gradio |

## Chạy trên Google Colab

1. Runtime -> Change runtime type -> **T4 GPU**.
2. Upload toàn bộ file của project lên Colab (hoặc `git clone` repo của nhóm), rồi chạy lần lượt:

```python
# Ô 1: cài thư viện
!pip install -q -r requirements.txt

# Ô 2: tải dataset (upload file kaggle.json trước)
!mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d alessiocorrado99/animals10
!unzip -q animals10.zip -d data
# Sau bước này phải có thư mục data/raw-img/ chứa 10 thư mục lớp

# Ô 3: kiểm tra dữ liệu và cách chia tập
!python data.py
```

Nếu tải Kaggle bị lỗi: tải tay file zip từ trang dataset, upload lên Google Drive, mount Drive vào Colab rồi giải nén.

## Thí nghiệm (chạy theo thứ tự)

```bash
# 1) Baseline: ResNet18 pretrained, chỉ train lớp cuối (nhanh nhất)
python train.py --model resnet18 --epochs_ft 0 --tag baseline_resnet18

# 2) Mô hình chính: EfficientNet-B0, train head 2 epoch rồi fine-tune 6 epoch
python train.py --model efficientnet_b0

# 3) (Nếu còn thời gian) Ablation: bỏ augmentation
python train.py --model efficientnet_b0 --no_aug

# 4) (Tùy chọn) ResNet50 để so sánh thêm
python train.py --model resnet50
```

Mỗi lần chạy lưu `outputs/<tag>.pt` (checkpoint tốt nhất theo val acc) và `outputs/<tag>_history.json`.

## Đánh giá và giải thích mô hình

```bash
python evaluate.py --ckpt outputs/efficientnet_b0.pt
python evaluate.py --ckpt outputs/baseline_resnet18.pt

python gradcam.py --ckpt outputs/efficientnet_b0.pt --n 8          # ảnh ngẫu nhiên
python gradcam.py --ckpt outputs/efficientnet_b0.pt --n 8 --wrong  # ảnh dự đoán sai
```

Kết quả nằm trong `outputs/`: `*_report.txt`, `*_confusion.png`, `*_errors.csv`, `*_gradcam.png`.

## Demo

```bash
python app.py --ckpt outputs/efficientnet_b0.pt
```

Gradio in ra link public (`...gradio.live`), mở bằng điện thoại hoặc laptop để thử upload ảnh.

## Gợi ý cho báo cáo

- **Bảng so sánh**: baseline vs mô hình chính (và có/không augmentation) trên tập test.
- **Phân tích lỗi**: xem cặp lớp hay nhầm trong log của `evaluate.py` và file `_errors.csv`.
- **Grad-CAM**: chọn vài ảnh đúng và vài ảnh sai, nhận xét mô hình nhìn vào con vật hay nhìn vào nền.
- **Hạn chế / hướng phát triển**: dữ liệu mất cân bằng (thử `--class_weights`), ảnh nhiều con vật, tách nền,
  thêm mô hình khác, phân loại giống trong cùng loài.

## Lưu ý

- Chia tập được lưu ở `outputs/splits.json`, nên mọi thí nghiệm dùng chung một cách chia và so sánh được với nhau.
  Nếu đổi dataset thì xóa file này.
- Lưu thư mục `outputs/` lên Google Drive thường xuyên để không mất checkpoint khi Colab ngắt.
- Chưa có số liệu accuracy thật trong repo: cần chạy huấn luyện để có kết quả cho báo cáo.
