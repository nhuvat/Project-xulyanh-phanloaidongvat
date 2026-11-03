---
title: Animal Image Classification
emoji: 🐾
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.20.1
app_file: app.py
pinned: false
---

# Animal Image Classification — Animals-10

Demo phân loại ảnh động vật bằng transfer learning (EfficientNet-B0 / ResNet) với PyTorch và Gradio.

## Trước khi deploy

**Repository cần có checkpoint model** (file `.pt`) đã huấn luyện. Repo GitHub hiện bỏ qua `outputs/` và file `.pt` qua `.gitignore`, vì vậy file model chưa tự được deploy cùng source.

1. Huấn luyện theo README bằng dataset Animals-10.
2. Đặt checkpoint được tạo, ví dụ `outputs/efficientnet_b0.pt`, vào nơi lưu trữ phù hợp.
3. Trên Space, tải checkpoint vào repo Space hoặc thiết lập biến môi trường `MODEL_PATH` trỏ tới checkpoint có sẵn.
4. Deploy. Space chạy Gradio app tại cổng 7860.

Không commit dữ liệu dataset lớn hoặc credentials Kaggle vào repository.
