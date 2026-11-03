"""Gradio web app for animal image classification.

Run locally: python app.py --ckpt outputs/efficientnet_b0.pt
For hosting, set MODEL_PATH to the checkpoint file path.
"""
import argparse
import os
from pathlib import Path

import gradio as gr
import torch

import data
import nets

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default=os.getenv("MODEL_PATH", "outputs/efficientnet_b0.pt"))
args, _unknown = parser.parse_known_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None
CLASSES = []
MODEL_NAME = "Chưa tải mô hình"
EVAL_TRANSFORM = None
LOAD_ERROR = None

def load_model():
    global MODEL, CLASSES, MODEL_NAME, EVAL_TRANSFORM, LOAD_ERROR
    checkpoint_path = Path(args.ckpt)
    if not checkpoint_path.is_file():
        LOAD_ERROR = (
            f"Chưa tìm thấy checkpoint tại: {checkpoint_path}. "
            "Hãy huấn luyện model hoặc đặt MODEL_PATH trỏ tới file .pt đã huấn luyện."
        )
        return
    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        CLASSES = checkpoint["classes"]
        MODEL_NAME = checkpoint["name"]
        MODEL = nets.build_model(MODEL_NAME, len(CLASSES), pretrained=False).to(DEVICE)
        MODEL.load_state_dict(checkpoint["model"])
        MODEL.eval()
        _, EVAL_TRANSFORM = data.get_transforms(augment=False)
    except Exception as exc:
        LOAD_ERROR = f"Lỗi khi tải mô hình: {exc}"

load_model()

@torch.no_grad()
def predict(image):
    if image is None:
        return {}, "Vui lòng tải ảnh động vật lên."
    if MODEL is None:
        return {}, LOAD_ERROR or "Mô hình chưa sẵn sàng."
    tensor = EVAL_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    probabilities = torch.softmax(MODEL(tensor), dim=1)[0].cpu()
    ranking = sorted(
        ((CLASSES[i], float(probabilities[i])) for i in range(len(CLASSES))),
        key=lambda item: item[1], reverse=True,
    )
    scores = {name: score for name, score in ranking[:3]}
    top_name, top_score = ranking[0]
    message = f"Dự đoán hàng đầu: **{top_name}** — độ tin cậy mô hình: **{top_score:.1%}**"
    return scores, message

with gr.Blocks(theme=gr.themes.Soft(), title="Animal Image Classifier") as demo:
    gr.Markdown(
        """
        # 🐾 Phân loại động vật từ ảnh
        Tải hình ảnh lên để mô hình học sâu dự đoán một trong 10 lớp động vật
        thuộc bộ dữ liệu Animals-10. Kết quả hiển thị **Top-3 dự đoán**.

        > Độ tin cậy là điểm xác suất do mô hình đưa ra, không phải bảo đảm ảnh được nhận diện đúng.
        """
    )
    with gr.Row():
        image_input = gr.Image(type="pil", label="Ảnh đầu vào")
        with gr.Column():
            output_scores = gr.Label(num_top_classes=3, label="Top-3 dự đoán")
            output_message = gr.Markdown()
    predict_button = gr.Button("🔍 Nhận diện động vật", variant="primary")
    clear_button = gr.ClearButton([image_input, output_scores, output_message])
    predict_button.click(predict, inputs=image_input, outputs=[output_scores, output_message])
    gr.Markdown(f"**Mô hình:** {MODEL_NAME}  |  **Thiết bị:** {DEVICE.type.upper()}  |  **Dataset:** Animals-10")

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        share=False,
    )
