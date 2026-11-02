"""Demo web: upload ảnh động vật, trả về top-3 dự đoán.

Chạy: python app.py --ckpt outputs/efficientnet_b0.pt
(Trên Colab sẽ in ra link public, chia sẻ được cho cả nhóm.)
"""
import argparse

import gradio as gr
import torch

import data
import nets

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="outputs/efficientnet_b0.pt")
args = ap.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck = torch.load(args.ckpt, map_location=device)
classes = ck["classes"]
model = nets.build_model(ck["name"], len(classes), pretrained=False).to(device)
model.load_state_dict(ck["model"])
model.eval()
_, eval_tf = data.get_transforms(augment=False)


@torch.no_grad()
def predict(img):
    x = eval_tf(img.convert("RGB")).unsqueeze(0).to(device)
    probs = torch.softmax(model(x), dim=1)[0].cpu()
    return {classes[i]: float(probs[i]) for i in range(len(classes))}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Ảnh động vật"),
    outputs=gr.Label(num_top_classes=3, label="Dự đoán"),
    title="Phân loại động vật từ ảnh",
    description=f"Mô hình: {ck['name']} | 10 lớp: " + ", ".join(classes),
)

if __name__ == "__main__":
    demo.launch(share=True)
