"""Grad-CAM: xem mô hình nhìn vào vùng nào của ảnh khi dự đoán.

Ví dụ: python gradcam.py --ckpt outputs/efficientnet_b0.pt --n 8
       python gradcam.py --ckpt outputs/efficientnet_b0.pt --wrong   # chỉ lấy ảnh dự đoán sai
"""
import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config
import data
import nets


class GradCAM:
    def __init__(self, model, layer):
        self.model = model
        self.acts = None
        self.grads = None
        layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inp, out):
        self.acts = out.detach()
        out.register_hook(lambda g: setattr(self, "grads", g.detach()))

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)
        if class_idx is None:
            class_idx = logits.argmax(1).item()
        logits[0, class_idx].backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy(), class_idx, probs[0].detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--wrong", action="store_true", help="chỉ lấy ảnh dự đoán sai")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device)
    classes = ck["classes"]
    model = nets.build_model(ck["name"], len(classes), pretrained=False).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    cam_fn = GradCAM(model, nets.target_layer(model, ck["name"]))

    _, eval_tf = data.get_transforms(augment=False)
    d = data.get_data(augment=False)
    paths, targets = d["test_paths"], d["test_targets"]

    random.seed(config.SEED)
    order = list(range(len(paths)))
    random.shuffle(order)

    results = []
    for i in order:
        img = Image.open(paths[i]).convert("RGB")
        x = eval_tf(img).unsqueeze(0).to(device)
        cam, pred, probs = cam_fn(x)
        if args.wrong and pred == targets[i]:
            continue
        # ảnh gốc đã resize/crop giống lúc đưa vào mô hình để overlay khớp
        shown = np.asarray(img.resize((256, 256)).crop((16, 16, 240, 240))) / 255.0
        results.append((shown, cam, classes[targets[i]], classes[pred], probs[pred]))
        if len(results) >= args.n:
            break

    cols = 4
    rows = (len(results) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (shown, cam, true, pred, conf) in zip(axes, results):
        heat = plt.cm.jet(cam)[..., :3]
        ax.imshow(0.55 * shown + 0.45 * heat)
        ax.set_title(f"thật: {true}\ndự đoán: {pred} ({conf:.0%})", fontsize=9,
                     color="green" if true == pred else "red")
    fig.tight_layout()
    tag = Path(args.ckpt).stem + ("_wrong" if args.wrong else "")
    out = config.OUT_DIR / f"{tag}_gradcam.png"
    fig.savefig(out, dpi=150)
    print("Đã lưu", out)


if __name__ == "__main__":
    main()
