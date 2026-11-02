"""Đánh giá trên tập test: accuracy, precision/recall/F1, confusion matrix, danh sách ảnh dự đoán sai.

Ví dụ: python evaluate.py --ckpt outputs/efficientnet_b0.pt
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

import config
import data
import nets


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    probs = []
    for x, _ in loader:
        probs.append(torch.softmax(model(x.to(device)), dim=1).cpu())
    return torch.cat(probs).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device)
    classes = ck["classes"]
    model = nets.build_model(ck["name"], len(classes), pretrained=False).to(device)
    model.load_state_dict(ck["model"])

    d = data.get_data(augment=False)
    probs = predict_all(model, d["loaders"]["test"], device)
    y_true = np.array(d["test_targets"])
    y_pred = probs.argmax(1)
    tag = Path(args.ckpt).stem

    acc = (y_true == y_pred).mean()
    report = classification_report(y_true, y_pred, target_names=classes, digits=4)
    print(f"Test accuracy: {acc:.4f}\n")
    print(report)
    (config.OUT_DIR / f"{tag}_report.txt").write_text(f"Test accuracy: {acc:.4f}\n\n{report}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    ax.set_title(f"Confusion matrix ({tag}) - acc {acc:.2%}")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(config.OUT_DIR / f"{tag}_confusion.png", dpi=150)

    # Các cặp lớp hay nhầm nhất
    pairs = [(cm[i, j], classes[i], classes[j])
             for i in range(len(classes)) for j in range(len(classes)) if i != j]
    print("Top 5 cặp hay nhầm (thực tế -> dự đoán):")
    for n, a, b in sorted(pairs, reverse=True)[:5]:
        print(f"  {a} -> {b}: {n} ảnh")

    # Danh sách ảnh sai, sắp xếp theo độ tự tin giảm dần
    wrong = np.where(y_true != y_pred)[0]
    wrong = sorted(wrong, key=lambda i: -probs[i].max())
    with open(config.OUT_DIR / f"{tag}_errors.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "true", "pred", "confidence"])
        for i in wrong:
            w.writerow([d["test_paths"][i], classes[y_true[i]], classes[y_pred[i]],
                        f"{probs[i].max():.4f}"])
    print(f"\nĐã lưu: outputs/{tag}_report.txt, _confusion.png, _errors.csv ({len(wrong)} ảnh sai)")


if __name__ == "__main__":
    main()
