"""Huấn luyện 2 giai đoạn: (1) chỉ train head, (2) mở khóa toàn bộ mạng với lr nhỏ.

Ví dụ:
  python train.py --model efficientnet_b0                 # mô hình chính
  python train.py --model resnet18 --epochs_ft 0          # baseline (chỉ train head)
  python train.py --model resnet18 --scratch --epochs_ft 10   # baseline train từ đầu
  python train.py --model efficientnet_b0 --no_aug        # ablation: bỏ augmentation
"""
import argparse
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn

import config
import data
import nets


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, use_amp=False):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, count = 0.0, 0, 0
    with torch.enable_grad() if train else torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(x)
                loss = criterion(out, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            count += x.size(0)
    return total_loss / count, correct / count


def fit(model, d, criterion, device, epochs, lr, state, args, tag, phase):
    if epochs <= 0:
        return
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, d["loaders"]["train"], criterion, device,
                                    optimizer, scaler, use_amp)
        va_loss, va_acc = run_epoch(model, d["loaders"]["val"], criterion, device,
                                    use_amp=use_amp)
        scheduler.step()
        state["history"].append({
            "phase": phase, "epoch": ep, "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": va_loss, "val_acc": va_acc})
        flag = ""
        if va_acc > state["best"]:
            state["best"] = va_acc
            torch.save({"model": model.state_dict(), "classes": d["classes"],
                        "name": args.model, "val_acc": va_acc},
                       config.OUT_DIR / f"{tag}.pt")
            flag = "  <- lưu checkpoint"
        print(f"[{phase}] epoch {ep}/{epochs} | train {tr_loss:.4f}/{tr_acc:.4f} | "
              f"val {va_loss:.4f}/{va_acc:.4f} | {time.time() - t0:.0f}s{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnet_b0", choices=nets.SUPPORTED)
    ap.add_argument("--epochs_head", type=int, default=2, help="số epoch chỉ train head")
    ap.add_argument("--epochs_ft", type=int, default=6, help="số epoch fine-tune toàn bộ")
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_ft", type=float, default=1e-4)
    ap.add_argument("--scratch", action="store_true", help="không dùng trọng số pretrained")
    ap.add_argument("--no_aug", action="store_true", help="tắt augmentation (ablation)")
    ap.add_argument("--class_weights", action="store_true", help="dùng trọng số lớp")
    ap.add_argument("--tag", default=None, help="tên lưu checkpoint")
    args = ap.parse_args()

    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Thiết bị:", device)

    tag = args.tag or (args.model + ("_scratch" if args.scratch else "")
                       + ("_noaug" if args.no_aug else ""))

    d = data.get_data(augment=not args.no_aug)
    print("Lớp:", d["classes"])
    print("Số ảnh train theo lớp:", d["counts"])

    model = nets.build_model(args.model, len(d["classes"]), pretrained=not args.scratch).to(device)
    weight = d["class_weights"].to(device) if args.class_weights else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    state = {"best": 0.0, "history": []}

    if args.scratch:
        # Train từ đầu: bỏ giai đoạn đóng băng, dùng lr lớn hơn
        nets.set_backbone_trainable(model, args.model, True)
        fit(model, d, criterion, device, args.epochs_ft, args.lr_head, state, args, tag, "scratch")
    else:
        nets.set_backbone_trainable(model, args.model, False)
        fit(model, d, criterion, device, args.epochs_head, args.lr_head, state, args, tag, "head")
        nets.set_backbone_trainable(model, args.model, True)
        fit(model, d, criterion, device, args.epochs_ft, args.lr_ft, state, args, tag, "finetune")

    (config.OUT_DIR / f"{tag}_history.json").write_text(json.dumps(state["history"], indent=2))
    print(f"Xong. Val acc tốt nhất: {state['best']:.4f} | checkpoint: outputs/{tag}.pt")


if __name__ == "__main__":
    main()
