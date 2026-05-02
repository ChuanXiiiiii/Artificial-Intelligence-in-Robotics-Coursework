from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

from common import (
    build_classification_outputs,
    ensure_dirs,
    get_class_names,
    save_confusion_matrix_figure,
    save_error_examples,
    set_global_seed,
    write_json,
)


class ImageManifestDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        class_to_idx: Dict[str, int],
        transform: transforms.Compose,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        row = self.frame.iloc[index]
        path = row["image_path"]
        image = Image.open(path).convert("RGB")
        x = cast(Tensor, self.transform(image))
        y = self.class_to_idx[row["label"]]
        return x, y, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate ResNet18 transfer learning baseline.")
    parser.add_argument("--manifest", type=Path, default=Path("outputs/tables/split_manifest.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate-head", type=float, default=1e-3)
    parser.add_argument("--learning-rate-full", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--class-weighting",
        type=str,
        choices=["none", "balanced", "sqrt_balanced"],
        default="sqrt_balanced",
    )
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    return parser.parse_args()


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    idx_to_class: Dict[int, str],
    device: torch.device,
) -> Tuple[List[str], List[str], pd.DataFrame, float]:
    model.eval()
    y_true: List[str] = []
    y_pred: List[str] = []
    rows: List[Dict] = []
    total_loss = 0.0
    n_items = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y, paths in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            n_items += x.size(0)
            pred = torch.argmax(logits, dim=1)

            for i in range(x.size(0)):
                label_true = idx_to_class[int(y[i].item())]
                label_pred = idx_to_class[int(pred[i].item())]
                y_true.append(label_true)
                y_pred.append(label_pred)
                rows.append(
                    {
                        "image_path": paths[i],
                        "label_true": label_true,
                        "label_pred": label_pred,
                    }
                )

    avg_loss = total_loss / max(n_items, 1)
    return y_true, y_pred, pd.DataFrame(rows), avg_loss


def save_outputs(
    method_name: str,
    seed: int,
    split_name: str,
    class_names: List[str],
    y_true: List[str],
    y_pred: List[str],
    pred_df: pd.DataFrame,
    outputs_root: Path,
) -> Dict[str, Any]:
    figures_dir = outputs_root / "figures"
    tables_dir = outputs_root / "tables"
    ensure_dirs(figures_dir, tables_dir)

    summary, per_class_df, cm_raw, cm_norm = build_classification_outputs(y_true, y_pred, class_names)
    summary["method"] = method_name
    summary["seed"] = seed
    summary["split"] = split_name

    per_class_df.to_csv(tables_dir / f"per_class_{method_name}_seed{seed}_{split_name}.csv", index=False)
    pred_df.to_csv(tables_dir / f"predictions_{method_name}_seed{seed}_{split_name}.csv", index=False)
    pd.DataFrame([summary]).to_csv(
        tables_dir / f"metrics_{method_name}_seed{seed}_{split_name}.csv", index=False
    )
    write_json(summary, tables_dir / f"metrics_{method_name}_seed{seed}_{split_name}.json")
    save_error_examples(pred_df, tables_dir / f"error_examples_{method_name}_seed{seed}_{split_name}.csv")

    save_confusion_matrix_figure(
        cm_raw,
        class_names,
        f"{method_name} confusion matrix ({split_name}, raw)",
        figures_dir / f"cm_{method_name}_seed{seed}_{split_name}_raw.png",
        normalize=False,
    )
    save_confusion_matrix_figure(
        cm_norm,
        class_names,
        f"{method_name} confusion matrix ({split_name}, normalized)",
        figures_dir / f"cm_{method_name}_seed{seed}_{split_name}_norm.png",
        normalize=True,
    )

    return summary


def make_dataloaders(
    manifest: pd.DataFrame,
    class_to_idx: Dict[str, int],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_tf = transforms.Compose(
        [
            transforms.Resize((80, 80)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((80, 80)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_df = manifest[manifest["split"] == "train"].reset_index(drop=True)
    val_df = manifest[manifest["split"] == "val"].reset_index(drop=True)
    test_df = manifest[manifest["split"] == "test"].reset_index(drop=True)

    train_ds = ImageManifestDataset(train_df, class_to_idx, train_tf)
    val_ds = ImageManifestDataset(val_df, class_to_idx, eval_tf)
    test_ds = ImageManifestDataset(test_df, class_to_idx, eval_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    outputs_root = args.outputs_root.resolve()
    ensure_dirs(outputs_root / "models", outputs_root / "tables", outputs_root / "figures", outputs_root / "logs")

    manifest = pd.read_csv(args.manifest)
    class_names = get_class_names(manifest["label"].tolist())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif mps_available:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    pin_memory = device.type == "cuda"

    train_loader, val_loader, test_loader = make_dataloaders(
        manifest,
        class_to_idx,
        args.batch_size,
        args.num_workers,
        pin_memory,
    )

    pretrained_used = True
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except Exception:
        model = models.resnet18(weights=None)
        pretrained_used = False

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(class_names))
    for param in model.fc.parameters():
        param.requires_grad = True

    model = model.to(device)

    train_counts = manifest[manifest["split"] == "train"]["label"].value_counts()
    class_weights = []
    for class_name in class_names:
        count = float(train_counts.get(class_name, 1.0))
        if args.class_weighting == "balanced":
            weight = 1.0 / count
        elif args.class_weighting == "sqrt_balanced":
            weight = 1.0 / np.sqrt(count)
        else:
            weight = 1.0
        class_weights.append(weight)

    class_weight_tensor = None
    if args.class_weighting != "none":
        class_weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        class_weight_tensor = class_weight_tensor / class_weight_tensor.mean()

    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate_head,
        weight_decay=args.weight_decay,
    )

    best_state = None
    best_val_f1 = -1.0
    best_epoch = -1
    patience_count = 0
    history_rows: List[Dict] = []

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        if epoch == args.warmup_epochs + 1:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate_full, weight_decay=args.weight_decay)

        model.train()
        running_loss = 0.0
        n_train_items = 0
        for x, y, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            n_train_items += x.size(0)

        train_loss = running_loss / max(n_train_items, 1)
        val_true, val_pred, _, val_loss = evaluate_model(model, val_loader, idx_to_class, device)
        val_summary, _, _, _ = build_classification_outputs(val_true, val_pred, class_names)

        history = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_summary["accuracy"]),
            "val_macro_f1": float(val_summary["macro_f1"]),
        }
        history_rows.append(history)

        if val_summary["macro_f1"] > best_val_f1:
            best_val_f1 = float(val_summary["macro_f1"])
            best_epoch = epoch
            best_state = {
                "model": model.state_dict(),
                "class_names": class_names,
            }
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= args.patience:
            break

    elapsed = time.time() - start_time

    if best_state is None:
        raise RuntimeError("ResNet18 training did not produce a valid checkpoint.")

    model.load_state_dict(best_state["model"])

    model_path = outputs_root / "models" / f"resnet18_seed{args.seed}.pt"
    torch.save(best_state, model_path)

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(outputs_root / "tables" / f"history_resnet18_seed{args.seed}.csv", index=False)

    # Save training curve.
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(history_df["epoch"], history_df["train_loss"], label="train_loss", color="tab:blue")
    ax1.plot(history_df["epoch"], history_df["val_loss"], label="val_loss", color="tab:orange")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")

    ax2 = ax1.twinx()
    ax2.plot(history_df["epoch"], history_df["val_macro_f1"], label="val_macro_f1", color="tab:green")
    ax2.set_ylabel("macro_f1")

    lines, labels = [], []
    for axis in [ax1, ax2]:
        line, label = axis.get_legend_handles_labels()
        lines.extend(line)
        labels.extend(label)
    ax1.legend(lines, labels, loc="center right")
    fig.tight_layout()
    fig.savefig(outputs_root / "figures" / f"curve_resnet18_seed{args.seed}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    val_true, val_pred, val_pred_df, _ = evaluate_model(model, val_loader, idx_to_class, device)
    val_summary = save_outputs(
        method_name="resnet18",
        seed=args.seed,
        split_name="val",
        class_names=class_names,
        y_true=val_true,
        y_pred=val_pred,
        pred_df=val_pred_df,
        outputs_root=outputs_root,
    )

    test_true, test_pred, test_pred_df, _ = evaluate_model(model, test_loader, idx_to_class, device)
    test_summary = save_outputs(
        method_name="resnet18",
        seed=args.seed,
        split_name="test",
        class_names=class_names,
        y_true=test_true,
        y_pred=test_pred,
        pred_df=test_pred_df,
        outputs_root=outputs_root,
    )

    run_log = {
        "method": "resnet18",
        "seed": args.seed,
        "device": str(device),
        "class_weighting": args.class_weighting,
        "pretrained_used": pretrained_used,
        "best_epoch": best_epoch,
        "val_macro_f1": val_summary["macro_f1"],
        "test_macro_f1": test_summary["macro_f1"],
        "test_accuracy": test_summary["accuracy"],
        "elapsed_seconds": round(elapsed, 2),
        "model_path": str(model_path),
    }
    write_json(run_log, outputs_root / "logs" / f"run_resnet18_seed{args.seed}.json")

    print("ResNet18 training finished")
    print(run_log)


if __name__ == "__main__":
    main()
