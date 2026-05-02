from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


CLASS_ORDER = ["halfling", "human", "lizard", "orc", "wingedrat"]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        # Torch may be unavailable for Method A only runs.
        pass


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def extract_prefix_group(file_name: str) -> str:
    stem = Path(file_name).stem
    token = stem.split("_", 1)[1] if "_" in stem else stem
    matched = re.match(r"([A-Za-z]+)", token)
    prefix = matched.group(1).upper() if matched else "UNK"
    return prefix


def get_class_names(labels: List[str]) -> List[str]:
    existing = set(labels)
    ordered = [name for name in CLASS_ORDER if name in existing]
    leftovers = sorted(existing - set(ordered))
    return ordered + leftovers


def build_classification_outputs(
    y_true: List[str], y_pred: List[str], class_names: List[str]
) -> Tuple[Dict[str, Any], pd.DataFrame, np.ndarray, np.ndarray]:
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    report = cast(
        Dict[str, Any],
        classification_report(
        y_true,
        y_pred,
        labels=class_names,
        output_dict=True,
        zero_division=0,
        ),
    )

    rows = []
    for class_name in class_names:
        class_block = cast(Dict[str, Any], report.get(class_name, {}))
        rows.append(
            {
                "class": class_name,
                "precision": float(class_block.get("precision", 0.0)),
                "recall": float(class_block.get("recall", 0.0)),
                "f1": float(class_block.get("f1-score", 0.0)),
                "support": int(class_block.get("support", 0)),
            }
        )

    cm_raw = confusion_matrix(y_true, y_pred, labels=class_names)
    cm_norm = confusion_matrix(y_true, y_pred, labels=class_names, normalize="true")

    summary = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "n_samples": int(len(y_true)),
    }

    return summary, pd.DataFrame(rows), cm_raw, cm_norm


def save_confusion_matrix_figure(
    cm: np.ndarray,
    class_names: List[str],
    title: str,
    output_path: Path,
    normalize: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_json(data: Dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def save_error_examples(
    df_pred: pd.DataFrame,
    output_path: Path,
    top_k: int = 100,
) -> None:
    errors = df_pred[df_pred["label_true"] != df_pred["label_pred"]].copy()
    if errors.empty:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=df_pred.columns.tolist()).to_csv(output_path, index=False)
        return

    pair_counts = (
        errors.groupby(["label_true", "label_pred"]).size().sort_values(ascending=False).reset_index(name="count")
    )
    errors = errors.merge(pair_counts, on=["label_true", "label_pred"], how="left")
    errors = errors.sort_values(["count", "label_true", "label_pred"], ascending=[False, True, True])
    errors.head(top_k).to_csv(output_path, index=False)
