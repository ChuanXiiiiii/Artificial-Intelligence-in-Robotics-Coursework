from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, cast

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate HOG + SVM baseline.")
    parser.add_argument("--manifest", type=Path, default=Path("outputs/tables/split_manifest.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=80)
    parser.add_argument("--c-values", type=str, default="0.1,1.0,3.0")
    parser.add_argument("--max-iter", type=int, default=30000)
    parser.add_argument("--svm-tol", type=float, default=1e-4)
    parser.add_argument("--svm-dual", type=str, choices=["true", "false"], default="false")
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    return parser.parse_args()


def build_hog_features(paths: List[str], image_size: int) -> np.ndarray:
    feats: List[np.ndarray] = []
    for path in tqdm(paths, desc="HOG features"):
        img = Image.open(path).convert("L").resize((image_size, image_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        feature = hog(
            arr,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
        )
        feats.append(feature)
    return np.asarray(feats, dtype=np.float32)


def evaluate_and_save(
    method_name: str,
    seed: int,
    split_name: str,
    class_names: List[str],
    y_true: List[str],
    y_pred: List[str],
    df_pred: pd.DataFrame,
    outputs_root: Path,
) -> Dict[str, Any]:
    figures_dir = outputs_root / "figures"
    tables_dir = outputs_root / "tables"
    logs_dir = outputs_root / "logs"
    ensure_dirs(figures_dir, tables_dir, logs_dir)

    summary, per_class_df, cm_raw, cm_norm = build_classification_outputs(y_true, y_pred, class_names)
    summary["method"] = method_name
    summary["seed"] = seed
    summary["split"] = split_name

    per_class_path = tables_dir / f"per_class_{method_name}_seed{seed}_{split_name}.csv"
    pred_path = tables_dir / f"predictions_{method_name}_seed{seed}_{split_name}.csv"
    metrics_json_path = tables_dir / f"metrics_{method_name}_seed{seed}_{split_name}.json"
    metrics_csv_path = tables_dir / f"metrics_{method_name}_seed{seed}_{split_name}.csv"
    err_path = tables_dir / f"error_examples_{method_name}_seed{seed}_{split_name}.csv"

    per_class_df.to_csv(per_class_path, index=False)
    df_pred.to_csv(pred_path, index=False)
    pd.DataFrame([summary]).to_csv(metrics_csv_path, index=False)
    save_error_examples(df_pred, err_path)
    write_json(summary, metrics_json_path)

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


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    manifest = pd.read_csv(args.manifest)
    class_names = get_class_names(manifest["label"].tolist())

    train_df = manifest[manifest["split"] == "train"].reset_index(drop=True)
    val_df = manifest[manifest["split"] == "val"].reset_index(drop=True)
    test_df = manifest[manifest["split"] == "test"].reset_index(drop=True)

    x_train = build_hog_features(train_df["image_path"].tolist(), args.image_size)
    x_val = build_hog_features(val_df["image_path"].tolist(), args.image_size)
    x_test = build_hog_features(test_df["image_path"].tolist(), args.image_size)

    y_train = train_df["label"].tolist()
    y_val = val_df["label"].tolist()
    y_test = test_df["label"].tolist()

    c_values = [float(item.strip()) for item in args.c_values.split(",") if item.strip()]

    best_model: Pipeline | None = None
    best_c = None
    best_val_f1 = -1.0
    svm_dual = args.svm_dual.lower() == "true"

    for c_val in c_values:
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "svm",
                    LinearSVC(
                        C=c_val,
                        class_weight="balanced",
                        max_iter=args.max_iter,
                        tol=args.svm_tol,
                        dual=svm_dual,
                        random_state=args.seed,
                    ),
                ),
            ]
        )
        model.fit(x_train, y_train)
        val_pred = model.predict(x_val)
        val_f1 = f1_score(y_val, val_pred, average="macro")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model = model
            best_c = c_val

    if best_model is None:
        raise RuntimeError("No SVM model was trained.")

    outputs_root = args.outputs_root.resolve()
    ensure_dirs(outputs_root / "models", outputs_root / "logs")

    model_path = outputs_root / "models" / f"hog_svm_seed{args.seed}.joblib"
    joblib.dump({"model": best_model, "class_names": class_names, "best_c": best_c}, model_path)

    # Validation summary for model-selection transparency.
    val_pred = cast(List[str], np.asarray(best_model.predict(x_val)).tolist())
    val_pred_df = val_df[["image_path", "label"]].rename(columns={"label": "label_true"})
    val_pred_df["label_pred"] = val_pred
    val_summary = evaluate_and_save(
        method_name="hog_svm",
        seed=args.seed,
        split_name="val",
        class_names=class_names,
        y_true=y_val,
        y_pred=val_pred,
        df_pred=val_pred_df,
        outputs_root=outputs_root,
    )

    test_pred = cast(List[str], np.asarray(best_model.predict(x_test)).tolist())
    test_pred_df = test_df[["image_path", "label"]].rename(columns={"label": "label_true"})
    test_pred_df["label_pred"] = test_pred
    test_summary = evaluate_and_save(
        method_name="hog_svm",
        seed=args.seed,
        split_name="test",
        class_names=class_names,
        y_true=y_test,
        y_pred=test_pred,
        df_pred=test_pred_df,
        outputs_root=outputs_root,
    )

    run_log = {
        "method": "hog_svm",
        "seed": args.seed,
        "best_c": best_c,
        "svm_dual": svm_dual,
        "max_iter": args.max_iter,
        "svm_tol": args.svm_tol,
        "val_macro_f1": val_summary["macro_f1"],
        "test_macro_f1": test_summary["macro_f1"],
        "test_accuracy": test_summary["accuracy"],
        "model_path": str(model_path),
    }
    write_json(run_log, outputs_root / "logs" / f"run_hog_svm_seed{args.seed}.json")

    print("HOG+SVM finished")
    print(run_log)


if __name__ == "__main__":
    main()
