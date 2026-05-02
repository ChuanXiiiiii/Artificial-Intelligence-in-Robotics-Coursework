from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PREPROCESSING_VERSION = "v1_median_impute_continuous_mode_binary_standardscale_continuous_drop_species"
BASE_CONTINUOUS_FEATURES = [
    "stench",
    "sound",
    "intelligence",
    "weight",
    "height",
    "strength",
    "heat",
]
OPTIONAL_CONTINUOUS_FEATURES = ["bribe"]
BINARY_FEATURES = ["magic", "flight"]


@dataclass
class PreprocessedData:
    scaled_features: pd.DataFrame
    imputed_features: pd.DataFrame
    species_labels: pd.Series
    feature_columns: List[str]
    metadata: Dict[str, Any]


def resolve_task2_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_default_data_path() -> Path:
    return resolve_task2_root().parent / "dungeon_sensorstats.csv"


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def write_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def software_versions() -> Dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "numpy": _package_version("numpy"),
        "pandas": _package_version("pandas"),
        "scikit-learn": _package_version("scikit-learn"),
        "matplotlib": _package_version("matplotlib"),
        "umap-learn": _package_version("umap-learn"),
    }


def _build_feature_columns(include_bribe: bool) -> List[str]:
    continuous = BASE_CONTINUOUS_FEATURES.copy()
    if include_bribe:
        continuous.extend(OPTIONAL_CONTINUOUS_FEATURES)
    return continuous + BINARY_FEATURES


def load_raw_dataset(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    required_cols = set(["species", *BASE_CONTINUOUS_FEATURES, *BINARY_FEATURES])
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def build_preprocessed_data(df: pd.DataFrame, include_bribe: bool) -> PreprocessedData:
    feature_columns = _build_feature_columns(include_bribe=include_bribe)
    species = df["species"].astype(str).copy()
    raw = df[feature_columns].copy()

    continuous_cols = BASE_CONTINUOUS_FEATURES.copy()
    if include_bribe:
        continuous_cols.extend(OPTIONAL_CONTINUOUS_FEATURES)
    binary_cols = BINARY_FEATURES.copy()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "continuous",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                continuous_cols,
            ),
            (
                "binary",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))]),
                binary_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    transformed = preprocessor.fit_transform(raw)
    transformed = np.asarray(transformed)
    scaled_df = pd.DataFrame(transformed, columns=feature_columns)

    imputed_df = raw.copy()
    for col in continuous_cols:
        median_value = float(imputed_df[col].median())
        imputed_df[col] = imputed_df[col].fillna(median_value)
    for col in binary_cols:
        mode_series = imputed_df[col].mode(dropna=True)
        fill_value = int(mode_series.iloc[0]) if not mode_series.empty else 0
        imputed_df[col] = imputed_df[col].fillna(fill_value)

    metadata: Dict[str, Any] = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "include_bribe": include_bribe,
        "feature_columns": feature_columns,
        "continuous_columns": continuous_cols,
        "binary_columns": binary_cols,
        "n_rows": int(len(df)),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    return PreprocessedData(
        scaled_features=scaled_df,
        imputed_features=imputed_df,
        species_labels=species,
        feature_columns=feature_columns,
        metadata=metadata,
    )


def save_preprocessed_artifacts(data: PreprocessedData, tables_dir: Path, artifact_prefix: str) -> None:
    ensure_dirs(tables_dir)

    scaled = data.scaled_features.copy()
    scaled.insert(0, "row_id", np.arange(len(scaled), dtype=int))
    scaled.to_csv(tables_dir / f"{artifact_prefix}_features_scaled.csv", index=False)

    imputed = data.imputed_features.copy()
    imputed.insert(0, "row_id", np.arange(len(imputed), dtype=int))
    imputed.to_csv(tables_dir / f"{artifact_prefix}_features_imputed.csv", index=False)

    labels = pd.DataFrame(
        {
            "row_id": np.arange(len(data.species_labels), dtype=int),
            "species": data.species_labels.values,
        }
    )
    labels.to_csv(tables_dir / f"{artifact_prefix}_species_labels.csv", index=False)

    feature_table = pd.DataFrame(
        {
            "feature": data.feature_columns,
            "feature_type": [
                "binary" if col in BINARY_FEATURES else "continuous"
                for col in data.feature_columns
            ],
        }
    )
    feature_table.to_csv(tables_dir / f"{artifact_prefix}_feature_columns.csv", index=False)

    write_json(data.metadata, tables_dir / f"{artifact_prefix}_metadata.json")


def load_preprocessed_artifacts(tables_dir: Path, artifact_prefix: str) -> PreprocessedData:
    scaled_path = tables_dir / f"{artifact_prefix}_features_scaled.csv"
    imputed_path = tables_dir / f"{artifact_prefix}_features_imputed.csv"
    labels_path = tables_dir / f"{artifact_prefix}_species_labels.csv"
    meta_path = tables_dir / f"{artifact_prefix}_metadata.json"

    for path in [scaled_path, imputed_path, labels_path, meta_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing preprocessing artifact: {path}")

    scaled_df = pd.read_csv(scaled_path).drop(columns=["row_id"])
    imputed_df = pd.read_csv(imputed_path).drop(columns=["row_id"])
    labels_df = pd.read_csv(labels_path)
    metadata = read_json(meta_path)

    return PreprocessedData(
        scaled_features=scaled_df,
        imputed_features=imputed_df,
        species_labels=labels_df["species"].astype(str),
        feature_columns=list(scaled_df.columns),
        metadata=metadata,
    )


def evaluate_internal_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        return {
            "silhouette": float("nan"),
            "davies_bouldin": float("nan"),
            "calinski_harabasz": float("nan"),
        }

    return {
        "silhouette": float(silhouette_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
    }


def evaluate_optional_external_metrics(y_true: Iterable[str], y_pred: np.ndarray) -> Dict[str, float]:
    true = np.asarray(list(y_true))
    pred = np.asarray(y_pred)
    return {
        "ari": float(adjusted_rand_score(true, pred)),
        "nmi": float(normalized_mutual_info_score(true, pred)),
    }


def select_best_candidate(df: pd.DataFrame) -> pd.Series:
    ranked = df.sort_values(
        by=["silhouette", "davies_bouldin", "calinski_harabasz"],
        ascending=[False, True, False],
    ).reset_index(drop=True)
    return ranked.iloc[0]


def build_cluster_profile(
    imputed_features: pd.DataFrame,
    labels: np.ndarray,
    method: str,
    seed: int,
    n_clusters: int,
    top_n: int = 3,
) -> pd.DataFrame:
    frame = imputed_features.copy()
    frame["cluster"] = labels

    overall_mean = imputed_features.mean(axis=0)
    overall_std = imputed_features.std(axis=0).replace(0.0, np.nan)

    rows: List[Dict[str, Any]] = []
    for cluster_id, part in frame.groupby("cluster"):
        cluster_int = int(np.asarray(cluster_id).item())
        row: Dict[str, Any] = {
            "method": method,
            "seed": int(seed),
            "n_clusters": int(n_clusters),
            "cluster": cluster_int,
            "size": int(len(part)),
            "ratio": float(len(part) / len(frame)),
        }

        z_scores: Dict[str, float] = {}
        for col in imputed_features.columns:
            cluster_mean = float(part[col].mean())
            row[f"mean_{col}"] = cluster_mean
            std_val = float(overall_std[col]) if not pd.isna(overall_std[col]) else np.nan
            z = (cluster_mean - float(overall_mean[col])) / std_val if std_val and not np.isnan(std_val) else 0.0
            z_scores[col] = float(z)
            row[f"z_{col}"] = float(z)

        top = sorted(z_scores.items(), key=lambda item: abs(item[1]), reverse=True)[:top_n]
        row["top_features_by_abs_z"] = "; ".join([f"{name}:{value:.2f}" for name, value in top])
        rows.append(row)

    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)


def now_stamp() -> str:
    return datetime.now().isoformat(timespec="seconds")
