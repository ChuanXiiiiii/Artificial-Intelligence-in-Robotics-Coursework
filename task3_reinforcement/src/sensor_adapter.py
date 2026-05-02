from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


SENSOR_COLUMNS = [
    "stench",
    "sound",
    "intelligence",
    "weight",
    "height",
    "strength",
    "heat",
    "magic",
    "flight",
]


@dataclass
class SensorObs:
    sensor_values: np.ndarray
    cluster_id: int


class SensorFeatureAdapter:
    """Builds synthetic sensor observation from nearby virtual entities."""

    def __init__(
        self,
        project_root: Path,
        fixed_cluster_k: int = 6,
        seed: int = 42,
    ) -> None:
        self.project_root = project_root
        self.fixed_cluster_k = fixed_cluster_k
        self.seed = seed
        self.raw_species_prototypes = self._load_species_prototypes()
        self.species_prototypes = self._scale_species_prototypes(self.raw_species_prototypes)
        self.cluster_centers = self._load_cluster_centers()

    def _load_species_prototypes(self) -> Dict[str, np.ndarray]:
        csv_path = self.project_root / "dungeon_sensorstats.csv"
        df = pd.read_csv(csv_path)
        cols = [col for col in SENSOR_COLUMNS if col in df.columns]
        clean_df = df.copy()
        for col in cols:
            if pd.api.types.is_numeric_dtype(clean_df[col]):
                clean_df[col] = clean_df[col].fillna(clean_df[col].median())
            else:
                mode = clean_df[col].mode(dropna=True)
                clean_df[col] = clean_df[col].fillna(mode.iloc[0] if not mode.empty else 0)

        clean_df["species"] = clean_df["species"].astype(str).str.replace("_", "", regex=False)
        grouped = clean_df.groupby("species", as_index=True)[cols].median()
        prototypes: Dict[str, np.ndarray] = {}
        for species, row in grouped.iterrows():
            prototypes[str(species)] = row.to_numpy(dtype=np.float32)
        return prototypes

    def _scale_species_prototypes(self, raw_prototypes: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        csv_path = self.project_root / "dungeon_sensorstats.csv"
        df = pd.read_csv(csv_path)
        cols = [col for col in SENSOR_COLUMNS if col in df.columns]
        clean_df = df.copy()
        for col in cols:
            if pd.api.types.is_numeric_dtype(clean_df[col]):
                clean_df[col] = clean_df[col].fillna(clean_df[col].median())
            else:
                mode = clean_df[col].mode(dropna=True)
                clean_df[col] = clean_df[col].fillna(mode.iloc[0] if not mode.empty else 0)

        continuous_cols = ["stench", "sound", "intelligence", "weight", "height", "strength", "heat"]
        means = clean_df[continuous_cols].mean(axis=0)
        stds = clean_df[continuous_cols].std(axis=0, ddof=0).replace(0, 1.0)

        scaled: Dict[str, np.ndarray] = {}
        for species, raw_vec in raw_prototypes.items():
            raw_series = pd.Series(raw_vec, index=SENSOR_COLUMNS, dtype="float32")
            scaled_vec = raw_series.copy()
            for col in continuous_cols:
                scaled_vec[col] = float((raw_series[col] - means[col]) / stds[col])
            scaled[species] = scaled_vec.to_numpy(dtype=np.float32)
        return scaled

    def _load_cluster_centers(self) -> np.ndarray:
        center_path = (
            self.project_root
            / "task2_unsupervised"
            / "outputs"
            / "tables"
            / f"cluster_centers_kmeans_seed{self.seed}_k{self.fixed_cluster_k}.csv"
        )
        if not center_path.exists():
            return np.zeros((1, len(SENSOR_COLUMNS)), dtype=np.float32)

        df = pd.read_csv(center_path)
        feature_cols = [col for col in SENSOR_COLUMNS if col in df.columns]
        if not feature_cols:
            return np.zeros((1, len(SENSOR_COLUMNS)), dtype=np.float32)
        return df[feature_cols].to_numpy(dtype=np.float32)

    def _prototype(self, species: str) -> np.ndarray:
        if species in self.species_prototypes:
            return self.species_prototypes[species]
        return np.zeros((len(SENSOR_COLUMNS),), dtype=np.float32)

    def species_cluster_id(self, species: str) -> int:
        sensor = self._prototype(species)
        centers = self.cluster_centers
        dists = np.linalg.norm(centers - sensor[None, :], axis=1)
        return int(np.argmin(dists)) if len(dists) else 0

    def species_bribe_cost(self, species: str, min_cost: float = 0.08, max_cost: float = 0.35) -> float:
        sensor = self._prototype(species)
        raw_sensor = self.raw_species_prototypes.get(species, np.zeros((len(SENSOR_COLUMNS),), dtype=np.float32))
        prototypes = (
            np.stack(list(self.raw_species_prototypes.values()), axis=0)
            if self.raw_species_prototypes
            else raw_sensor[None, :]
        )

        score_parts: List[float] = []
        for col in ("weight", "height"):
            if col not in SENSOR_COLUMNS:
                continue
            idx = SENSOR_COLUMNS.index(col)
            values = prototypes[:, idx].astype(np.float32)
            lo = float(np.min(values))
            hi = float(np.max(values))
            if hi <= lo:
                continue
            score_parts.append(float((float(raw_sensor[idx]) - lo) / (hi - lo)))

        score = float(np.mean(score_parts)) if score_parts else 0.5
        score = float(np.clip(score, 0.0, 1.0))
        return float(min_cost + score * (max_cost - min_cost))

    def build_sensor_from_entities(
        self,
        robot_position: np.ndarray,
        entities: Iterable[Tuple[Tuple[int, int], str]],
    ) -> SensorObs:
        weights: List[float] = []
        vectors: List[np.ndarray] = []
        r_x, r_y = int(robot_position[0]), int(robot_position[1])
        for (x, y), species in entities:
            dist = abs(x - r_x) + abs(y - r_y)
            w = 1.0 / float(dist + 1)
            weights.append(w)
            vectors.append(self._prototype(species))

        if not vectors:
            sensor = np.zeros((len(SENSOR_COLUMNS),), dtype=np.float32)
        else:
            stacked = np.stack(vectors, axis=0)
            w_arr = np.asarray(weights, dtype=np.float32)
            sensor = (stacked * w_arr[:, None]).sum(axis=0) / float(w_arr.sum())

        centers = self.cluster_centers
        dists = np.linalg.norm(centers - sensor[None, :], axis=1)
        cluster_id = int(np.argmin(dists)) if len(dists) else 0
        return SensorObs(sensor_values=sensor.astype(np.float32), cluster_id=cluster_id)

    @property
    def cluster_count(self) -> int:
        return int(self.cluster_centers.shape[0]) if self.cluster_centers.ndim == 2 else 1
