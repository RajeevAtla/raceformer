"""
CMHT dataset loader scaffolding using Grain.

This module is a placeholder; implement timestamp alignment (LiDAR master clock),
interpolation for GPS/IMU, nearest-neighbor for RGB/IR/Radar, and modality-specific
preprocessing (BEV projections, resizing, normalization).
"""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import grain as gr  # type: ignore
except ImportError:  # pragma: no cover - optional until deps installed
    gr = None


@dataclass
class CMHTSample:
    """Container for a synchronized CMHT timestep."""

    lidar: Any
    rgb: Any | None
    radar: Any | None
    ir: Any | None
    gps: Any
    imu: Any
    timestamp: float


class CMHTDataSource:
    """
    Grain-compatible data source for CMHT.

    Replace this scaffold with logic that scans the dataset, aligns timestamps,
    and yields synchronized modality batches for pretraining.
    """

    def __init__(self, data_dir: str | Path, include_modalities: tuple[str, ...] | None = None):
        self.data_dir = Path(data_dir)
        self.include_modalities = include_modalities
        if gr is not None:
            self.data_source = gr.FileDataSource(str(self.data_dir))  # placeholder
        else:
            self.data_source = None

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        if self.data_source is None:
            raise RuntimeError("Grain is not installed; install deps to use CMHTDataSource.")
        # TODO: implement actual iteration over synchronized CMHT samples.
        for _ in ():  # pragma: no cover - placeholder
            yield {}

    def _sync_modalities(self, lidar_timestamp: float) -> CMHTSample:
        """
        Synchronize modalities using LiDAR as the master clock.

        Implement interpolation for GPS/IMU and nearest-neighbor for RGB/IR/Radar.
        """
        raise NotImplementedError

    def _preprocess_lidar(self, lidar_raw: Any) -> np.ndarray:
        """Voxelize or project LiDAR to BEV images or patch tokens."""
        raise NotImplementedError

    def _preprocess_radar(self, radar_raw: Any) -> np.ndarray:
        """Voxelize or project Radar to BEV images or patch tokens."""
        raise NotImplementedError

    def _preprocess_image(self, image_raw: Any, *, is_ir: bool = False) -> np.ndarray:
        """Resize to ViT resolution and normalize (ImageNet for RGB, min/max for range data)."""
        raise NotImplementedError
