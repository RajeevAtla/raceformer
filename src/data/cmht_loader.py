"""
CMHT dataset loader scaffolding using Grain.

Implements timestamp alignment (LiDAR master clock), interpolation helpers, and
preprocessing placeholders. Replace file loading logic with actual CMHT parsing.
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
class ModalityRecord:
    """Single modality reading with timestamp and payload."""

    timestamp: float
    data: Any


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

    Scans modality streams, aligns them on LiDAR timestamps, and yields synchronized samples.
    """

    def __init__(self, data_dir: str | Path, include_modalities: tuple[str, ...] | None = None):
        self.data_dir = Path(data_dir)
        self.include_modalities = include_modalities or ("lidar", "rgb", "radar", "ir", "gps", "imu")
        if gr is not None:
            self.data_source = gr.FileDataSource(str(self.data_dir))  # placeholder for real Grain pipeline
        else:
            self.data_source = None
        self.index = self._load_index()

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        """
        Yield synchronized, preprocessed samples keyed by modality.

        Replace dummy payloads with actual decoded arrays when hooking into CMHT files.
        """
        if "lidar" not in self.index or not self.index["lidar"]:
            raise RuntimeError("No LiDAR records found; ensure CMHT data is indexed.")

        for lidar_rec in self.index["lidar"]:
            ts = lidar_rec.timestamp
            rgb = self._nearest(self.index.get("rgb", []), ts)
            radar = self._nearest(self.index.get("radar", []), ts)
            ir = self._nearest(self.index.get("ir", []), ts)
            gps_prev, gps_next = self._neighbors(self.index.get("gps", []), ts)
            imu_prev, imu_next = self._neighbors(self.index.get("imu", []), ts)

            sample = CMHTSample(
                lidar=self._preprocess_lidar(lidar_rec.data),
                rgb=self._preprocess_image(rgb.data, is_ir=False) if rgb else None,
                radar=self._preprocess_radar(radar.data) if radar else None,
                ir=self._preprocess_image(ir.data, is_ir=True) if ir else None,
                gps=self._interpolate(gps_prev, gps_next, ts),
                imu=self._interpolate(imu_prev, imu_next, ts),
                timestamp=ts,
            )
            yield {
                "lidar": sample.lidar,
                "rgb": sample.rgb,
                "radar": sample.radar,
                "ir": sample.ir,
                "gps": sample.gps,
                "imu": sample.imu,
                "timestamp": sample.timestamp,
            }

    def _load_index(self) -> dict[str, list[ModalityRecord]]:
        """
        Build an index of modality records.

        Replace this stub with real file scanning/decoding for CMHT.
        """
        index: dict[str, list[ModalityRecord]] = {m: [] for m in self.include_modalities}
        # TODO: parse CMHT structure; for now populate minimal synthetic entries to keep pipeline runnable.
        synthetic_ts = [0.0, 0.1, 0.2]
        for ts in synthetic_ts:
            index["lidar"].append(ModalityRecord(ts, np.zeros((32, 32, 1), dtype=np.float32)))
            index["rgb"].append(ModalityRecord(ts, np.zeros((32, 32, 3), dtype=np.uint8)))
            index["gps"].append(ModalityRecord(ts, np.zeros((3,), dtype=np.float32)))
            index["imu"].append(ModalityRecord(ts, np.zeros((6,), dtype=np.float32)))
        return index

    @staticmethod
    def _nearest(records: list[ModalityRecord], target_ts: float) -> ModalityRecord | None:
        if not records:
            return None
        return min(records, key=lambda r: abs(r.timestamp - target_ts))

    @staticmethod
    def _neighbors(records: list[ModalityRecord], target_ts: float) -> tuple[ModalityRecord | None, ModalityRecord | None]:
        if not records:
            return None, None
        prev = max((r for r in records if r.timestamp <= target_ts), default=None, key=lambda r: r.timestamp)
        nxt = min((r for r in records if r.timestamp >= target_ts), default=None, key=lambda r: r.timestamp)
        return prev, nxt

    @staticmethod
    def _interpolate(prev: ModalityRecord | None, nxt: ModalityRecord | None, target_ts: float) -> Any:
        """Linear interpolate between two records; fallback to nearest."""
        if prev is None and nxt is None:
            return None
        if prev is None:
            return nxt.data
        if nxt is None:
            return prev.data
        if nxt.timestamp == prev.timestamp:
            return prev.data
        alpha = (target_ts - prev.timestamp) / (nxt.timestamp - prev.timestamp)
        return (1 - alpha) * np.asarray(prev.data) + alpha * np.asarray(nxt.data)

    def _preprocess_lidar(self, lidar_raw: Any) -> np.ndarray:
        """Voxelize or project LiDAR to BEV images or patch tokens. Placeholder returns input as float32."""
        return np.asarray(lidar_raw, dtype=np.float32)

    def _preprocess_radar(self, radar_raw: Any) -> np.ndarray:
        """Voxelize or project Radar to BEV images or patch tokens. Placeholder returns input as float32."""
        return np.asarray(radar_raw, dtype=np.float32)

    def _preprocess_image(self, image_raw: Any, *, is_ir: bool = False) -> np.ndarray:
        """
        Resize to ViT resolution and normalize (ImageNet for RGB; min/max for range data).
        Placeholder passes through after scaling to float32.
        """
        img = np.asarray(image_raw)
        if not is_ir:
            img = img.astype(np.float32) / 255.0
        else:
            img = img.astype(np.float32)
            if img.max() > img.min():
                img = (img - img.min()) / (img.max() - img.min())
        return img
