# data/processed_dataset.py
import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class ViewData:
    depth: str
    point_map: str
    feature: str
    confidence: str
    extrinsic: Optional[str] = None
    intrinsic: Optional[str] = None


class VGGTProcessedDataset(Dataset):
    """Dataset that loads pre-computed VGGT outputs.

    The preprocessing step is expected to produce a directory structure::

        root/scene_i/depth/<frame>_depth.npy
        root/scene_i/point_map/<frame>_points.npy
        root/scene_i/features/<frame>_feat.npy
        root/scene_i/confidence/<frame>_conf.npy
        root/scene_i/extrinsic/<frame>.npy (optional)
        root/scene_i/intrinsic/<frame>.npy (optional)

    Each call to ``__getitem__`` returns a dictionary containing tensors for
    these entries.
    """

    def __init__(self, root: str):
        self.root = root
        self.items: List[ViewData] = []
        for scene_dir in sorted(glob.glob(os.path.join(root, "*"))):
            depth_dir = os.path.join(scene_dir, "depth")
            for depth_file in sorted(glob.glob(os.path.join(depth_dir, "*_depth.npy"))):
                base = os.path.basename(depth_file).replace("_depth.npy", "")
                item = ViewData(
                    depth=depth_file,
                    point_map=os.path.join(scene_dir, "point_map", f"{base}_points.npy"),
                    feature=os.path.join(scene_dir, "features", f"{base}_feat.npy"),
                    confidence=os.path.join(scene_dir, "confidence", f"{base}_conf.npy"),
                )
                extr = os.path.join(scene_dir, "extrinsic", f"{base}.npy")
                intr = os.path.join(scene_dir, "intrinsic", f"{base}.npy")
                if os.path.exists(extr):
                    item.extrinsic = extr
                if os.path.exists(intr):
                    item.intrinsic = intr
                self.items.append(item)
        if not self.items:
            raise RuntimeError(f"No processed data found in {root}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.items[idx]
        depth = torch.from_numpy(np.load(item.depth)).float()
        point_map = torch.from_numpy(np.load(item.point_map)).float()
        feature = torch.from_numpy(np.load(item.feature)).float()
        confidence = torch.from_numpy(np.load(item.confidence)).float()

        sample: Dict[str, torch.Tensor] = {
            "depth": depth,
            "point_map": point_map,
            "feature": feature,
            "confidence": confidence,
        }
        if item.extrinsic:
            sample["extrinsic"] = torch.from_numpy(np.load(item.extrinsic)).float()
        if item.intrinsic:
            sample["intrinsic"] = torch.from_numpy(np.load(item.intrinsic)).float()
        return sample
