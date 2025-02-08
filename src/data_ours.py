import bisect
import json
import os
import random
import typing
from functools import lru_cache

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data.dataloader import default_collate
from tqdm.auto import tqdm

from torch.utils.data import DataLoader

import utils

import torch.nn as nn
import torch.nn.functional as F

def pad_to_length(x: np.ndarray, length: int, mode: str = "wrap") -> np.ndarray:
    if x.ndim != 1:
        raise ValueError("Input array must be one-dimensional.")
    
    if x.shape[0] == length:
        return x
    
    pad_size = length - x.shape[0]
        
    if pad_size > 0:
        return np.pad(x, (pad_size, 0), mode=mode)
    else:
        return x[:length]

# TODO: change into new version
class NSDDataset(Dataset):
    def __init__(
        self,
        dataset_path: os.PathLike,
        subj: typing.Literal[1, 2, 3, 4, 5, 6, 7, 8],
        split: typing.Literal["train", "test"],
        mean_three_stimuli: bool = False,
        fmri_type: typing.Literal["fsaverage", "nativesurface"] = "fsaverage",
        use_data: dict = {},
        fmri_zscore_type: typing.Literal["left/right", "all", "no"] = "all",
        brain_regions = None,
        fmri_level: int = 8,
        subdir: str = None
    ) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        if subdir is None:
            subdir = fmri_type
            if fmri_level < 8 and fmri_type == "fsaverage":
                subdir = f"{subdir}{fmri_level - 1}"
        self.fmri_level = fmri_level
        self.fmri_path = os.path.join(dataset_path, "tfMRI", subdir, f"subj{subj:02}")
        self.image_path = os.path.join(dataset_path, "stimuli", "imgs")
        self.coco_text_path = os.path.join(dataset_path, "stimuli", "captions")
        self.subj = subj
        self.split = split
        self.mean_three_stimuli = mean_three_stimuli
        self.fmri_type = fmri_type
        self.use_data = use_data
        self.brain_regions = brain_regions
        self.get_zscore_mean_std(fmri_zscore_type)
        self._read_all_data()
        # if self.split == "train":
        #     self.all_data = self.all_data[:8]
        self.negative_left_fmri = None
        self.negative_right_fmri = None
        self._pad_fmri = False
        
    def _read_all_data(self) -> None:
        split_json_path = os.path.join(self.dataset_path, "split", f"subj{self.subj:02}.json")
        with open(split_json_path, "r") as file:
            split_dict = json.load(file)[self.split]
        all_data = []
        for image_id, fmri_ids in split_dict.items():
            if self.mean_three_stimuli:
                all_data.append({"image_id": int(image_id), "fmri_id": fmri_ids})
            else:
                for fmri_id in fmri_ids:
                    all_data.append({"image_id": int(image_id), "fmri_id": fmri_id})
        self.all_data = all_data
    
    def __len__(self) -> int:
        return len(self.all_data)
    
    def get_zscore_mean_std(self, fmri_zscore_type: typing.Literal["left/right", "all", "no"] = "all") -> None:
        self.no_zscore = False
        if fmri_zscore_type == "no":
            self.no_zscore = True
            return
        with open(os.path.join(self.fmri_path, "stats.json"), "r") as file:
            stats = json.load(file)
        total_sum, total_count, total_sq_sum = 0, 0, 0
        if self.brain_regions:
            for brain_region in self.brain_regions:
                counts = stats[brain_region]["all"]["num_voxels"]
                means = stats[brain_region]["all"]["mean"]
                variances = stats[brain_region]["all"]["var"]
                total_sum += counts * means
                total_sq_sum += counts * (variances + means ** 2)
                total_count += counts
            self._left_mean = self._right_mean = float(total_sum / total_count)
            self._left_std = self._right_std = float(np.sqrt((total_sq_sum / total_count) - self._left_mean ** 2))
        else:
            if fmri_zscore_type == "all":
                self._left_mean = self._right_mean = float(stats["all"]["all"]["mean"])
                self._left_std = self._right_std = float(stats["all"]["all"]["std"])
            elif fmri_zscore_type == "left/right":
                self._left_mean = float(stats["all"]["left"]["mean"])
                self._right_mean = float(stats["all"]["right"]["mean"])
                self._left_std = float(stats["all"]["left"]["std"])
                self._right_std = float(stats["all"]["right"]["std"])
    
    def load_fmri_one_file(self, path: os.PathLike, loc: typing.Literal["left", "right"]) -> np.ndarray:
        fmri = np.load(path).astype(np.float32)
        if self.no_zscore:
            return fmri
        if loc == "left":
            fmri = (fmri - self._left_mean) / self._left_std
        elif loc == "right":
            fmri = (fmri - self._right_mean) / self._right_std
        return fmri
    
    def load_fmri_one_sample(self, index: int) -> typing.Tuple[np.ndarray]:
        fmri_ids = self.all_data[index]["fmri_id"]
        if isinstance(fmri_ids, list):
            left_fmris = []
            right_fmris = []
            for fmri_id in fmri_ids:
                left_fmri = self.load_fmri_one_file(os.path.join(self.fmri_path, f"lh.{fmri_id}.npy"), loc="left")
                left_fmris.append(left_fmri)
                right_fmri = self.load_fmri_one_file(os.path.join(self.fmri_path, f"rh.{fmri_id}.npy"), loc="right")
                right_fmris.append(right_fmri)
            return np.mean(left_fmris, axis=0), np.mean(right_fmris, axis=0)
        else:
            left_fmri = self.load_fmri_one_file(os.path.join(self.fmri_path, f"lh.{fmri_ids}.npy"), loc="left")
            right_fmri = self.load_fmri_one_file(os.path.join(self.fmri_path, f"rh.{fmri_ids}.npy"), loc="right")
            return left_fmri, right_fmri
    
    def load_image_one_sample(self, index: int) -> np.ndarray:
        image_id = self.all_data[index]["image_id"]
        image = np.load(os.path.join(self.image_path, f"{image_id}.npy"))
        return image
    
    def load_coco_text_one_sample(self, index: int) -> str:
        image_id = self.all_data[index]["image_id"]
        with open(os.path.join(self.coco_text_path, f"{image_id}.json"), "r") as file:
            texts = json.load(file)
        return random.choice(texts)
        
    def __getitem__(self, index: int) -> typing.Dict[str, np.ndarray]:
        return_dict = dict(index=index)
        if self.use_data.get("fmri", True):
            left_fmri, right_fmri = self.load_fmri_one_sample(index)
            if self.brain_regions:
                left_mask, right_mask = self.get_mask(self.fmri_level, self.brain_regions)
                left_fmri, right_fmri = left_fmri[left_mask[:, 0]], right_fmri[right_mask[:, 0]]
                return_dict["fmri"] = np.concatenate([left_fmri, right_fmri], axis=0)
                if self._pad_fmri:
                    return_dict["fmri"] = pad_to_length(return_dict["fmri"], self.num_voxels)
            else:
                return_dict["left_fmri"] = left_fmri
                return_dict["right_fmri"] = right_fmri
            if self.use_data.get("negative_fmri", True):
                if self.negative_left_fmri is None:
                    return_dict["negative_left_fmri"] = np.zeros_like(left_fmri)
                else:
                    return_dict["negative_left_fmri"] = self.negative_left_fmri
                if self.negative_right_fmri is None:
                    return_dict["negative_right_fmri"] = np.zeros_like(right_fmri)
                else:
                    return_dict["negative_right_fmri"] = self.negative_right_fmri
        if self.use_data.get("image", True):
            return_dict["image"] = self.load_image_one_sample(index)
        if self.use_data.get("text", False):
            return_dict["text"] = self.load_coco_text_one_sample(index)
        return return_dict
    
    def get_mask(
        self, 
        level: typing.Literal[1, 2, 3, 4, 5, 6, 7, 8], 
        brain_regions: typing.List[str] | str
    ) -> torch.Tensor:
        if level == 8:
            subdir = "fsaverage"
        else:
            subdir = f"fsaverage{level - 1}"
        
        num_voxels = 40962
        if brain_regions == "nsdgeneral":
            left_mask = np.load(os.path.join(self.dataset_path, "label", f"{num_voxels}", "lh.nsdgeneral.npy"))
            right_mask = np.load(os.path.join(self.dataset_path, "label", f"{num_voxels}", "rh.nsdgeneral.npy"))
            left_mask = left_mask.astype(np.bool)
            right_mask = right_mask.astype(np.bool)
        else:
            if self.fmri_type == "fsaverage":
                parcellation_name_to_idx_dict_path = os.path.join(
                    self.dataset_path, "label", "fsaverage", "HCP_MMP1.name_to_idx_dict.json"
                )
            with open(parcellation_name_to_idx_dict_path, "r") as file:
                parcellation_name_to_idx_dict = json.load(file)
            labels = [parcellation_name_to_idx_dict[brain_region] for brain_region in brain_regions]
            parcellation_path = os.path.join(self.dataset_path, "label", subdir)
            left_parcellation = np.load(os.path.join(parcellation_path, "lh.HCP_MMP1.npy"))
            right_parcellation = np.load(os.path.join(parcellation_path, "rh.HCP_MMP1.npy"))
            left_mask = torch.from_numpy(np.isin(left_parcellation, labels))
            right_mask = torch.from_numpy(np.isin(right_parcellation, labels))
        return left_mask, right_mask
    
    def set_patch_size(self, patch_size: int):
        if not self.brain_regions:
            raise ValueError("`brain_regions` is `None`.")
        left_mask, right_mask = self.get_mask(self.fmri_level, self.brain_regions)
        self.num_voxels = (left_mask.sum() + right_mask.sum()).item()
        remainder = self.num_voxels % patch_size
        if remainder != 0:
            self.num_voxels = self.num_voxels + patch_size - remainder
        self._pad_fmri = True
        
class NSDDatasetV2(Dataset):
    def __init__(
        self,
        dataset_path: os.PathLike,
        subj: typing.Literal[1, 2, 3, 4, 5, 6, 7, 8],
        split: typing.Literal["train", "val", "test"],
        mean_three_stimuli: bool = False,
        use_data: dict = {},
        nsdgeneral: bool = False,
        fmri_level: int = 7,
        pool_num=None, pool_type=None  # mindbridge's
    ) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.fmri_level = fmri_level
        self.subj = subj
        self.split = split
        self.mean_three_stimuli = mean_three_stimuli
        self.use_data = use_data
        self.nsdgeneral = nsdgeneral
        
        self.pool_num = pool_num
        self.pool_type = pool_type
        
        if fmri_level == 7:
            subdir = "sphere40962"
        else:
            raise ValueError(f"No fmri_level {fmri_level}")
        
        self.fmri_path = os.path.join(dataset_path, "tfMRI", subdir, f"subj{subj:02}")
        self.image_path = os.path.join(dataset_path, "stimuli", "imgs")
        self.coco_text_path = os.path.join(dataset_path, "stimuli", "captions")
        self.features_path = os.path.join(dataset_path, "stimuli", "features")
        self.smri_path = os.path.join(dataset_path, "surf", f"subj{subj:02}")
        
        self._read_all_data()
        
        # self.all_data = self.all_data[:32]  # DEBUG Only
        
        self._pad_fmri = False
        
    def _read_all_data(self) -> None:
        split_json_path = os.path.join(self.dataset_path, "split", f"subj{self.subj:02}.json")
        with open(split_json_path, "r") as file:
            split_dict = json.load(file)[self.split]
        all_data = []
        for image_id, fmri_ids in split_dict.items():
            if self.mean_three_stimuli:
                all_data.append({"image_id": int(image_id), "fmri_id": fmri_ids})
            else:
                for fmri_id in fmri_ids:
                    all_data.append({"image_id": int(image_id), "fmri_id": fmri_id})
        self.all_data = all_data
        
        if self.use_data.get("ViT_H_14_img", False):
            self.ViT_H_14_img = np.load(os.path.join(self.features_path, "ViT-H-14.img.npy"))
        
        if self.use_data.get("ViT_H_14_txt", False):
            self.ViT_H_14_txt = np.load(os.path.join(self.features_path, "ViT-H-14.txt.npy"))
        
        if self.use_data.get("structure", False):
            
            num_voxels = 40962
            
            def _get_smri(semi_brain):
                area = np.load(os.path.join(self.smri_path, f"area.{num_voxels}.{semi_brain}.npy"))
                area = (area - area.mean()) / area.std()
                curv = np.load(os.path.join(self.smri_path, f"curv.{num_voxels}.{semi_brain}.npy"))
                curv = (curv - curv.mean()) / curv.std()
                sulc = np.load(os.path.join(self.smri_path, f"sulc.{num_voxels}.{semi_brain}.npy"))
                sulc = (sulc - sulc.mean()) / sulc.std()
                thic = np.load(os.path.join(self.smri_path, f"thickness.{num_voxels}.{semi_brain}.npy"))
                thic = (thic - thic.mean()) / thic.std()
                smri = np.concatenate([area, curv, sulc, thic], axis=1)
                return smri
            
            self.left_smri = _get_smri("lh")
            self.right_smri = _get_smri("rh")
    
    def __len__(self) -> int:
        return len(self.all_data)
    
    def load_fmri_one_file(self, path: os.PathLike) -> np.ndarray:
        fmri = np.load(path).astype(np.float32)
        return fmri
    
    def load_fmri_one_sample(self, index: int) -> typing.Tuple[np.ndarray, ...]:
        fmri_ids = self.all_data[index]["fmri_id"]
        if isinstance(fmri_ids, list):
            left_fmris = []
            right_fmris = []
            for fmri_id in fmri_ids:
                left_fmri = self.load_fmri_one_file(os.path.join(self.fmri_path, f"lh.{fmri_id}.npy"))
                left_fmris.append(left_fmri)
                right_fmri = self.load_fmri_one_file(os.path.join(self.fmri_path, f"rh.{fmri_id}.npy"))
                right_fmris.append(right_fmri)
            return np.stack(left_fmris, axis=0), np.stack(right_fmris, axis=0)
        else:
            left_fmri = self.load_fmri_one_file(os.path.join(self.fmri_path, f"lh.{fmri_ids}.npy"))
            right_fmri = self.load_fmri_one_file(os.path.join(self.fmri_path, f"rh.{fmri_ids}.npy"))
            return left_fmri, right_fmri
    
    def load_image_one_sample(self, index: int) -> np.ndarray:
        image_id = self.all_data[index]["image_id"]
        image = np.load(os.path.join(self.image_path, f"{image_id}.npy"))
        return image
    
    def load_coco_text_one_sample(self, index: int) -> str:
        image_id = self.all_data[index]["image_id"]
        with open(os.path.join(self.coco_text_path, f"{image_id}.json"), "r") as file:
            texts = json.load(file)
        return random.choice(texts)
    
    def load_ViT_H_14_img_one_sample(self, index) -> np.ndarray:
        image_id = self.all_data[index]["image_id"]
        return self.ViT_H_14_img[image_id]
    
    def load_ViT_H_14_txt_one_sample(self, index) -> np.ndarray:
        image_id = self.all_data[index]["image_id"]
        return self.ViT_H_14_txt[image_id]
        
    def __getitem__(self, index: int) -> typing.Dict[str, np.ndarray]:
        return_dict = dict(index=index, subj=self.subj)
        if self.use_data.get("fmri", True):
            
            left_fmri, right_fmri = self.load_fmri_one_sample(index)
            
            if self.nsdgeneral:
                left_mask, right_mask = self.get_mask(self.fmri_level)
                left_fmri, right_fmri = left_fmri[:, left_mask[:, 0]], right_fmri[:, right_mask[:, 0]]
                return_dict["fmri"] = np.concatenate([left_fmri, right_fmri], axis=1)
                if self._pad_fmri:
                    return_dict["fmri"] = pad_to_length(return_dict["fmri"], self.num_voxels)
                return_dict["fmri"] = torch.from_numpy(return_dict["fmri"])
                return_dict["fmri"] = pool_voxels(return_dict["fmri"], self.pool_num, self.pool_type)
            else:
                return_dict["left_fmri"] = left_fmri
                return_dict["right_fmri"] = right_fmri

        if self.use_data.get("image", True):
            return_dict["image"] = self.load_image_one_sample(index)
            return_dict["image"] = return_dict["image"].astype(np.float32) / 255.0
            return_dict["image"] = torch.from_numpy(return_dict["image"].transpose(2, 0, 1))
        
        if self.use_data.get("text", False):
            return_dict["text"] = self.load_coco_text_one_sample(index)
        
        if self.use_data.get("ViT_H_14_img", False):
            return_dict["ViT_H_14_img"] = self.load_ViT_H_14_img_one_sample(index)
        
        if self.use_data.get("ViT_H_14_txt", False):
            return_dict["ViT_H_14_txt"] = self.load_ViT_H_14_txt_one_sample(index)
        
        if self.use_data.get("structure", False):
            return_dict["left_smri"] = self.left_smri
            return_dict["right_smri"] = self.right_smri
        
        return return_dict
    
    def get_mask(self, level: typing.Literal[1, 2, 3, 4, 5, 6, 7, 8]) -> typing.Tuple[np.ndarray, ...]:
        num_voxels = 40962
        left_mask = np.load(os.path.join(self.dataset_path, "label", f"{num_voxels}", "lh.nsdgeneral.npy"))
        right_mask = np.load(os.path.join(self.dataset_path, "label", f"{num_voxels}", "rh.nsdgeneral.npy"))
        left_mask = left_mask.astype(np.bool_)
        right_mask = right_mask.astype(np.bool_)
        return left_mask, right_mask
    
    def set_patch_size(self, patch_size: int):
        left_mask, right_mask = self.get_mask(self.fmri_level)
        self.num_voxels = int(left_mask.sum() + right_mask.sum())
        remainder = self.num_voxels % patch_size
        if remainder != 0:
            self.num_voxels = self.num_voxels + patch_size - remainder
        self._pad_fmri = True
        
class NSDDatasetV3(Dataset):
    def __init__(
        self,
        dataset_path: os.PathLike,
        subj: typing.Literal[1, 2, 3, 4, 5, 6, 7, 8],
        split: typing.Literal["train", "val", "test"],
        mean_three_stimuli: bool = False,
        use_data: dict = {},
        nsdgeneral: bool = False,
        fmri_level: int = 7,
        alpha: float = 0.0
    ) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.fmri_level = fmri_level
        self.subj = subj
        self.split = split
        self.mean_three_stimuli = mean_three_stimuli
        self.use_data = use_data
        self.nsdgeneral = nsdgeneral
        self.alpha = alpha
        
        if fmri_level == 7:
            subdir = "sphere40962"
        else:
            raise ValueError(f"No fmri_level {fmri_level}")
        
        self.fmri_path = os.path.join(dataset_path, "tfMRI", subdir, f"subj{subj:02}")
        self.image_path = os.path.join(dataset_path, "stimuli", "imgs")
        self.coco_text_path = os.path.join(dataset_path, "stimuli", "captions")
        self.features_path = os.path.join(dataset_path, "stimuli", "features")
        self.smri_path = os.path.join(dataset_path, "surf", f"subj{subj:02}")
        
        self._read_all_data()
        
        # self.all_data = self.all_data[:32]  # DEBUG Only
        
        self._pad_fmri = False
        
    def _read_all_data(self) -> None:
        split_json_path = os.path.join(self.dataset_path, "split", f"subj{self.subj:02}.json")
        with open(split_json_path, "r") as file:
            all_data = json.load(file)[self.split]
        
        self.all_data = []
        for image_id, fmri_id in all_data.items():
            self.all_data.append({"image_id": image_id, "fmri_id": fmri_id})
        
        if self.use_data.get("ViT_H_14_img", False):
            self.ViT_H_14_img = np.load(os.path.join(self.features_path, "ViT-H-14.img.npy"))
        
        if self.use_data.get("ViT_H_14_txt", False):
            self.ViT_H_14_txt = np.load(os.path.join(self.features_path, "ViT-H-14.txt.npy"))
        
        if self.use_data.get("structure", False):
            
            num_voxels = 40962
            
            def _get_smri(semi_brain):
                area = np.load(os.path.join(self.smri_path, f"area.{num_voxels}.{semi_brain}.npy"))
                area = (area - area.mean()) / area.std()
                curv = np.load(os.path.join(self.smri_path, f"curv.{num_voxels}.{semi_brain}.npy"))
                curv = (curv - curv.mean()) / curv.std()
                sulc = np.load(os.path.join(self.smri_path, f"sulc.{num_voxels}.{semi_brain}.npy"))
                sulc = (sulc - sulc.mean()) / sulc.std()
                thic = np.load(os.path.join(self.smri_path, f"thickness.{num_voxels}.{semi_brain}.npy"))
                thic = (thic - thic.mean()) / thic.std()
                smri = np.concatenate([area, curv, sulc, thic], axis=1)
                return smri
            
            self.left_smri = _get_smri("lh")
            self.right_smri = _get_smri("rh")
    
    def __len__(self) -> int:
        return len(self.all_data)
    
    def load_fmri_one_file(self, path: os.PathLike) -> np.ndarray:
        fmri = np.load(path).astype(np.float32)
        return fmri
    
    def load_fmri_one_sample(self, index: int) -> typing.Tuple[np.ndarray, ...]:
        fmri_ids = self.all_data[index]["fmri_id"]
        left_fmris = []
        right_fmris = []
        for fmri_id in fmri_ids:
            left_fmri = self.load_fmri_one_file(os.path.join(self.fmri_path, f"lh.{fmri_id}.npy"))
            left_fmris.append(left_fmri)
            right_fmri = self.load_fmri_one_file(os.path.join(self.fmri_path, f"rh.{fmri_id}.npy"))
            right_fmris.append(right_fmri)
        return np.stack(left_fmris, axis=0), np.stack(right_fmris, axis=0)
    
    def load_image_one_sample(self, index: int) -> np.ndarray:
        image_id = self.all_data[index]["image_id"]
        image = np.load(os.path.join(self.image_path, f"{image_id}.npy"))
        return image
    
    def load_coco_text_one_sample(self, index: int) -> str:
        image_id = self.all_data[index]["image_id"]
        with open(os.path.join(self.coco_text_path, f"{image_id}.json"), "r") as file:
            texts = json.load(file)
        return random.choice(texts)
    
    def load_ViT_H_14_img_one_sample(self, index) -> np.ndarray:
        image_id = self.all_data[index]["image_id"]
        return self.ViT_H_14_img[image_id]
    
    def load_ViT_H_14_txt_one_sample(self, index) -> np.ndarray:
        image_id = self.all_data[index]["image_id"]
        return self.ViT_H_14_txt[image_id]
        
    def __getitem__(self, index: int) -> typing.Dict[str, np.ndarray]:
        return_dict = dict(index=index, subj=self.subj)
        if self.use_data.get("fmri", True):
            
            left_fmris, right_fmris = self.load_fmri_one_sample(index)
            
            if self.nsdgeneral:
                left_mask, right_mask = self.get_mask(self.fmri_level)
                left_fmris, right_fmris = left_fmris[:, left_mask[:, 0]], right_fmris[:, right_mask[:, 0]]
                return_dict["fmri"] = np.concatenate([left_fmris, right_fmris], axis=1)
                if self._pad_fmri:
                    padded = []
                    for i in range(return_dict["fmri"].shape[0]):
                        padded.append(pad_to_length(return_dict["fmri"][i, :], self.num_voxels))
                    return_dict["fmri"] = np.stack(padded, axis=0)
                    return_dict["fmri_num"] = return_dict["fmri"].shape[0]
            else:
                return_dict["left_fmri"] = left_fmris
                return_dict["right_fmri"] = right_fmris
                return_dict["fmri_num"] = left_fmris.shape[0]
            

        if self.use_data.get("image", True):
            return_dict["image"] = self.load_image_one_sample(index)
        
        if self.use_data.get("text", False):
            return_dict["text"] = self.load_coco_text_one_sample(index)
        
        if self.use_data.get("ViT_H_14_img", False):
            return_dict["ViT_H_14_img"] = self.load_ViT_H_14_img_one_sample(index)
        
        if self.use_data.get("ViT_H_14_txt", False):
            return_dict["ViT_H_14_txt"] = self.load_ViT_H_14_txt_one_sample(index)
        
        if self.use_data.get("structure", False):
            return_dict["left_smri"] = self.left_smri
            return_dict["right_smri"] = self.right_smri
        
        return return_dict
    
    def get_mask(self, level: typing.Literal[1, 2, 3, 4, 5, 6, 7, 8]) -> typing.Tuple[np.ndarray, ...]:
        num_voxels = 40962
        left_mask = np.load(os.path.join(self.dataset_path, "label", f"{num_voxels}", "lh.nsdgeneral.npy"))
        right_mask = np.load(os.path.join(self.dataset_path, "label", f"{num_voxels}", "rh.nsdgeneral.npy"))
        left_mask = left_mask.astype(np.bool_)
        right_mask = right_mask.astype(np.bool_)
        return left_mask, right_mask
    
    def set_patch_size(self, patch_size: int):
        left_mask, right_mask = self.get_mask(self.fmri_level)
        self.num_voxels = int(left_mask.sum() + right_mask.sum())
        remainder = self.num_voxels % patch_size
        if remainder != 0:
            self.num_voxels = self.num_voxels + patch_size - remainder
        self._pad_fmri = True

def get_collate_fn(
    mixup_alpha: float, 
    mixup_output_size: int = 3,
    mixup_fn: str = "linear_random",
    nomixup_fn: str = "default",
    train_mode: bool = True
):
    
    def mixup_linear_random(data: torch.Tensor) -> torch.Tensor:
        a = np.ones(data.shape[0])
        weights = torch.from_numpy(np.random.dirichlet(a, size=mixup_output_size))
        data = weights.to(data.dtype) @ data
        return data
    
    def nomixup_mean(data: torch.Tensor) -> torch.Tensor:
        data = torch.mean(data, dim=0)
        data = data.unsqueeze(0)
        return data
    
    def nomixup_default(data: torch.Tensor) -> torch.Tensor:
        return data
    
    def test_time_linear_mean(data: torch.Tensor) -> torch.Tensor:
        data = torch.mean(data, dim=0)
        data = data.unsqueeze(0)
        return data
    
    mixup_fn_map = {
        "linear_random": mixup_linear_random
    }
    nomixup_fn_map = {
        "mean": nomixup_mean,
        "default": nomixup_default
    }
    test_time_fn_map = {
        "linear_random": test_time_linear_mean
    }
    
    assert mixup_fn in mixup_fn_map.keys(), f"'mixup_fn' should in: {mixup_fn_map.keys()}"
    assert nomixup_fn in nomixup_fn_map.keys(), f"'nomixup_fn' should in: {nomixup_fn_map.keys()}"
    
    test_time_fn = test_time_fn_map[mixup_fn]
    mixup_fn = mixup_fn_map[mixup_fn]
    nomixup_fn = nomixup_fn_map[nomixup_fn]

    def collate_fn(batch: typing.List[typing.Dict]):
        
        batch_size = len(batch)
        mixup_index = np.random.rand(batch_size) < mixup_alpha
        keys = batch[0].keys()
        
        collated_batch = {}

        # for key in keys:
            
        #     collated_batch[key] = default_collate([
        #         tensor for sample in [
        #             mixup_fn(torch.from_numpy(sample[key])) if is_mixup else torch.from_numpy(sample[key])
        #             for is_mixup, sample in zip(mixup_index, batch)
        #             ] for tensor in torch.unbind(sample, dim=0)
        #         ] if key in ["left_fmri", "right_fmri", "fmri"] else [
        #         sample[key] for is_mixup, sample in zip(mixup_index, batch)
        #         for _ in range(
        #             mixup_output_size if is_mixup else (
        #                 1 if nomixup_fn == nomixup_mean else sample["fmri_num"]
        #         ))]
        #     )
        collated_batch = {
            key: default_collate([
            tensor for sample in [(
                    mixup_fn(torch.from_numpy(sample[key])) if is_mixup 
                    else nomixup_fn(torch.from_numpy(sample[key])))
                if train_mode else (test_time_fn(torch.from_numpy(sample[key])))
                for is_mixup, sample in zip(mixup_index, batch)
                ] for tensor in torch.unbind(sample, dim=0)
            ] if key in ["left_fmri", "right_fmri", "fmri"] else [
            sample[key] for is_mixup, sample in zip(mixup_index, batch)
            for _ in range(
                (mixup_output_size if is_mixup else (
                    1 if nomixup_fn == nomixup_mean else sample["fmri_num"]
            )) if train_mode else 1)]
        ) for key in keys}
        
        collated_batch.pop("fmri_num")
        
        return collated_batch
    
    return collate_fn


class ConcatNSDDataset(Dataset):

    datasets: typing.List[NSDDatasetV2]
    cumulative_sizes: typing.List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: typing.Iterable[Dataset]) -> None:
        super().__init__()
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, "datasets should not be an empty iterable"  # type: ignore[arg-type]
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]
    
    def get_mask(self, level: typing.Literal[1, 2, 3, 4, 5, 6, 7, 8]) -> typing.Tuple[np.ndarray, ...]:
        return self.datasets[0].get_mask(level)
    
    def set_patch_size(self, patch_size: int):
        for dataset in self.datasets:
            dataset.set_patch_size(patch_size)

def get_nsd_dataset(
    dataset_path: os.PathLike,
    subjs: typing.List[int],
    split: typing.Literal["train", "val", "test"],
    mean_three_stimuli: bool = False,
    use_data: dict = {},
    nsdgeneral: bool = False,
    fmri_level: int = 7,
    version: int = 2
) -> ConcatNSDDataset:
    
    if isinstance(subjs, int):
        subjs = [subjs]
    
    nsd_dataset_cls = NSDDatasetV2 if version == 2 else NSDDatasetV3
    
    datasets = []
    for subj in subjs:
        datasets.append(nsd_dataset_cls(
            dataset_path=dataset_path,
            subj=subj,
            split=split,
            mean_three_stimuli=mean_three_stimuli,
            use_data=use_data,
            nsdgeneral=nsdgeneral,
            fmri_level=fmri_level
        ))
    
    return ConcatNSDDataset(datasets)

class EcphoryMemory:
    
    def __init__(
        self, 
        dataset_path: os.PathLike, 
        cls_key: bool = True, 
        base_dtype: torch.dtype = torch.float32
    ) -> None:
        
        self.dataset_path = dataset_path
        
        if not cls_key:
            base_dtype = torch.float16
        
        if cls_key:
            self.image_feats = np.load(os.path.join(dataset_path, "stimuli", "clip_embeddings", "image.cls.npy"))
            self.image_feats = torch.from_numpy(self.image_feats).to(base_dtype)
            self.image_feats = torch.nn.functional.normalize(self.image_feats, dim=-1)
            self.text_feats = np.load(os.path.join(dataset_path, "stimuli", "clip_embeddings", "text.cls.npy"))
            self.text_feats = torch.from_numpy(self.text_feats).to(base_dtype)
            self.text_feats = torch.nn.functional.normalize(self.text_feats, dim=-1)
        else:
            print("reading images")
            self.image_feats = np.load(os.path.join(dataset_path, "stimuli", "clip_embeddings", "image.npy"))
            self.image_feats = torch.from_numpy(self.image_feats).to(base_dtype)
            self.image_feats = self.image_feats.reshape(73000, 257 * 768)
            self.image_feats = torch.nn.functional.normalize(self.image_feats, dim=-1)
            print("reading texts")
            self.text_feats = np.load(os.path.join(dataset_path, "stimuli", "clip_embeddings", "text.npy"))
            self.text_feats = torch.from_numpy(self.text_feats).to(base_dtype)
            self.text_feats = self.text_feats.reshape(73000, 77 * 768)
            self.text_feats = torch.nn.functional.normalize(self.text_feats, dim=-1)
        
        print(self.image_feats.shape)
        print(self.text_feats.shape)
        
        self.image_ids = dict()
        for subj_id in range(1, 9):
            subj = f"subj0{subj_id}"
            with open(os.path.join(dataset_path, "split", f"{subj}.json"), "r") as file:
                subj_split = json.load(file)
                subj_train_images = subj_split["train"].keys()
                subj_train_images = [int(image_id) for image_id in subj_train_images]
                self.image_ids[str(subj_id)] = torch.tensor(subj_train_images, dtype=torch.int64)
        
        self.cls_key = cls_key
        self.base_dtype = base_dtype
    
    @lru_cache(maxsize=1000)
    def load_image_embed(self, index: int) -> torch.Tensor:
        if self.cls_key:
            embed = np.load(os.path.join(self.dataset_path, "stimuli", "clip_embeddings", "image", f"{index}.npy"))
            embed = torch.from_numpy(embed).to(self.base_dtype)
            return embed
        else:
            embed = self.image_feats[index, :].reshape(257, 768)
            return embed
    
    @lru_cache(maxsize=1000)
    def load_text_embed(self, index: int) -> torch.Tensor:
        if self.cls_key:
            embed = np.load(os.path.join(self.dataset_path, "stimuli", "clip_embeddings", "text", f"{index}.npy"))
            embed = torch.from_numpy(embed).to(self.base_dtype)
            return embed
        else:
            embed = self.text_feats[index, :].reshape(77, 768)
            return embed
                
    def find(
        self, 
        subj_id: int, 
        image_cls: torch.Tensor, 
        text_cls: torch.Tensor
    ) -> typing.Tuple[torch.Tensor, ...]:
        
        orig_device = image_cls.device
        orig_dtype = image_cls.dtype
        image_cls = image_cls.to(device="cpu", dtype=self.base_dtype)
        text_cls = text_cls.to(device="cpu", dtype=self.base_dtype)
        
        image_ids = self.image_ids[str(subj_id)]
        image_memories = self.image_feats[image_ids, :]
        text_memories = self.text_feats[image_ids, :]
        
        image_cls = torch.nn.functional.normalize(image_cls, dim=-1)
        text_cls = torch.nn.functional.normalize(text_cls, dim=-1)
        
        image_similarity = torch.mm(image_cls, image_memories.t())
        text_similarity = torch.mm(text_cls, text_memories.t())
        
        image_indices = torch.argmax(image_similarity, dim=1).tolist()
        text_indices = torch.argmax(text_similarity, dim=1).tolist()
        
        ret_image_emb = torch.stack([self.load_image_embed(image_ids[index]) for index in image_indices], dim=0)
        ret_text_emb = torch.stack([self.load_text_embed(image_ids[index]) for index in text_indices], dim=0)
        ret_image_emb = ret_image_emb.to(device=orig_device, dtype=orig_dtype)
        ret_text_emb = ret_text_emb.to(device=orig_device, dtype=orig_dtype)
        
        return ret_image_emb, ret_text_emb
    
def pool_voxels(voxels, pool_num, pool_type):
    voxels = voxels.float()
    if pool_type == 'avg':
        voxels = nn.AdaptiveAvgPool1d(pool_num)(voxels)
    elif pool_type == 'max':
        voxels = nn.AdaptiveMaxPool1d(pool_num)(voxels)
    elif pool_type == "resize":
        voxels = voxels.unsqueeze(1) # Add a dimension to make it (B, 1, L)
        voxels = F.interpolate(voxels, size=pool_num, mode='linear', align_corners=False)
        voxels = voxels.squeeze(1)

    return voxels
    
def get_dataloader(
        root_dir,
        batch_size,
        num_workers=1,
        seed=42,
        is_shuffle=True,
        extensions=['nsdgeneral.npy', "jpg", 'coco73k.npy', "subj"],
        pool_type=None,
        pool_num=None,
        length=None,
        subj=None,
        split=None,
    ):
    utils.seed_everything(seed)
    
    dataset = NSDDatasetV2(
        dataset_path="/home/yusijin/datasets/nsd/preprocess",  # replace to yours
        subj=subj,
        split=split,
        use_data={"image": True, "text": True},
        nsdgeneral=True,
        mean_three_stimuli=True,
        pool_num=pool_num, pool_type=pool_type  # mindbridge's
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=is_shuffle)

    return dataloader

def get_dls(subject, data_path, batch_size, val_batch_size, num_workers, pool_type, pool_num, length, seed):
    train_path = "{}/webdataset_avg_split/train/subj0{}".format(data_path, subject)
    val_path = "{}/webdataset_avg_split/val/subj0{}".format(data_path, subject)
    extensions = ['nsdgeneral.npy', "jpg", 'coco73k.npy', "subj"]

    train_dl = get_dataloader(
        train_path,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        extensions=extensions,
        pool_type=pool_type,
        pool_num=pool_num,
        is_shuffle=True,
        length=length,
        subj=subject,
        split="train"
    )

    val_dl = get_dataloader(
        val_path,
        batch_size=val_batch_size,
        num_workers=num_workers,
        seed=seed,
        extensions=extensions,
        pool_type=pool_type,
        pool_num=pool_num,
        is_shuffle=False,
        subj=subject,
        split="val"
    )

    num_train=len(train_dl.dataset)
    num_val=len(val_dl.dataset)
    print(train_path,"\n",val_path)
    print("number of train data:", num_train)
    print("batch_size", batch_size)
    print("number of val data:", num_val)
    print("val_batch_size", val_batch_size)

    return train_dl, val_dl

import kornia
from kornia.augmentation.container import AugmentationSequential

img_augment = AugmentationSequential(
    kornia.augmentation.RandomResizedCrop((224,224), (0.8,1), p=0.3),
    kornia.augmentation.Resize((224, 224)),
    kornia.augmentation.RandomBrightness(brightness=(0.8, 1.2), clip_output=True, p=0.2),
    kornia.augmentation.RandomContrast(contrast=(0.8, 1.2), clip_output=True, p=0.2),
    kornia.augmentation.RandomGamma((0.8, 1.2), (1.0, 1.3), p=0.2),
    kornia.augmentation.RandomSaturation((0.8,1.2), p=0.2),
    kornia.augmentation.RandomHue((-0.1,0.1), p=0.2),
    kornia.augmentation.RandomSharpness((0.8, 1.2), p=0.2),
    kornia.augmentation.RandomGrayscale(p=0.2),
    data_keys=["input"],
)

