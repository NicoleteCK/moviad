import glob
import os
import pandas as pd

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from moviad.datasets.vad_dataset import VADDataset
from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.utilities.configurations import Split, LabelName

CATEGORIES = (
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
)

class VISADataset(VADDataset):
    def __init__(
        self,
        dataset_arguments: DatasetArguments,
        category: str,
        split: Split | list[Split]
    ):
        super().__init__(dataset_arguments, category, split)
        self.category = category
        self.dataset_root = self.dataset_arguments.dataset_path 

        self.df = pd.read_csv(os.path.join(self.dataset_root, "split_csv", "1cls.csv"))
        split_str = split.value if hasattr(split, 'value') else split
        self.df = self.df[(self.df["object"]==category) & (self.df["split"]==split_str)]

    def __len__(self):
        return len(self.df)
    '''
    def __getitem__(self, index):
        path = os.path.join(self.dataset_root, self.df.iloc[index]["image"])
        image = self.transform_image(
            Image.open(path).convert("RGB")
        )
        label = LabelName.NORMAL if self.df.iloc[index]["label"] == "normal" else LabelName.ABNORMAL

        if self.split == Split.TRAIN:
            return image
        else:

            mask = None
            if self.df.iloc[index]["label"] == "normal":
                mask = torch.zeros((1, self.dataset_arguments.gt_mask_size[0], self.dataset_arguments.gt_mask_size[1]))
            else:
                mask_path = os.path.join(self.dataset_root, self.df.iloc[index]["mask"])
                mask = Image.open(mask_path).convert("L")
                mask = self.transform_mask(mask)

                mask = torch.where(
                    mask > 0.0, torch.ones_like(mask), torch.zeros_like(mask)
                )

            return image, label, mask, path, self.category
    '''
    def __getitem__(self, index):
        path = os.path.join(self.dataset_root, self.df.iloc[index]["image"])
        raw_image = Image.open(path).convert("RGB")
        
        label = LabelName.NORMAL if self.df.iloc[index]["label"] == "normal" else LabelName.ABNORMAL

        # 1. Recupero o generazione della maschera base in formato PIL
        if label == LabelName.NORMAL:
            # Creiamo una maschera PIL nera della dimensione corretta
            raw_mask = Image.new("L", self.dataset_arguments.gt_mask_size, 0)
        else:
            mask_path = os.path.join(self.dataset_root, self.df.iloc[index]["mask"])
            raw_mask = Image.open(mask_path).convert("L")

        # 2. Applicazione del ridimensionamento e delle trasformazioni di colore (Solo su Immagine)
        image = self.transform_image_pil(raw_image)
        mask = self.transform_mask_pil(raw_mask)

        # 3. SINCRONIZZAZIONE GEOMETRICA (Blocco del seed su PIL Image)
        state = torch.get_rng_state()
        image = self.transform_geometry(image)
        
        torch.set_rng_state(state)
        mask = self.transform_geometry(mask)

        # 4. CONVERSIONE IN TENSORI E BINARIZZAZIONE MASCHERA
        image = self.to_tensor(image)
        mask = self.to_tensor(mask)  # Tensore [1, H, W]

        # Sostituisce la logica torch.where per garantire che la maschera sia puramente binaria (0 o 1)
        mask = torch.where(mask > 0.0, torch.ones_like(mask), torch.zeros_like(mask))

        # 5. NORMALIZZAZIONE FINALE (Solo sull'immagine)
        image = self.normalize(image)

        # 6. OUTPUT IN BASE ALLO SPLIT
        if self.split == Split.TRAIN:
            return image
        else:
            return image, label, mask.int(), path, self.category

    @staticmethod
    def get_categories() -> list:
        return list(CATEGORIES)