import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision.transforms import transforms

from moviad.datasets.vad_dataset import VADDataset
from moviad.datasets.exceptions.exceptions import DatasetTooSmallToContaminateException
from moviad.utilities.configurations import Split, LabelName
from moviad.datasets.dataset_arguments import DatasetArguments

IMG_EXTENSIONS = (".png", ".PNG", ".jpg", ".JPG")

#Not real categories, just for parsing the dataset structure
CATEGORIES = (
    "cps_1", 
    "cps_2",   
    "cps_3",
    "cps_4",
    "cps_5",
    "cps_6",
)

IMG_SIZE = (3, 270, 480)

"""Create CPS-AD2D samples by parsing the CPS-AD2D data file structure.

    The files are expected to follow the structure:
        path/to/dataset/category/images/train/image_filename.jpg
        path/to/dataset/category/annotation/train/mask_filename.png

"""

class CPSAD2DDataset(VADDataset):
    """CPS-AD2D dataset class."""

    def __init__(
        self,
        dataset_arguments: DatasetArguments,
        split: Split | list[Split],
        category: str | None = None
    ) -> None:

        super().__init__(
            dataset_arguments,
            category,
            split
        )

        self.dataset_root = Path(self.dataset_arguments.dataset_path)
        self.samples: pd.DataFrame = None
        self.load_dataset()
    
    def is_loaded(self) -> bool:
        return self.samples is not None
    
    def load_dataset(self):
        if self.is_loaded():
            print("Dataset is already loaded.")
            return
        
        all_samples = []
        
        for category_path in self.dataset_root.iterdir():
            if not category_path.is_dir(): continue
            
            category_name = category_path.name
            images_path = category_path / "images" / "train"
            annotations_path = category_path / "annotations" / "train"

            if not images_path.exists(): continue

            for image_path in images_path.glob("*"):
                if image_path.suffix in IMG_EXTENSIONS:

                    mask_path = annotations_path / (image_path.stem + ".png")

                    label = LabelName.NORMAL

                    if mask_path.exists():
                        with Image.open(mask_path) as img:
                            if img.convert("L").getextrema()[1] > 0:
                                label = LabelName.ABNORMAL

                    all_samples.append({
                        "image_path": str(image_path),
                        "mask_path": str(mask_path) if mask_path.exists() else None,
                        "label": label,
                        "category": category_name,
                        "split": "train" # Default temporaneo
                    })

        df_samples = pd.DataFrame(all_samples)


        if df_samples.empty:
            raise RuntimeError(f"Nessuna immagine trovata in {self.dataset_root}")

        # --- LOGICA DI SPLIT AUTOMATICO ---
        self.samples = self._apply_custom_split(df_samples, train_ratio=0.8)

        target_split = self.split.value if hasattr(self.split, 'value') else self.split

        if target_split == "train":
            initial_count = len(self.samples)
            self.samples = self.samples[self.samples.label == LabelName.NORMAL].reset_index(drop=True)


    def _apply_custom_split(self, df: pd.DataFrame, train_ratio: float):
        # Il parametro train_ratio viene ignorato per rispettare il vincolo dei 300 campioni
        seed = 42 
        num_train_normal = 300
        
        # Separazione tra normali e anomalie
        is_normal = df['label'] == LabelName.NORMAL
        df_normal_all = df[is_normal]
        df_anomaly_all = df[~is_normal]
        
        # 1. Shuffle e selezione Training (300 campioni normali casuali)
        if len(df_normal_all) < num_train_normal:
            raise ValueError(f"Dataset insufficiente: richiesti {num_train_normal} normali, presenti {len(df_normal_all)}.")
        
        # Mischiamo le normali e prendiamo le prime 300 per il train
        df_normal_shuffled = df_normal_all.sample(frac=1, random_state=seed)
        train_indices = df_normal_shuffled.index[:num_train_normal]
        
        # 2. Selezione Test (Tutte le normali rimanenti)
        remaining_normal_indices = df_normal_shuffled.index[num_train_normal:]
        num_test_normal = len(remaining_normal_indices)
        
        # 3. Selezione Casuale Anomalie per il bilanciamento
        if len(df_anomaly_all) >= num_test_normal:
            test_anomaly_indices = df_anomaly_all.sample(n=num_test_normal, random_state=seed).index
        else:
            # Se ci sono meno anomalie delle normali rimaste, le prendiamo tutte (shuffled)
            print(f"Warning: Solo {len(df_anomaly_all)} anomalie disponibili per {num_test_normal} normali.")
            test_anomaly_indices = df_anomaly_all.sample(frac=1, random_state=seed).index
        
        # Unione degli indici per il test set
        test_indices = list(remaining_normal_indices) + list(test_anomaly_indices)

        # Marcatura nel DataFrame originale
        df['split'] = 'excluded' 
        df.loc[train_indices, 'split'] = "train"
        df.loc[test_indices, 'split'] = "test"

        # Selezione del subset richiesto (train o test)
        target_split = self.split.value if hasattr(self.split, 'value') else self.split
        return df[df['split'] == target_split].reset_index(drop=True)
    
    def __getitem__(self, index: int):
        """
        Args:
            index (int): indice dell'elemento da restituire
        Returns:
            TRAIN: image, label (e opzionalmente mask)
            TEST: image, label, mask, path
        """
        if self.samples is None:
            self.load_dataset()

        sample = self.samples.iloc[index]


        image = self.transform_image(
            Image.open(self.samples.iloc[index].image_path).convert("RGB")
        )

        label = sample.label
        path = str(sample.image_path)


        if label == LabelName.NORMAL or sample.mask_path is None:
            mask = torch.zeros(1, image.shape[1], image.shape[2])
        else:
            mask = Image.open(sample.mask_path).convert("L")
            mask = self.transform_mask(mask)

        if self.split == Split.TRAIN:
            return image
        else:
            return image, label, mask.int(), path
        
    def __len__(self) -> int:
        if self.samples is None:
            return 0
        return len(self.samples)
            
            
