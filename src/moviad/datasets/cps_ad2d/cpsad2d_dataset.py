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

PATCH_SIZE = 224

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
                    
                    # Gestione speciale per categoria cps_6 (come discusso precedentemente)
                    '''if category_name == "cps_6":
                        mask_path = annotations_path.parent / "defect" / (image_path.stem + ".png")
                    else:
                        mask_path = annotations_path / (image_path.stem + ".png")'''
                    
                    mask_path = annotations_path / (image_path.stem + ".png")

                    if not mask_path.exists():
                        continue 

                    # Apriamo immagine e maschera per estrarre le patch
                    with Image.open(image_path) as img_raw, Image.open(mask_path) as mask_raw:
                        img_gray = img_raw.convert("L")
                        mask_gray = mask_raw.convert("L")
                        
                        width, height = img_gray.size
                        
                        # Definiamo le coordinate delle due patch (Sinistra e Destra)
                        # Patch 1: (0, 0, 224, 224)
                        # Patch 2: (W-224, 0, W, 224)
                        coords = [
                            (0, 0, PATCH_SIZE, PATCH_SIZE),
                            (max(0, width - PATCH_SIZE), 0, width, PATCH_SIZE)
                        ]

                        for i, box in enumerate(coords):
                            # Ritaglio della patch sulla maschera
                            patch_mask = mask_gray.crop(box)
                            
                            # Determiniamo la label specifica per questa patch
                            # Se il valore massimo nella patch della maschera è > 0, è anomala
                            if patch_mask.getextrema()[1] > 0:
                                current_label = LabelName.ABNORMAL
                            else:
                                current_label = LabelName.NORMAL

                            # Aggiungiamo la patch alla lista dei campioni
                            all_samples.append({
                                "image_path": str(image_path),
                                "mask_path": str(mask_path),
                                "label": current_label,
                                "category": category_name,
                                "split": "train",
                                "patch_index": i,     # Identificatore della patch (0 o 1)
                                "patch_box": box      # Salviamo le coordinate per il dataloader
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
        num_train_normal = 600
        
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
        
        # 2. Selezione Test 
        remaining_normal_indices = df_normal_shuffled.index[num_train_normal:num_train_normal + 200]
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
                TRAIN: image, 
                TEST: image, label, mask, path
            """
            if self.samples is None:
                self.load_dataset()

            sample = self.samples.iloc[index]
            
            # 1. Caricamento e Ritaglio Immagine
            full_img = Image.open(sample.image_path).convert("RGB")
            # Usiamo il patch_box salvato (che contiene le coordinate [x0, y0, x1, y1])
            patch_img = full_img.crop(sample.patch_box)
            image = self.transform_image(patch_img)

            label = sample.label
            path = str(sample.image_path)

            # 2. Gestione Maschera (con Ritaglio)
            if label == LabelName.NORMAL or sample.mask_path is None:
                # Creiamo una maschera nera delle dimensioni della patch (224x224)
                mask = torch.zeros(1, image.shape[1], image.shape[2])
            else:
                full_mask = Image.open(sample.mask_path).convert("L")
                # Ritagliamo anche la maschera con lo stesso box dell'immagine
                patch_mask = full_mask.crop(sample.patch_box)
                mask = self.transform_mask(patch_mask)

            # 3. Restituzione in base allo split
            if self.split == Split.TRAIN:
                return image
            else:
                return image, label, mask.int(), path
        
    def __len__(self) -> int:
        if self.samples is None:
            return 0
        return len(self.samples)
            
            
