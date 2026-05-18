from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import torch
from PIL.Image import Image
import PIL
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from moviad.datasets.vad_dataset import VADDataset
from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.utilities.configurations import Split, TaskType, LabelName

"""
The MIIC dataset when downloaded follows this general structure:

miic_folder
    |- Anomaly_test
        |- ..
    |- Anomaly_train
        |- ..
    |- Inpainting_test
        |- ..
    |- Inpainting_train
        |- ..
    |- MANIFEST.TXT
    |- readme.txt
"""


class MiicDatasetClassEnum(Enum):
    """
    Enum class for MIIC dataset classes.
    """
    NORMAL = "normal"
    ANOMALY = "abnormal"


@dataclass
class MiicDatasetEntry:
    """
    Represents a single entry in the Miic dataset.
    Example of entry path: '/root/train_root/<split>_<class_name>_<image_id>.jpg'
    """
    image_path: Path
    mask_path: str
    class_name: MiicDatasetClassEnum
    image_id: int
    split: Split
    image: Image
    mask: Image

    def __init__(self, image_path: Path, mask_path: str = None, bounding_box_path: str = None):
        image_file_name = image_path.name
        self.image_path = image_path

        image_file_name_split = image_file_name.split('_')
        self.split = Split(image_file_name_split[0])
        self.class_name = MiicDatasetClassEnum(image_file_name_split[1])

        self.image_id = int(image_file_name_split[2].split('.')[0])
        self.mask_path = mask_path
        self.bounding_box_path = bounding_box_path

        self.image = None
        self.mask = None


class MiicDataset(VADDataset):
    """
    MIIC Dataset for Visual Anomaly Detection.
    
    Args:
        arguments (DatasetArguments): Dataset configuration arguments
        category (str): Dataset category (default: 'semiconductor')
        split (Split | list[Split]): Data split(s) to load
        dataset_path (Path): Path to the MIIC dataset root folder
        preload_images (bool): Whether to preload images into memory
    """
    
    def __init__(
        self,
        arguments: DatasetArguments,
        category: str = 'semiconductor',
        split: Split | list[Split] = Split.TRAIN,
        preload_images: bool = False
    ):
        super().__init__(arguments, category, split)
        
        self.dataset_path = Path(arguments.dataset_path) if arguments.dataset_path else None
        self.preload_images = preload_images
        self.data = []
        self._loaded = False
        
        # Set up dataset paths
        if self.dataset_path:
            self.training_root_path = self.dataset_path / "Anomaly_train"
            self.test_abnormal_image_root_path = self.dataset_path / "Anomaly_test/abnormal_img"
            self.test_normal_image_root_path = self.dataset_path / "Anomaly_test/normal_img"
            self.test_abnormal_mask_root_path = self.dataset_path / "Anomaly_test/abnormal_mask"
            self.test_abnormal_bounding_box_root_path = self.dataset_path / "Anomaly_test/abnormal_bbox"
        
        # Load the dataset
        if self.dataset_path:
            self.load_dataset()

    def is_loaded(self) -> bool:
        """Check if the dataset has been loaded."""
        return self._loaded

    @staticmethod
    def get_categories() -> list:
        """Get available categories for MIIC dataset."""
        return ['semiconductor']

    def split_dataset(self, train_size, valid_size):
        """
        Split the dataset into training and validation sets.
        
        Args:
            train_size: Size or ratio of training set
            valid_size: Size or ratio of validation set
            
        Note: Not implemented for MIIC as it uses predefined splits.
        """
        raise NotImplementedError("MIIC dataset uses predefined train/test splits.")

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        
        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            For training: image tensor
            For testing: (image, label, mask, path) tuple
        """
        image_entry = self.data[idx]
        
        # Load image if not preloaded
        if not self.preload_images:
            with PIL.Image.open(image_entry.image_path) as img:
                image_entry.image = img.convert("RGB")

        image = image_entry.image
        image = self.transform_image(image)

        # Training split returns only images
        if self.split == Split.TRAIN or (isinstance(self.split, list) and Split.TRAIN in self.split):
            return image

        # Test split returns image, label, mask, and path
        label = (LabelName.NORMAL.value 
                if image_entry.class_name.value == MiicDatasetClassEnum.NORMAL.value 
                else LabelName.ABNORMAL.value)
        
        mask_path = image_entry.mask_path
        path = str(image_entry.image_path)
        
        if mask_path is not None:
            if not self.preload_images:
                with PIL.Image.open(mask_path) as mask_img:
                    mask = mask_img.convert("L")
            else:
                mask = image_entry.mask
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros((1, *self.dataset_arguments.gt_mask_size), dtype=torch.float32)

        return image, label, mask, path

    def load_dataset(self):
        """Load the dataset based on the specified split."""
        if isinstance(self.split, list):
            # Handle multiple splits
            for split in self.split:
                if split == Split.TRAIN:
                    self.__load_training_data(self.training_root_path)
                elif split == Split.TEST:
                    self.__load_test_data(
                        self.test_normal_image_root_path,
                        self.test_abnormal_image_root_path,
                        self.test_abnormal_mask_root_path,
                        self.test_abnormal_bounding_box_root_path
                    )
        else:
            # Handle single split
            if self.split == Split.TRAIN:
                self.__load_training_data(self.training_root_path)
            elif self.split == Split.TEST:
                self.__load_test_data(
                    self.test_normal_image_root_path,
                    self.test_abnormal_image_root_path,
                    self.test_abnormal_mask_root_path,
                    self.test_abnormal_bounding_box_root_path
                )
        
        self._loaded = True

    def __load_training_data(self, normal_images_root_path: Path):
        """Load training data (only normal images)."""
        assert normal_images_root_path.exists(), \
            f"Normal images root path {normal_images_root_path} does not exist"
        
        image_file_list = list(normal_images_root_path.glob('**/*train_normal_*.jpg'))

        if image_file_list is None or len(image_file_list) == 0:
            raise FileNotFoundError(f"No images found in {normal_images_root_path}")

        for image in image_file_list:
            image_entry = MiicDatasetEntry(image)
            if self.preload_images:
                with PIL.Image.open(image_entry.image_path) as img:
                    image_entry.image = img.convert("RGB")
            self.data.append(image_entry)

    def __load_test_data(
        self,
        normal_images_root_path: Path,
        abnormal_image_root_path: Path,
        mask_root_path: Path,
        bounding_box_root_path: Path
    ):
        """Load test data (normal and abnormal images with masks)."""
        assert normal_images_root_path.exists(), \
            f"Normal images root path {normal_images_root_path} does not exist"
        assert abnormal_image_root_path.exists(), \
            f"Abnormal images root path {abnormal_image_root_path} does not exist"
        assert mask_root_path.exists(), \
            f"Mask root path {mask_root_path} does not exist"
        assert bounding_box_root_path.exists(), \
            f"Bounding box root path {bounding_box_root_path} does not exist"

        normal_image_file_list = sorted(list(normal_images_root_path.glob('**/*.jpg')))
        abnormal_image_file_list = sorted(list(abnormal_image_root_path.glob('**/*.jpg')), key=lambda x: x)
        mask_file_list = sorted(list(mask_root_path.glob('**/*.jpg')), key=lambda x: x)
        bounding_box_file_list = sorted(list(bounding_box_root_path.glob('**/*.jpg')), key=lambda x: x)

        if normal_image_file_list is None or len(normal_image_file_list) == 0:
            raise FileNotFoundError(f"No images found in {normal_images_root_path}")
        if abnormal_image_file_list is None or len(abnormal_image_file_list) == 0:
            raise FileNotFoundError(f"No images found in {abnormal_image_root_path}")
        if mask_file_list is None or len(mask_file_list) == 0:
            raise FileNotFoundError(f"No images found in {mask_root_path}")

        # Load normal images
        for image in normal_image_file_list:
            image_entry = MiicDatasetEntry(image)
            if self.preload_images:
                with PIL.Image.open(image_entry.image_path) as img:
                    image_entry.image = img.convert("RGB")
            self.data.append(image_entry)

        # Load abnormal images with masks
        for item in zip(abnormal_image_file_list, mask_file_list, bounding_box_file_list):
            abnormal_image_path, mask_path, bounding_box_path = item
            image_entry = MiicDatasetEntry(abnormal_image_path, mask_path, bounding_box_path)
            if self.preload_images:
                with PIL.Image.open(abnormal_image_path) as img:
                    image_entry.image = img.convert("RGB")
                with PIL.Image.open(mask_path) as mask:
                    image_entry.mask = mask.convert("L")
            self.data.append(image_entry)

    def contaminate(self, ratio: float, seed: int = 42) -> int:
        """
        Contaminate the dataset with anomalies.
        
        Args:
            ratio: Contamination ratio
            seed: Random seed
            
        Returns:
            Number of contaminated samples
        """
        raise NotImplementedError("Dataset contamination not yet supported on this dataset.")

    def compute_contamination_ratio(self) -> float:
        """
        Compute the contamination ratio of the dataset.

        Returns:
            float: The contamination ratio.
        """
        raise NotImplementedError("Dataset contamination not yet supported on this dataset.")