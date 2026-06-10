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

IMG_EXTENSIONS = (".png", ".PNG")

CATEGORIES = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)

IMG_SIZE = (3, 900, 900)

"""Create MVTec AD samples by parsing the MVTec AD data file structure.

    The files are expected to follow the structure:
        path/to/dataset/split/category/image_filename.png
        path/to/dataset/ground_truth/category/mask_filename.png

    This function creates a dataframe to store the parsed information based on the following format:
"""


class MVTecDataset(VADDataset):
    """MVTec dataset class."""

    def __init__(
        self,
        dataset_arguments: DatasetArguments,
        category: str,
        split: Split | list[Split]
    ) -> None:

        super().__init__(
            dataset_arguments,
            category,
            split
        )

        self.root_category = Path(self.dataset_arguments.dataset_path) / Path(self.category)
        self.samples: pd.DataFrame = None

        self.load_dataset()

    def is_loaded(self) -> bool:
        return self.samples is not None

    def load_dataset(self):
        if self.is_loaded():
            print("Dataset already loaded")
            return

        root = Path(self.root_category)

        samples_list = [
            (str(root),) + f.parts[-3:]
            for f in root.glob(r"**/*")
            if f.suffix in IMG_EXTENSIONS
        ]

        if not samples_list:
            msg = f"Found 0 images in {root}"
            raise RuntimeError(msg)

        samples = pd.DataFrame(
            samples_list, columns=["path", "split", "label", "image_path"]
        )

        # Modify image_path column by converting to absolute path
        samples["image_path"] = (
                samples.path
                + "/"
                + samples.split
                + "/"
                + samples.label
                + "/"
                + samples.image_path
        )

        # Create label index for normal (0) and anomalous (1) images.
        samples.loc[(samples.label == "good"), "label_index"] = LabelName.NORMAL
        samples.loc[(samples.label != "good"), "label_index"] = LabelName.ABNORMAL
        samples.label_index = samples.label_index.astype(int)

        if self.split == Split.TEST:

            # separate masks from samples
            mask_samples = samples.loc[samples.split == "ground_truth"].sort_values(
                by="image_path", ignore_index=True
            )
            samples = samples[samples.split != "ground_truth"].sort_values(
                by="image_path", ignore_index=True
            )

            # assign mask paths to anomalous test images
            samples["mask_path"] = ""
            samples.loc[
                (samples.split == "test") & (samples.label_index == LabelName.ABNORMAL),
                "mask_path",
            ] = mask_samples.image_path.to_numpy()

            # assert that the right mask files are associated with the right test images
            abnormal_samples = samples.loc[samples.label_index == LabelName.ABNORMAL]
            if (
                    len(abnormal_samples)
                    and not abnormal_samples.apply(
                lambda x: Path(x.image_path).stem in Path(x.mask_path).stem, axis=1
            ).all()
            ):
                msg = """Mismatch between anomalous images and ground truth masks. Make sure t
                he mask files in 'ground_truth' folder follow the same naming convention as the
                anomalous images in the dataset (e.g. image: '000.png', mask: '000.png' or '000_mask.png')."""
                raise Exception(msg)

        split_str = self.split.value if hasattr(self.split, 'value') else self.split
        self.samples = samples[samples.split == split_str].reset_index(drop=True)

        # if not self.use_original_splits:
        #     self.split_dataset(train_size=0.7, valid_size=0.2)

    def __len__(self) -> int:
        return len(self.samples)

    def contaminate(self, source: VADDataset, ratio: float, seed: int = 42) -> int:
        if type(source) is not MVTecDataset:
            raise ValueError("Dataset should be of type MVTecDataset")
        if self.samples is None:
            raise ValueError("Destination dataset is not loaded")
        if source.samples is None:
            raise ValueError("Source dataset is not loaded")

        torch.manual_seed(seed)
        contamination_set_size = int(math.floor(len(self.samples) * ratio))
        contaminated_entries_indices = source.samples[source.samples["label_index"] == LabelName.ABNORMAL.value].index
        if len(contaminated_entries_indices) < contamination_set_size:
            raise DatasetTooSmallToContaminateException(
                f"Source dataset does not contain enough abnormal entries to contaminate the destination dataset. "
                f"Source dataset contains {len(contaminated_entries_indices)} abnormal entries, "
                f"while {contamination_set_size} are required."
            )

        contaminated_entries_indices = np.random.choice(contaminated_entries_indices, contamination_set_size,
                                                        replace=False)
        for index in contaminated_entries_indices:
            entry_metadata = source.samples.iloc[index]
            if source.preload_imgs:
                entry = source.data[index]
                self.data.append(entry)
            else:
                entry = self.transform_image(
                    Image.open(self.samples.iloc[index].image_path).convert("RGB")
                )
                self.data.append(entry)
                source.data = [e for e in source.data if hash(e) != hash(entry)]

            self.samples = pd.concat([self.samples, pd.DataFrame([entry_metadata])], ignore_index=True)
            index_label = source.samples.index[index]

        source.samples = source.samples.drop(contaminated_entries_indices).reset_index(drop=True)
        source.data = [source.data[i] for i in range(len(source.data)) if i not in contaminated_entries_indices]
        return contamination_set_size
    
    def __getitem__(self, index: int):
        """
        Args:
            index (int) : index of the element to be returned

        Returns:
            image (Tensor) : tensor of shape (C,H,W) with values in [0,1]
            label (int) : label of the image
            mask (Tensor) : tensor of shape (1,H,W) with values in [0,1]
            path (str) : path of the input image
        """

        if self.samples is None:
            self.load_dataset()

        # open the image and get the tensor
        image = self.transform_image(
            Image.open(self.samples.iloc[index].image_path).convert("RGB")
        )

        if self.split == Split.TRAIN:
            return image
        else:
            # return also the label, the mask and the path
            label = self.samples.iloc[index].label_index
            path = self.samples.iloc[index].image_path
            if label == LabelName.ABNORMAL:
                mask = Image.open(self.samples.iloc[index].mask_path).convert("L")
                mask = self.transform_mask(mask)

            else:
                mask = torch.zeros(1, *self.dataset_arguments.img_size)

            return image, label, mask.int(), path , self.category 

    @staticmethod
    def get_categories() -> list:
        return list(CATEGORIES)