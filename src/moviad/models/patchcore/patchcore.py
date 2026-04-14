"""PyTorch model for the PatchCore model implementation."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
#from memory_profiler import profile
from torch import Tensor, autocast, nn
from tqdm import tqdm

from .product_quantizer import ProductQuantizer
from moviad.models.vad_model import VADModel
from moviad.models.patchcore.anomaly_map import AnomalyMapGenerator
from moviad.models.patchcore.kcenter_greedy import CoresetExtractor
from moviad.models.training_args import TrainingArgs
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.utilities.get_sizes import params_to_mb, count_params

class PatchCore(VADModel):
    """Patchcore Module."""

    def __init__(
        self,
        feature_extractor: CustomFeatureExtractor,
        coreset_extractor: CoresetExtractor = None,
        num_neighbors: int = 9,
        apply_quantization: bool = False,
        memory_bank_size: int = 10000
    ) -> None:

        """
        Constructor of the patch-core model

        Args:
            device (torch.device): device to be used during the training
            input_size (tuple[int]): size of the input images
            feature_extractor (CustomFeatureExtractor): feature extractor to be used
            num_neighbors (int): number of neighbors to be considered in the k-nn search
        """

        super().__init__()

        self.num_neighbors = num_neighbors
        self.device = feature_extractor.device
        self.dtype = torch.float32  # Explicit dtype for consistency across training/inference

        self.feature_extractor = feature_extractor
        self.feature_pooler = torch.nn.AvgPool2d(3,1,1)
        self.anomaly_map_generator = AnomalyMapGenerator()
        self.memory_bank_size = memory_bank_size
        self.memory_bank: Tensor

        self.apply_quantization = apply_quantization
        if apply_quantization:
            self.product_quantizer = ProductQuantizer()

        if coreset_extractor is None:
            self.coreset_extractor = CoresetExtractor(False , self.device, k=self.memory_bank_size)
        else:
            self.coreset_extractor = coreset_extractor

    def to(self, device: torch.device):
        super().to(device)
        self.feature_extractor.to(device)

        if hasattr(self, "memory_bank"):
            if type(self.memory_bank) is dict:
                for k,v in self.memory_bank.items():
                    self.memory_bank[k] = v.to(self.device)
            else:
                self.memory_bank = self.memory_bank.to(self.device)

        self.device = device

    def extract_embedding(self, batch:torch.Tensor):
        with torch.no_grad():
            features = self.feature_extractor(batch.to(self.device))

        #concatenate the embeddings
        if isinstance(features, dict):
            features = list(features.values())

        # Apply smoothing (3x3 average pooling) to the features.
        smoothing = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        features  = [smoothing(feature) for feature in features]

        # Compute maximum shape.
        H_max = max([f.shape[2] for f in features])
        W_max = max([f.shape[3] for f in features])

        # Create resize function instance.
        resizer = torch.nn.Upsample(size=(H_max, W_max), mode="nearest")

        # Apply resize function for all input tensors.
        features = [resizer(f) for f in features]

        embedding =  torch.cat(features, dim=1)

        batch_size, _, width, height = embedding.shape
        embedding = self.reshape_embedding(embedding)

        return embedding, batch_size, width, height

    def forward(self, input_tensor: Tensor) -> Tensor | dict[str, Tensor]:
        """Return Embedding during training, or a tuple of anomaly map and anomaly score during testing.

        Steps performed:
        1. Get features from a CNN.
        2. Generate embedding based on the features.
        3. Compute anomaly map in test mode, return the embedding in training mode

        Args:
            input_tensor (Tensor): Input tensor

        Returns:
            Tensor | dict[str, Tensor]: Embedding for training,
                anomaly map and anomaly score for testing.
        """

        image_size = input_tensor.shape[2:]

        #extract the features for the input tensor
        embedding, batch_size, width, height = self.extract_embedding(input_tensor)

        #embedding shape: (#num_patches, emb_dim)

        if self.training:
            return embedding
        else:
            return self.calculate_anomaly_maps_scores(
                embedding=embedding,
                batch_size=batch_size,
                width=width,
                height=height,
                image_size=image_size,
                memory_bank=self.memory_bank
            )

    def calculate_anomaly_maps_scores(
        self,
        embedding: Tensor,
        memory_bank: Tensor,
        batch_size: int,
        width: int,
        height: int,
        image_size: tuple,
    ) -> tuple[Tensor, Tensor]:

        if self.feature_extractor.quantized:
            embedding = torch.int_repr(embedding).to(torch.float64)
        else:
            # Ensure embedding dtype matches memory_bank for consistent distance computation
            embedding = embedding.to(self.dtype)

        # apply nearest neighbor search
        if self.apply_quantization:
            patch_scores, locations = self.nearest_neighbors_quantized(embedding=embedding, n_neighbors=1)
        else:
            patch_scores, locations = self.nearest_neighbors(embedding=embedding, n_neighbors=1, memory_bank=memory_bank)

        # print(patch_scores.shape)
        # print("Locations" + str(locations.shape))

        # reshape to batch dimension
        patch_scores = patch_scores.reshape((batch_size, -1))
        locations = locations.reshape((batch_size, -1))

        # compute the anomaly score of the images
        pred_scores = self.compute_anomaly_score(patch_scores, locations, embedding, memory_bank)

        # reshape to w,h
        patch_scores = patch_scores.reshape((batch_size, 1, width, height))

        # get the anomaly map
        anomaly_maps = self.anomaly_map_generator(patch_scores, image_size = image_size)

        return anomaly_maps, pred_scores

    def train_epoch(
        self, epoch, train_dataloader, training_args: TrainingArgs
    ):
        embeddings = []

        with torch.no_grad():

            print("Embedding Extraction:")
            for batch in tqdm(iter(train_dataloader)):
                with autocast(device_type=self.device.type, enabled=True):
                    embedding = self(batch.to(self.device))
                    embeddings.append(embedding)

            embeddings = torch.cat(embeddings, dim = 0)
            torch.cuda.empty_cache()

            #apply coreset reduction
            print("Coreset Extraction:")
            coreset = self.coreset_extractor.extract_coreset(embeddings)

            if self.apply_quantization:
                assert self.product_quantizer is not None, "Product Quantizer not initialized"

                self.product_quantizer.fit(coreset)
                coreset = self.product_quantizer.encode(coreset)

            # Ensure memory_bank is always float32 for consistent distance computation
            self.memory_bank = coreset.to(self.dtype)

    def train_chunk(
        self, train_dataloader, training_args: TrainingArgs
    ):
        embeddings = []

        with torch.no_grad():

            print("Embedding Extraction:")
            for batch in tqdm(iter(train_dataloader)):
                with autocast(device_type=self.device.type, enabled=True):
                    embedding = self(batch.to(self.device))
                    embeddings.append(embedding)

            embeddings = torch.cat(embeddings, dim = 0)
            torch.cuda.empty_cache()

            total_embbeddings = torch.cat([embeddings, self.memory_bank], dim=0) if hasattr(self, "memory_bank") else embeddings

            #apply coreset reduction
            print("Coreset Extraction:")
            coreset = self.coreset_extractor.extract_coreset(total_embbeddings)

            if self.apply_quantization:
                assert self.product_quantizer is not None, "Product Quantizer not initialized"

                self.product_quantizer.fit(coreset)
                coreset = self.product_quantizer.encode(coreset)

            # Ensure memory_bank is always float32 for consistent distance computation
            self.memory_bank = coreset.to(self.dtype)

            del embeddings
            del total_embbeddings
            torch.cuda.empty_cache()

    def generate_embedding(self, features: dict[str, Tensor]) -> Tensor:
        """Generate embedding from hierarchical feature map.

        Args:
            features: dict[str:Tensor]: Hierarchical feature map from a CNN

        Returns:
            Embedding vector [Tensor]
        """

        embeddings = [features[self.layers[0]]]

        for layer in self.layers[1:]:
            layer_embedding = features[layer]
            layer_embedding = F.interpolate(layer_embedding, size = embeddings[0].shape[-2], mode = "bilinear")
            if self.quantize:
                layer_embedding = layer_embedding.quantize()
            embeddings.append(layer_embedding)

        embeddings = torch.cat(embeddings, 1)

        return embeddings

    @staticmethod
    def reshape_embedding(embedding: Tensor) -> Tensor:
        """
        Reshape Embedding.

        Reshapes Embedding to the following format:
        [Batch, Embedding, Patch, Patch] to [Batch*Patch*Patch, Embedding]

        Args:
            embedding (Tensor): Embedding tensor extracted from CNN features.

        Returns:
            Tensor: Reshaped embedding tensor.
        """

        embedding_size = embedding.size(1)
        embedding = embedding.permute(0,2,3,1).reshape(-1, embedding_size)
        return embedding

    def euclidean_distance(x: Tensor, y: Tensor, quantized:bool) -> Tensor:
        """
        Calculates pair-wise distance between row vectors in x and those in y.

        Args:
            x: input tensor 1
            y: input tensor 2
            quantized: bool, True if x and y are quantized tensors

        Returns:
            Matrix of distances between row vectors in x and y.
        """

        if quantized:
            return torch.cdist(x.dequantize(), y.dequantize())

        return torch.cdist(x.to(torch.float32), y.to(torch.float32))

    def nearest_neighbors(self, embedding: Tensor, n_neighbors: int, memory_bank: torch.Tensor = None) -> tuple[Tensor, Tensor]:
        """
        Nearest neighbors using brute force method and euclidean norm, processed in chunks to save VRAM.
        """
        if memory_bank is None:
            memory_bank = self.memory_bank

        # Ensure both tensors are on the same device and have the same dtype
        embedding = embedding.to(self.device, dtype=self.dtype)
        memory_bank = memory_bank.to(self.device, dtype=self.dtype)

        chunk_size = 1024
        patch_scores_list = []
        locations_list = []

        # Process the embeddings in smaller chunks
        for i in range(0, embedding.size(0), chunk_size):
            chunk = embedding[i : i + chunk_size]

            distances = PatchCore.euclidean_distance(
                chunk,
                memory_bank,
                quantized=self.feature_extractor.quantized
            )

            if n_neighbors == 1:
                scores, locs = distances.min(1)
            else:
                scores, locs = distances.topk(k=n_neighbors, largest=False, dim=1)

            patch_scores_list.append(scores)
            locations_list.append(locs)

        # Concatenate the results back together
        patch_scores = torch.cat(patch_scores_list, dim=0)
        locations = torch.cat(locations_list, dim=0)

        return patch_scores, locations

    def nearest_neighbors_quantized(self, embedding: Tensor, n_neighbors: int) -> tuple[Tensor, Tensor]:

        device = self.device
        embedding = embedding.to(device)
        self.memory_bank = self.memory_bank.to(device)

        quantized_embedding = self.product_quantizer.encode(embedding).to(device)

        distances = PatchCore.euclidean_distance(
            quantized_embedding,
            self.memory_bank,
            quantized=self.feature_extractor.quantized,
        )

        _, top_locations = distances.topk(
            k=1000, largest=False, dim=1
        )

        quantized_neighbors = self.memory_bank[top_locations]

        B, K, Dq = quantized_neighbors.shape

        decoded_neighbors = self.product_quantizer.decode(
            quantized_neighbors.view(-1, Dq)
        ).view(B, K, -1).to(device)

        embedding_expanded = embedding.unsqueeze(1)

        exact_distances = PatchCore.euclidean_distance(
            embedding_expanded,
            decoded_neighbors,
            quantized=False,
        )

        exact_distances = exact_distances.squeeze(1)  # [B, top_k]

        patch_scores, locations = exact_distances.topk(
            k=n_neighbors, largest=False, dim=1
        )

        return patch_scores, locations


    def compute_anomaly_score_old(self, patch_scores: Tensor, locations: Tensor, embedding: Tensor, memory_bank: Tensor) -> Tensor:
        """
        Compute Image-Level Anomaly Score.

        Args:
            patch_scores (torch.Tensor): Patch-level anomaly scores
            locations: Memory bank locations of the nearest neighbor for each patch location
            embedding: The feature embeddings that generated the patch scores

        Returns:
            Tensor: Image-level anomaly scores
        """
        if self.apply_quantization:
            assert self.product_quantizer is not None
            memory_bank = self.product_quantizer.decode(memory_bank)
            memory_bank = memory_bank.to(self.device)

        # Don't need to compute weights if num_neighbors is 1
        if self.num_neighbors == 1:
            return patch_scores.amax(1)

        batch_size, num_patches = patch_scores.shape
        # 1. Find the patch with the largest distance to it's nearest neighbor in each image
        max_patches = torch.argmax(patch_scores, dim=1)  # indices of m^test,* in the paper
        # m^test,* in the paper
        max_patches_features = embedding.reshape(batch_size, num_patches, -1)[torch.arange(batch_size), max_patches]

        # 2. Find the distance of the patch to it's nearest neighbor, and the location of the nn in the membank
        score = patch_scores[torch.arange(batch_size), max_patches]  # s^* in the paper
        nn_index = locations[torch.arange(batch_size), max_patches]  # indices of m^* in the paper

        # 3. Find the support samples of the nearest neighbor in the membank
        nn_sample = memory_bank[nn_index, :]  # m^* in the paper

        # indices of N_b(m^*) in the paper
        memory_bank_effective_size = memory_bank.shape[0]  # edge case when memory bank is too small
        if self.apply_quantization:
            _, support_samples = self.nearest_neighbors_quantized(
                nn_sample,
                n_neighbors=min(self.num_neighbors, memory_bank_effective_size),
            )
        else:
            _, support_samples = self.nearest_neighbors(
                nn_sample,
                n_neighbors=min(self.num_neighbors, memory_bank_effective_size),
                memory_bank=memory_bank
            )

        # 4. Find the distance of the patch features to each of the support samples
        distances = PatchCore.euclidean_distance(max_patches_features.unsqueeze(1), memory_bank[support_samples], self.feature_extractor.quantized)

        # 5. Apply softmax to find the weights
        weights = (1 - F.softmax(distances.squeeze(1), 1))[..., 0]

        # 6. Apply the weight factor to the score
        return weights * score  # s in the paper

    def compute_anomaly_score(self, patch_scores: Tensor, locations: Tensor, embedding: Tensor, memory_bank: Tensor) -> Tensor:
        """
        Compute Image-Level Anomaly Score.
        """
        if self.apply_quantization:
            assert self.product_quantizer is not None
            memory_bank = self.product_quantizer.decode(memory_bank)
            memory_bank = memory_bank.to(self.device)

        if self.num_neighbors == 1:
            return patch_scores.amax(1)

        batch_size, num_patches = patch_scores.shape
        max_patches = torch.argmax(patch_scores, dim=1)
        max_patches_features = embedding.reshape(batch_size, num_patches, -1)[torch.arange(batch_size), max_patches]

        score = patch_scores[torch.arange(batch_size), max_patches]
        nn_index = locations[torch.arange(batch_size), max_patches]
        nn_sample = memory_bank[nn_index, :]

        memory_bank_effective_size = memory_bank.shape[0]
        if self.apply_quantization:
            _, support_samples = self.nearest_neighbors_quantized(
                nn_sample,
                n_neighbors=min(self.num_neighbors, memory_bank_effective_size),
            )
        else:
            _, support_samples = self.nearest_neighbors(
                nn_sample,
                n_neighbors=min(self.num_neighbors, memory_bank_effective_size),
                memory_bank=memory_bank
            )

        # 4. Find the distance of the patch features to each of the support samples
        distances = PatchCore.euclidean_distance(max_patches_features.unsqueeze(1), memory_bank[support_samples], self.feature_extractor.quantized)

        # 5. Apply Temperature Scaling (sqrt of embedding dimensions)
        D = torch.sqrt(torch.tensor(embedding.shape[-1], dtype=torch.float32, device=distances.device))
        scaled_distances = distances.squeeze(1) / D

        # Apply softmax to the scaled distances
        weights = (1 - F.softmax(scaled_distances, dim=1))[..., 0]

        # 6. Apply the weight factor to the score
        return weights * score

    def save(self, save_path: str):
        """
        Save the Patchcore model

        Parameters:
        ----------
            save_path (str): where the model will be saved
        """
        if self.apply_quantization:
            assert self.product_quantizer is not None
            self.product_quantizer.save(save_path + "/product_quantizer.bin")
        torch.save(self.memory_bank, save_path)

    def load(self, model_state_dict_patch, quantizer_state_dict_path):
        """
        Load the Patchcore model

        Parameters:
        ----------
            model_state_dict_patch (dict): model state dictionary
            quantizer_state_dict_patch (dict): quantizer state dictionary
        """

        self._load_model(model_state_dict_patch)
        if quantizer_state_dict_path is not None:
            self.product_quantizer = ProductQuantizer()
            self.product_quantizer.load(quantizer_state_dict_path)
            self.apply_quantization = True

    def _load_model(self, path):
        """
        Load the Patchcore memory bank

        Parameters:
        ----------
            path (str): where the pt file containing the memory bank is stored
        """

        state_dict = torch.load(path, map_location=self.device)

        ## TODO: MemoryBank quantization

        if "memory_bank" not in state_dict.keys():
            raise RuntimeError("Memory Bank tensor not in model checkpoint")

        # load the memory bank and ensure it's always float32
        self.memory_bank = state_dict["memory_bank"].to(self.dtype)

    def reset_model(self):
        self.memory_bank = None

    def save_anomaly_map(self, dirpath, anomaly_map, pred_score, filepath, x_type, mask):
        """
        Args:
            dirpath     (str)       : Output directory path.
            anomaly_map (np.ndarray): Anomaly map with the same size as the input image.
            filepath    (str)       : Path of the input image.
            x_type      (str)       : Anomaly type (e.g. "good", "crack", etc).
            contour     (float)     : Threshold of contour, or None.
        """
        def min_max_norm(image):
            a_min, a_max = image.min(), image.max()
            return (image - a_min) / (a_max - a_min)

        def cvt2heatmap(gray):
            return cv.applyColorMap(np.uint8(gray), cv.COLORMAP_JET)

        # Get the image file name.
        filename = os.path.basename(filepath)

        # Load the image file and resize.
        original_image = cv.imread(filepath)
        original_image = cv.resize(original_image, anomaly_map.shape[:2])

        # Normalize anomaly map for easier visualization.
        anomaly_map_norm = cvt2heatmap(255 * min_max_norm(anomaly_map))

        # Overlay the anomaly map to the origimal image.
        output_image = (anomaly_map_norm / 2 + original_image / 2).astype(np.uint8)

        # Create a figure and axes
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))

        #convert the images to RGB
        original_image = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)
        output_image = cv.cvtColor(output_image, cv.COLOR_BGR2RGB)

        # Display the input image
        axes[0].imshow(original_image)
        axes[0].set_title(f'Original Image {x_type}')
        axes[0].axis('off')

        # Display the mask image
        axes[1].imshow(mask.squeeze(), cmap ='gray')
        axes[1].set_title(f'Mask')
        axes[1].axis('off')

        # Display the final image
        axes[2].imshow(output_image)
        axes[2].set_title(f'Heatmap {pred_score}')
        axes[2].axis('off')

        # Show the plot
        plt.savefig(str(dirpath + f"/{x_type}_{filename}.jpg"))

    def get_model_size(self):

        feautre_extractor_params = self.feature_extractor.get_size()
        memory_bank_params = params_to_mb(count_params(self.memory_bank))

        return {
            "feature_extractor": feautre_extractor_params,
            "memory_bank": memory_bank_params,
            "total": feautre_extractor_params + memory_bank_params
        }
