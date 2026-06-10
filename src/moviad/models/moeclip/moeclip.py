import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import os
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

# Import from MOVIAD
from moviad.models.vad_model import VADModel
from moviad.models.training_args import TrainingArgs

# Import from MoE-CLIP
from .utils import setup_seed
from .MoECLIP.moe_adapter import MoECLIP
from .MoECLIP.clip import create_model
from .forward_utils import (
    get_adapted_single_class_text_embedding,
    calculate_similarity_map,
    calculate_seg_loss,
    get_adapted_text_embedding
)
from .constants import DOMAINS, REAL_NAMES, PROMPTS , CLASS_NAMES



class MoECLIPArgs(TrainingArgs):

    def init_train(self, model):
        
        if not hasattr(self, 'optimizer') or self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                model.params_to_train,
                lr = 5e-4,
                betas = (0.5, 0.999)
            )

        if not hasattr(self, 'scheduler') or self.scheduler is None:
            self.scheduler = MultiStepLR(self.optimizer, milestones=[10, 15], gamma=0.5)

    
    def __to_dict__(self):
        return {
            "optimizer": self.optimizer_to_dict(self.optimizer) if hasattr(self, 'optimizer') and self.optimizer else None,
        }


class MoECLIPModel(VADModel):

    def __init__(
        self,
        device: torch.device,
        clip_model_name: str = "ViT-L-14-336", #or ViT-B-16-plus-240
        img_size: int = 336,
        use_paa: bool = True, #Flag on patch average aggregation
        seg_proj_sharing_strategy: str = "shared", #or "separate" 
        image_adapt_weight: float = 0.1,
        moe_r: int =8,
        use_fofs: bool = True,
        moe_lora_alpha: int = 16,
        moe_num_experts: int = 4,
        moe_top_k: int = 2,
        moe_layers: list = [5, 11, 17, 23],
        relu: bool = False,
        balance_loss_lambda: float = 0.01,
        etf_loss_lambda: float = 0.01,
        checkpoint_path: str = None,
        dataset: str = None,
    ):  
        '''Initialize the MoECLIPModel

        Args:
            device (torch.device): The device to run the model on.
            clip_model_name (str, optional): The name of the CLIP model to use. Defaults to "ViT-L-14-336".
            img_size (int, optional): The size of the input images. Defaults to 336.
            use_paa (bool, optional): Whether to use patch average aggregation. Defaults to True.
            seg_proj_sharing_strategy (str, optional): The strategy for sharing segmentation projection layers. Defaults to "shared".
            image_adapt_weight (float, optional): The weight for image adaptation loss. Defaults to 0.1.
            moe_r (int, optional): The reduction factor for the MoE layers. Defaults to 8.
            use_fofs (bool, optional): Whether to use feature-only fine-tuning strategy. Defaults to True.
            moe_lora_alpha (int, optional): The alpha value for LoRA in MoE layers. Defaults to 16.
            moe_num_experts (int, optional): The number of experts in each MoE layer. Defaults to 4.
            moe_top_k (int, optional): The top-k value for expert selection in MoE layers. Defaults to 2.
            moe_layers (list, optional): The list of layer indices where MoE is applied. Defaults to [5, 11, 17, 23].
            relu (bool, optional): Whether to apply ReLU activation after the final projection layer. Defaults to False.
        '''
        
        super().__init__()

        self.device = device
        self.clip_model_name = clip_model_name
        self.img_size = img_size
        self.use_paa = use_paa
        self.seg_proj_sharing_strategy = seg_proj_sharing_strategy
        self.image_adapt_weight = image_adapt_weight
        self.moe_r = moe_r
        self.use_fofs = use_fofs
        self.moe_lora_alpha = moe_lora_alpha
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.moe_layers = moe_layers
        self.relu = relu
        self.checkpoint_path = checkpoint_path
        self.dataset = dataset
        self.balance_loss_lambda = balance_loss_lambda
        self.etf_loss_lambda = etf_loss_lambda
        self.adapt_text = False 
        self.category = None

        self.cached_class_text_embedding = None

        self.clip_model = create_model(
            model_name=self.clip_model_name,
            img_size=self.img_size,
            device=self.device,
            pretrained= 'openai',
            require_pretrained = True,
        )

        self.clip_model.eval()

        self.model = MoECLIP(
            clip_model=self.clip_model,
            use_paa=self.use_paa,
            seg_proj_sharing_strategy=self.seg_proj_sharing_strategy,
            image_adapt_weight=self.image_adapt_weight,
            moe_r=self.moe_r,
            use_fofs=self.use_fofs,
            moe_lora_alpha=self.moe_lora_alpha,
            moe_num_experts=self.moe_num_experts,
            moe_top_k=self.moe_top_k,
            moe_layers=self.moe_layers,
            relu=self.relu
        ).to(self.device)

        self.model.eval()

        self.params_to_train = self._extract_trainable_params(self.model)

        if self.checkpoint_path is not None and os.path.exists(self.checkpoint_path):
            print(f"Loading MoE-CLIP checkpoint from {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            if "text_adapter" in checkpoint:
                self.model.text_adapter.load_state_dict(checkpoint["text_adapter"])
                self.adapt_text = True 
            if "image_adapter" in checkpoint:
                self.model.image_adapter.load_state_dict(checkpoint["image_adapter"])
            '''if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])'''
        
        self._prepare_text_embeddings()
    
    def _prepare_text_embeddings(self):
        """Generates and caches text embeddings matching MoViAD single-class streaming behavior."""
        text_encoder_model = self.model if self.adapt_text else self.clip_model
        
        with torch.no_grad():
            # Verify if the requested category is configured within MoECLIP definitions
            if self.dataset in CLASS_NAMES and self.category in CLASS_NAMES[self.dataset]:
                self.cached_class_text_embedding = get_adapted_single_class_text_embedding(
                    text_encoder_model, self.dataset, self.category, self.device
                )
                #print(f"[MoE-CLIP] Successfully generated cached text embeddings for class: {self.category}")
            else:
                # Fallback mechanism if MoViAD category string is missing from constants.py
                print(f"[MoE-CLIP] Warning: Category '{self.category}' not found in CLASS_NAMES[{self.dataset}].")
                fallback_class = CLASS_NAMES[self.dataset][0]
                self.cached_class_text_embedding = get_adapted_single_class_text_embedding(
                    text_encoder_model, self.dataset, fallback_class, self.device
                )
                print(f"[MoE-CLIP] Fallback active. Using embedding configuration of: {fallback_class}")

    def forward(self, images: torch.Tensor):
        """
        Forward pass for inference.
        
        Args:
            images: Input images [B, C, H, W]
            
        Returns:
            anomaly maps and scores
        """

        images = images.to(self.device)

        epoch_text_feature = self.cached_class_text_embedding

        patch_features, det_feature, _, _ = self.model(images)

        #Image score estimation
        pred_image = det_feature @ epoch_text_feature
        anomaly_scores = (pred_image[:, 1] + 1) / 2


        #Pixel-level anomaly map estimation
        anomaly_maps_list = []
        for f in patch_features:
            patch_pred = calculate_similarity_map(
                f, 
                epoch_text_feature, 
                self.img_size, 
                test=True, 
                domain=DOMAINS[self.dataset]  
            )
            anomaly_maps_list.append(patch_pred)

        anomaly_maps = torch.cat(anomaly_maps_list, dim=1).sum(1)

        return anomaly_maps, anomaly_scores
        
    def train_step(self, batch: torch.Tensor, training_args: MoECLIPArgs):

        image = batch[0].to(self.device)
        label = batch[1].to(self.device)
        gt_mask = batch[2].to(self.device)
        category = batch[4][0]  # Assuming all samples in the batch belong to the same category

        if self.category != category:
            self.category = category
            self._prepare_text_embeddings()  # Update cached text embeddings for the new category

        epoch_text_feature = self.cached_class_text_embedding

        patch_features, det_feature, aux_loss, special_loss = self.model(image)

        loss = 0.0
        det_feature = det_feature.unsqueeze(1)  # (B,1,D)
        cls_preds = torch.matmul(det_feature, epoch_text_feature)[:, 0]
        loss += F.cross_entropy(cls_preds, label)

        for f in patch_features:
            # f: (B, patch_num, D)
            patch_preds = calculate_similarity_map(f, epoch_text_feature, self.img_size)  # (B,C,H,W)
            # segmentation loss
            loss += calculate_seg_loss(patch_preds, gt_mask)

        loss += aux_loss * self.balance_loss_lambda
        loss += special_loss * self.etf_loss_lambda
        # backward
        training_args.optimizer.zero_grad()
        loss.backward()
        training_args.optimizer.step()
        
        return loss.item()
    
    def train_epoch(self, epoch, train_dataloader, training_args: MoECLIPArgs):

        avg_batch_loss = 0
        
        # train the model
        for batch in tqdm(train_dataloader):
            avg_batch_loss += self.train_step(batch, training_args)

        avg_batch_loss /= len(train_dataloader)

        if training_args.scheduler is not None:
            training_args.scheduler.step()
            
        return avg_batch_loss


    def save(self, save_path: str, training_args: MoECLIPArgs):
        
        checkpoint = {
            "optimizer_state_dict": training_args.optimizer.state_dict(),
            "text_adapter": self.model.text_adapter.state_dict(),
            "image_adapter": self.model.image_adapter.state_dict(),
        }
        
        torch.save(checkpoint, save_path)
    
    def reset_model(self):
        self.model = MoECLIP(
            clip_model=self.clip_model,
            use_paa=self.use_paa,
            seg_proj_sharing_strategy=self.seg_proj_sharing_strategy,
            image_adapt_weight=self.image_adapt_weight,
            moe_r=self.moe_r,
            use_fofs=self.use_fofs,
            moe_lora_alpha=self.moe_lora_alpha,
            moe_num_experts=self.moe_num_experts,
            moe_top_k=self.moe_top_k,
            moe_layers=self.moe_layers,
            relu=self.relu
        ).to(self.device)

        if self.checkpoint_path is not None and os.path.exists(self.checkpoint_path):
            print(f"Loading MoE-CLIP checkpoint from {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            if "text_adapter" in checkpoint:
                self.model.text_adapter.load_state_dict(checkpoint["text_adapter"])
                self.adapt_text = True 
            if "image_adapter" in checkpoint:
                self.model.image_adapter.load_state_dict(checkpoint["image_adapter"])
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    def get_model_size(self):
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
        }
    

    def _extract_trainable_params(self, model):

        for param in model.parameters():
            param.requires_grad = False
        params_to_train = []
        
        for param in model.text_adapter.parameters():
            param.requires_grad = True
        params_to_train.append({"params": model.text_adapter.parameters()})    

        image_params = []
        if self.use_fofs:
            for name, param in model.image_adapter.named_parameters():
                if "lora_A" in name:
                    param.requires_grad = False 
                else:
                    param.requires_grad = True
                    image_params.append(param)

            params_to_train.append({"params": image_params})

            for name, param in model.named_parameters():
                if "lora_A" in name:
                    print(f"{name}: requires_grad={param.requires_grad}")
        else:
            for param in model.image_adapter.parameters():
                param.requires_grad = True
            params_to_train.append({"params": model.image_adapter.parameters()})
        
        return params_to_train