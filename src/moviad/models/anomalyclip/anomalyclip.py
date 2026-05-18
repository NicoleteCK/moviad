"""
AnomalyCLIP model as a VADModel subclass.
"""

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

# Import from MOVIAD
from moviad.models.vad_model import VADModel
from moviad.models.training_args import TrainingArgs

# Import from AnomalyCLIP
from . import AnomalyCLIP_lib
from .prompt_ensemble import AnomalyCLIP_PromptLearner
from .loss import FocalLoss, BinaryDiceLoss

class AnomalyCLIPArgs(TrainingArgs):

    def init_train(self, model):
        if not hasattr(self, 'optimizer') or self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                list(model.prompt_learner.parameters()),
                lr=0.001,
                betas=(0.5, 0.999)
            )
        
        # Inizializzazione del ReduceLROnPlateau Scheduler
        if not hasattr(self, 'scheduler') or self.scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',       
                factor=0.5,       
                patience=3,       
                verbose=True      
            )
    def __to_dict__(self):
        return {
            "optimizer": self.optimizer_to_dict(self.optimizer) if hasattr(self, 'optimizer') and self.optimizer else None,
            "scheduler": self.scheduler.state_dict() if hasattr(self, 'scheduler') and self.scheduler else None
        }

class AnomalyCLIPModel(VADModel):
    """
    AnomalyCLIP model adapted for MOVIAD framework.
    
    This model implements vision-language anomaly detection using CLIP
    with learnable prompts and dual-path attention mechanism (DPAM).
    """
    
    def __init__(
        self,
        device: torch.device,
        image_size: int = 224,
        features_list: list = [6, 12, 18, 24],
        feature_map_layer: list = [0, 1, 2, 3],
        dpam_layer: int = 20,
        n_ctx: int = 12,
        depth: int = 9,
        t_n_ctx: int = 4,
        pretrained_model: str = "ViT-L/14@336px",
        checkpoint: str | None = None,
        sigma: int = 4
    ):
        """
        Initialize AnomalyCLIP model.
        
        Args:
            device: Device to run the model on
            image_size: Input image size
            features_list: Feature layers to extract from vision encoder
            feature_map_layer: Layers to use for anomaly map generation
            dpam_layer: Number of layers to apply DPAM (Dual-Path Attention Module)
            n_ctx: Number of context tokens for prompt learning
            depth: Depth of learnable text embeddings
            t_n_ctx: Number of learnable text embedding tokens
            pretrained_model: Pretrained CLIP model variant
        """
        super().__init__()
        
        self.device = device
        self.image_size = image_size
        self.features_list = features_list
        self.feature_map_layer = feature_map_layer
        self.dpam_layer = dpam_layer
        self.sigma = sigma
        
        # AnomalyCLIP parameters
        anomalyclip_params = {
            "Prompt_length": n_ctx,
            "learnabel_text_embedding_depth": depth,
            "learnabel_text_embedding_length": t_n_ctx
        }
        
        # Load pretrained CLIP model
        self.model, _ = AnomalyCLIP_lib.load(
            pretrained_model, 
            device=device, 
            design_details=anomalyclip_params
        )
        self.model.eval()
        
        # Initialize prompt learner
        self.prompt_learner = AnomalyCLIP_PromptLearner(
            self.model.to("cpu"), 
            anomalyclip_params
        )

        if checkpoint is not None:
            checkpoint_path = AnomalyCLIP_lib.download_prompt_learner(checkpoint)
            self.load(checkpoint_path)

        self.prompt_learner.to(device)
        self.model.to(device)
        
        # Apply DPAM to vision encoder
        self.model.visual.DAPM_replace(DPAM_layer=dpam_layer)
        
        # Loss functions
        self.loss_focal = FocalLoss()
        self.loss_dice = BinaryDiceLoss()
        self.lam = 4  # Loss scaling factor
        
        # Keep model in eval mode, only train prompt learner
        self.model.eval()
        
    def to(self, device: torch.device):
        """Move model to specified device."""
        super().to(device)
        self.model.to(device)
        self.prompt_learner.to(device)
        self.device = device
        return self
    
    def forward(self, images: torch.Tensor):
        """
        Forward pass for inference.
        
        Args:
            images: Input images [B, C, H, W]
            
        Returns:
            anomaly maps and scores
    """
        if len(images.shape) == 3:
            images = images.unsqueeze(0)  # Add batch dimension if missing

        with torch.no_grad():
            image_features, patch_features = self.model.encode_image(
                images, 
                self.features_list, 
                DPAM_layer=self.dpam_layer
            )
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Get text features from learned prompts
        prompts, tokenized_prompts, compound_prompts_text = self.prompt_learner(cls_id=None)
        text_features = self.model.encode_text_learn(
            prompts, 
            tokenized_prompts, 
            compound_prompts_text
        ).float()
        
        text_features = torch.stack(
            torch.chunk(text_features, dim=0, chunks=2), 
            dim=1
        )
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # Compute image-level anomaly scores
        text_probs = image_features.unsqueeze(1) @ text_features.permute(0, 2, 1)
        text_probs = text_probs[:, 0, ...] / 0.07

        # Generate multi-scale anomaly maps
        similarity_map_list = []
        for idx, patch_feature in enumerate(patch_features):
            if idx >= self.feature_map_layer[0]:
                patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                similarity, _ = AnomalyCLIP_lib.compute_similarity(
                    patch_feature, 
                    text_features[0]
                )
                similarity_map = AnomalyCLIP_lib.get_similarity_map(
                    similarity[:, 1:, :], 
                    self.image_size
                ).permute(0, 3, 1, 2)
                similarity_map_list.append(similarity_map)
        
        if self.prompt_learner.training:
            return text_probs, similarity_map_list
        
        else:
            anomaly_scores = F.softmax(text_probs, dim=-1)[:, 1]

            # Aggregate anomaly maps
            if len(similarity_map_list) > 0:
                anomaly_maps_scaled = []
                for similarity_map in similarity_map_list:
                    am = (similarity_map[:, 1, :, :] + 1 - similarity_map[:, 0, :, :]) / 2.0
                    anomaly_maps_scaled.append(am)
                
                anomaly_map = torch.stack(anomaly_maps_scaled).sum(dim=0)
                # Apply Gaussian smoothing to anomaly map
                anomaly_map = torch.stack([
                    torch.from_numpy(gaussian_filter(i, sigma=self.sigma)) 
                    for i in anomaly_map.detach().cpu()
                ], dim=0)
            else:
                anomaly_map = torch.zeros(images.shape[0], self.image_size, self.image_size, device=self.device)
        
            return anomaly_map, anomaly_scores
        
    def train_step(self, batch: torch.Tensor , training_args: TrainingArgs):
        """
        Single training step.
        
        Args:
            x (torch.Tensor) : batch of images
            training_args: Training arguments
            
        Returns:
            Loss value
        """
        image = batch[0].to(self.device)
        label = batch[1].to(self.device)
        gt = batch[2].to(self.device)
        
        # Binarize ground truth
        gt[gt > 0.5] = 1
        gt[gt <= 0.5] = 0
        
        # Forward pass
        text_probs, similarity_map_list = self(image)

        # Compute image-level anomaly detection loss
        if text_probs.shape[1] == 1:
            text_probs = text_probs.squeeze(1)

        image_loss = F.cross_entropy(text_probs, label.long())
           
        # Compute segmentation losses
        seg_loss = 0
        for similarity_map in similarity_map_list:
            seg_loss += self.loss_focal(similarity_map, gt)
            seg_loss += self.loss_dice(similarity_map[:, 1, :, :], gt)
            seg_loss += self.loss_dice(similarity_map[:, 0, :, :], 1 - gt)
        
        # Total loss
        total_loss = self.lam * seg_loss + image_loss
        
        # Backward pass
        training_args.optimizer.zero_grad()
        total_loss.backward()
        training_args.optimizer.step()
        
        return total_loss.item()
    
    def train_epoch(self, epoch: int, train_dataloader, training_args: TrainingArgs):
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            train_dataloader: Training data loader
            training_args: Training arguments
            
        Returns:
            Average batch loss
        """
        self.model.eval()  # Keep CLIP frozen
        self.prompt_learner.train()  # Only train prompts
        
        avg_batch_loss = 0
        
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
            loss = self.train_step(batch, training_args)
            avg_batch_loss += loss
        
        avg_batch_loss /= len(train_dataloader)
        return avg_batch_loss
    
    
    def reset_model(self):
        """Reset the prompt learner to initial state."""
        # Reinitialize prompt learner
        anomalyclip_params = {
            "Prompt_length": 12,
            "learnabel_text_embedding_depth": 9,
            "learnabel_text_embedding_length": 4
        }
        self.prompt_learner = AnomalyCLIP_PromptLearner(
            self.model.to("cpu"), 
            anomalyclip_params
        )
        self.prompt_learner.to(self.device)
        self.model.to(self.device)
    
    def save_model(self, save_path: str):
        """
        Save the model (only prompt learner parameters).
        
        Args:
            save_path: Path to save the model
        """
        torch.save({
            "prompt_learner": self.prompt_learner.state_dict()
        }, save_path)
    
    def load(self, load_path: str):
        """
        Load the model (only prompt learner parameters).
        
        Args:
            load_path: Path to load the model from
        """
        checkpoint = torch.load(load_path, map_location=torch.device("cpu"))
        self.prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    
    def get_model_size(self):
        """
        Get model size information.
        
        Returns:
            Dictionary with model size details
        """
        # Count parameters in prompt learner (trainable)
        prompt_learner_params = sum(p.numel() for p in self.prompt_learner.parameters())
        
        # Count parameters in CLIP model (frozen)
        clip_params = sum(p.numel() for p in self.model.parameters())
        
        return {
            "prompt_learner_params": prompt_learner_params,
            "clip_model_params": clip_params,
            "total_params": prompt_learner_params + clip_params,
            "trainable_params": prompt_learner_params
        }
    
