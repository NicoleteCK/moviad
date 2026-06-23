import os
import copy
import numpy as np
import torch
from addict import Dict

# Grounding DINO
from .FiLo.GroundingDINO.groundingdino.models import build_model
from .FiLo.GroundingDINO.groundingdino.util.slconfig import SLConfig
from .FiLo.GroundingDINO.groundingdino.util.utils import (
    clean_state_dict,
    get_phrases_from_posmap,
)

# Moviad
from moviad.models.vad_model import VADModel
from moviad.models.training_args import TrainingArgs

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import torchvision
import re

import torchvision.transforms as transforms
from sklearn.metrics import (
    auc,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)

import torch

import copy
import torch.nn.functional as F
from .FiLo import vv_open_clip as open_clip
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
from matplotlib import pyplot as plt
from .FiLo.FiLo import FiLo
from .loss import FocalLoss, BinaryDiceLoss
from . import prompt


# ---------------------------------------------------------------------------
# Memory bank for few-shot patch matching
# ---------------------------------------------------------------------------

class MemoryBank:
    """
    Stores patch-level CLIP features extracted from normal reference images.

    One bank is maintained per CLIP stage.  Features are accumulated by calling
    ``update`` and then queried during inference through ``min_cosine_distance``.
    """

    def __init__(self, device: torch.device):
        self.device = device
        # Dict[int -> Tensor(N, C)]  –  stage_index -> stacked normal patches
        self._banks: dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def update(self, stage_idx: int, features: torch.Tensor) -> None:
        """
        Append patch features for *stage_idx*.

        Args:
            stage_idx: Index of the CLIP feature stage (e.g. 6, 12, 18, 24).
            features:  Tensor of shape (B, C, H, W) or (N_patches, C).
                       Will be reshaped to (N_patches, C) automatically.
        """
        if features.dim() == 3:
            B, N, C = features.shape
            features = features.reshape(-1, C)  # (B*N, C)

        features = F.normalize(features.float(), dim=-1).to(self.device)

        if stage_idx in self._banks:
            self._banks[stage_idx] = torch.cat(
                [self._banks[stage_idx], features], dim=0
            )
        else:
            self._banks[stage_idx] = features

    def clear(self) -> None:
        """Remove all stored features."""
        self._banks.clear()

    def is_empty(self) -> bool:
        return len(self._banks) == 0

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def min_cosine_distance(self, stage_idx, query_features):
        if stage_idx not in self._banks:
            raise KeyError(f"No memory bank built for stage {stage_idx}.")
        
        B, N, C = query_features.shape
        q = F.normalize(query_features.float(), dim=-1)  # (B, N, C)
        mem = self._banks[stage_idx]  # (N_mem, C)
        
        sim = torch.einsum("bnc,mc->bnm", q, mem)  # (B, N, N_mem)
        dist = 1.0 - sim
        min_dist = dist.min(dim=-1).values  # (B, N)
        H = W = int(min_dist[:, 1:].shape[1] ** 0.5)
        return min_dist[:, 1:].reshape(B, H, W)


# ---------------------------------------------------------------------------
# Training arguments (unchanged)
# ---------------------------------------------------------------------------

class FiLoArgs(TrainingArgs):

    def __init__(self, train_adapter=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_adapter = train_adapter

    def init_train(self, model):

        if not hasattr(self, 'optimizer_main_part') or self.optimizer_main_part is None:
            main_part_param_groups = [
                {'params': model.filo_model.decoder_cov.parameters(), 'lr': 1e-4},
                {'params': model.filo_model.decoder_linear.parameters(), 'lr': 1e-4},
                {'params': model.filo_model.normal_prompt_learner.parameters(), 'lr': 1e-3},
                {'params': model.filo_model.abnormal_prompt_learner.parameters(), 'lr': 1e-3}
            ]
            self.optimizer_main_part = torch.optim.AdamW(
                main_part_param_groups,
                betas=(0.5, 0.999)
            )

        if not hasattr(self, 'optimizer_adapter') or self.optimizer_adapter is None:
            self.optimizer_adapter = torch.optim.AdamW(
                model.filo_model.adapter.parameters(),
                lr=1e-5,
                betas=(0.5, 0.999)
            )

        if not hasattr(self, 'train_adapter'):
            self.train_adapter = False

    def __to_dict__(self):
        return {
            "optimizer_main_part": self.optimizer_to_dict(self.optimizer_main_part)
                if hasattr(self, 'optimizer_main_part') and self.optimizer_main_part else None,
            "optimizer_adapter": self.optimizer_to_dict(self.optimizer_adapter)
                if hasattr(self, 'optimizer_adapter') and self.optimizer_adapter else None,
            "train_adapter": self.train_adapter
                if hasattr(self, 'train_adapter') else False,
        }


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class FiLoModel(VADModel):

    def __init__(
        self,
        device: torch.device,
        dataset_name: str,
        dino_config_path: str = "./src/moviad/models/filo/FiLo/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        dino_checkpoint_path: str = None,
        filo_checkpoint_path: str = None,
        clip_model: str = "ViT-L-14-336",
        clip_pretrain: str = "openai",
        image_size: int = 518,
        feature_list: list = [6, 12, 18, 24],
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        area_threshold: float = 0.7,
        n_ctx: int = 12,
        # ---- few-shot parameters ----------------------------------------
        few_shot_k: int = 0,
    ):
        """
        Initialize the FiLoModel.

        Args:
            few_shot_k: Number of normal reference images for few-shot mode.
                        0 (default) disables the few-shot branch entirely,
                        keeping identical behaviour to the original model.
                        When > 0, call ``build_memory_bank`` before inference.
        """
        super().__init__()
        self.device = device
        self.dino_config_path = dino_config_path
        self.dino_checkpoint_path = dino_checkpoint_path
        self.filo_checkpoint_path = filo_checkpoint_path
        self.clip_model = clip_model
        self.clip_pretrain = clip_pretrain
        self.image_size = image_size
        self.feature_list = feature_list
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.area_threshold = area_threshold
        self.n_ctx = n_ctx
        self.dataset_name = dataset_name
        self.few_shot_k = few_shot_k
        self.gaussian_filter = torchvision.transforms.GaussianBlur(3, 4.0)
        #self.idx = 0

        # Memory bank (populated lazily via build_memory_bank)
        self.memory_bank = MemoryBank(device=self.device)

        if self.dino_checkpoint_path is not None:
            self.dino_checkpoint_path = download_groundingdino_weights_to_cache(
                self.dino_checkpoint_path
            )

        if self.filo_checkpoint_path is not None:
            self.filo_checkpoint_path = get_filo_weights_cache(self.filo_checkpoint_path)

        self.dino_model = self.load_dino_model(
            self.dino_config_path, self.dino_checkpoint_path, self.device
        )

        self.obj_list = [
            obj.replace(" ", "_")
            for obj in self.get_dataset_object_list(self.dataset_name)
        ]

        filo_args = Dict({
            "clip_model": self.clip_model,
            "clip_pretrain": self.clip_pretrain,
            "image_size": self.image_size,
            "n_ctx": self.n_ctx,
            "features_list": self.feature_list,
            "device": self.device
        })

        self.filo_model = FiLo(
            obj_list=self.obj_list, args=filo_args, device=self.device
        ).to(self.device)

        if self.filo_checkpoint_path is not None and os.path.isfile(
            self.filo_checkpoint_path
        ):
            self.load_filo_checkpoint_complete(
                checkpoint_path=self.filo_checkpoint_path
            )

        self.location = self.generate_location_grid()

    # ------------------------------------------------------------------
    # Few-shot memory bank construction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def build_memory_bank(
        self, normal_images: list[torch.Tensor], category: str
    ) -> None:
        """
        Populate the memory bank from a set of known-normal images.

        This should be called once per category before ``forward`` is used in
        few-shot mode.  The model's CLIP encoder is used to extract per-stage
        patch features; no gradients are computed.

        Args:
            normal_images: List of image tensors, each with shape (1, C, H, W)
                           or (C, H, W).  Passed one-by-one through the encoder.
            category:      Category name (same string passed to ``forward``).

        Example::

            model.build_memory_bank(
                normal_images=[img1, img2, img3],
                category="bottle",
            )
            anom_map, anom_score = model(query_image, category="bottle")
        """
        if not normal_images:
            raise ValueError("normal_images must contain at least one image.")

        self.memory_bank.clear()
        cls_name = category if isinstance(category, str) else category[0]
        print(
            f"Building few-shot memory bank for '{cls_name}' "
            f"using {len(normal_images)} reference image(s)…"
        )

        self.filo_model.eval()
        for img in tqdm(normal_images, desc="Encoding normal references"):
            if img.dim() == 3:
                img = img.unsqueeze(0)
            img = img.to(self.device)

            items = {"img": img, "cls_name": cls_name}
            # extract_patch_features must return a list of (B, C, H, W) tensors,
            # one per feature stage listed in self.feature_list.
            patch_features = self._extract_patch_features(items)

            for stage_idx, feats in zip(self.feature_list, patch_features):
                self.memory_bank.update(stage_idx, feats)

        print(
            f"Memory bank built: "
            + ", ".join(
                f"stage {s}: {self.memory_bank._banks[s].shape[0]} patches"
                for s in self.feature_list
                if s in self.memory_bank._banks
            )
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _extract_patch_features(
        self, items: dict
    ) -> list[torch.Tensor]:
        """
        Extract per-stage CLIP patch features without running the full FiLo
        decoder.

        Returns a list of tensors, one per stage in ``self.feature_list``,
        each with shape (B, C, H_i, W_i).

        The implementation delegates to FiLo's internal CLIP encoder.
        Adjust the call below if your FiLo version exposes a different API.
        """
        # FiLo stores intermediate feature maps in a dict keyed by layer index.
        # We call the CLIP image encoder and capture the requested layers.
        image = items["img"]  # (B, 3, H, W)

        # filo_model.clip_model is an open_clip VisualTransformer.
        # encode_image_features returns a list of feature maps in layer order
        # when feature_list is provided; adapt if your FiLo fork differs.
        _ , patch_features = self.filo_model.clip_model.encode_image(
            image, out_layers=self.feature_list
        )

        return patch_features 

    def _compute_fewshot_map(
        self, items: dict, target_size: tuple[int, int]
    ) -> torch.Tensor:
        """
        Compute the few-shot anomaly localisation map M^few (eq. 7).

        For each CLIP stage i:
          M^few_i = min( cos_distance(P_i, Mem_i) )           (eq. 6)

        The per-stage maps are upsampled to *target_size*, summed, and
        min-max normalised.

        Args:
            items:       Dict with keys "img" and "cls_name".
            target_size: (H, W) of the output map.

        Returns:
            Tensor (B, 1, H, W) – normalised few-shot anomaly map.
        """
        patch_features = self._extract_patch_features(items)

        H_out, W_out = target_size
        B = patch_features[0].shape[0]
        accumulated = torch.zeros(B, H_out, W_out, device=self.device)

        for stage_idx, feats in zip(self.feature_list, patch_features):
            # (B, H_i, W_i)
            dist_map = self.memory_bank.min_cosine_distance(stage_idx, feats)

            # Upsample to target spatial resolution
            dist_map_up = F.interpolate(
                dist_map.unsqueeze(1).float(),
                size=(H_out, W_out),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)  # (B, H_out, W_out)

            accumulated += dist_map_up

        # Min-max normalisation over spatial dimensions (eq. 7 "Norm")
        flat = accumulated.reshape(B, -1)
        mins = flat.min(dim=-1).values[:, None, None]
        maxs = flat.max(dim=-1).values[:, None, None]
        eps = 1e-8
        normalised = (accumulated - mins) / (maxs - mins + eps)

        return normalised.unsqueeze(1)  # (B, 1, H, W)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, image, category=None, **kwargs):

        cls_name = category if isinstance(category, str) else category[0]

        text_prompt = self.make_text_prompt(self.dataset_name, category)

        boxes_filt, pred_phrases = self.get_grounding_output(
            model=self.dino_model,
            image=image,
            caption=text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            category=category,
            device=self.device,
            area_thr=self.area_threshold,
        )

        boxes_filt_copy = copy.deepcopy(boxes_filt)
        if self.dataset_name not in prompt.dataset_anomaly_maps:
            raise ValueError(
                f"Dataset '{self.dataset_name}' non configurato in dataset_anomaly_maps."
            )

        current_anomaly_detail = prompt.dataset_anomaly_maps[self.dataset_name]

        for i in range(boxes_filt.size(0)):
            allowed_elements = (
                current_anomaly_detail[cls_name] + prompt.anomaly_status_general
            )

            if not check_elements_in_array(allowed_elements, pred_phrases[i]):
                pred_phrases[i] += "#$%"
                continue

            boxes_filt_copy[i] = boxes_filt_copy[i] * torch.Tensor(
                [self.image_size] * 4
            )
            boxes_filt_copy[i][:2] -= boxes_filt_copy[i][2:] / 2
            boxes_filt_copy[i][2:] += boxes_filt_copy[i][:2]

        boxes_filt = boxes_filt_copy.cpu()

        position = []
        max_box = None
        max_pred = 0
        for i in range(boxes_filt.size(0)):
            if "#$%" in pred_phrases[i]:
                continue
            number = float(re.search(r"\((.*?)\)", pred_phrases[i]).group(1))
            if number >= max_pred:
                max_box = boxes_filt[i]
                max_pred = number
        if max_box is not None:
            center = (
                (max_box[0] + max_box[2]) / 2,
                (max_box[1] + max_box[3]) / 2,
            )
        else:
            center = self.image_size / 2, self.image_size / 2
        for region, ((x1, y1), (x2, y2)) in self.location.items():
            if x1 <= center[0] <= x2 and y1 <= center[1] <= y2:
                position.append(region)
                break

        # -------------------------------------------------------------------
        # VL branch (original FiLo text-image anomaly localisation)
        # -------------------------------------------------------------------
        items = {"img": image, "cls_name": cls_name}

        text_probs, anomaly_maps = self.filo_model(
            items, with_adapter=True, positions=position
        )

        for i in range(len(anomaly_maps)):
            anomaly_maps[i] = self.gaussian_filter(
                (anomaly_maps[i][:, 1, :, :] - anomaly_maps[i][:, 0, :, :] + 1) / 2
            )

        # M^vl:  (B, 1, H, W)
        anomaly_map_vl = torch.mean(
            torch.stack(anomaly_maps, dim=0), dim=0
        ).unsqueeze(1)

        # Bounding-box suppression (score boost inside boxes)
        anomaly_score_vl = anomaly_map_vl.clone()
        for rect in boxes_filt:
            lx = int(rect[0].item())
            ly = int(rect[1].item())
            rx = int(rect[2].item())
            ry = int(rect[3].item())
            anomaly_score_vl[:, :, ly:ry, lx:rx] = 1

        anomaly_score_vl = torch.where(
            anomaly_score_vl == 1,
            anomaly_map_vl,
            anomaly_map_vl * 0.7,
        )

        # -------------------------------------------------------------------
        # Few-shot branch (patch matching, eq. 6-7)
        # -------------------------------------------------------------------
        use_fewshot = self.few_shot_k > 0 and not self.memory_bank.is_empty()

        if use_fewshot:
            _, _, H_vl, W_vl = anomaly_map_vl.shape

            with torch.no_grad():
                # M^few: (B, 1, H, W)
                fewshot_map = self._compute_fewshot_map(
                    items, target_size=(H_vl, W_vl)
                )

            # Position-enhanced suppression: zero out regions outside DINO boxes
            # (same suppression strategy used for the VL branch)
            suppression_mask = torch.zeros_like(fewshot_map)
            for rect in boxes_filt:
                lx = int(rect[0].item())
                ly = int(rect[1].item())
                rx = int(rect[2].item())
                ry = int(rect[3].item())
                suppression_mask[:, :, ly:ry, lx:rx] = 1.0

            # Inside DINO boxes: keep score; outside: attenuate by 0.7
            fewshot_map = torch.where(
                suppression_mask == 1,
                fewshot_map,
                fewshot_map * 0.7,
            )

            # -----------------------------------------------------------
            # Fusion (eq. 8):  M = G_σ( (M^vl + M^few) / 2 )
            # -----------------------------------------------------------
            combined = (anomaly_map_vl + fewshot_map) / 2.0
            anom_maps_out = self.gaussian_filter(combined.squeeze(1))  # (B, H, W)

            # Image-level score: max of the fused map (with box boosting)
            fused_score = torch.where(
                suppression_mask == 1,
                combined,
                combined * 0.7,
            )
            anom_scores_out = (
                fused_score.max(dim=-1)[0].max(dim=-1)[0].squeeze(-1)
            )

        else:
            # -----------------------------------------------------------
            # Zero-shot only (original behaviour)
            # -----------------------------------------------------------
            anom_maps_out = anomaly_map_vl.squeeze(1)  # (B, H, W)
            anom_scores_out = (
                anomaly_score_vl.max(dim=-1)[0].max(dim=-1)[0].squeeze(-1)
            )

        return anom_maps_out, anom_scores_out

    # ------------------------------------------------------------------
    # Training (unchanged from original)
    # ------------------------------------------------------------------

    def train_step(self, batch: dict, training_args: FiLoArgs):
        """Perform a single training step on a batch."""

        loss_focal = FocalLoss()
        loss_dice = BinaryDiceLoss()

        image = batch[0].to(self.device)
        label = batch[1].to(self.device)
        cls_name = batch[4][0]

        items = {"img": image, "cls_name": cls_name}

        if not training_args.train_adapter:
            text_probs, anomaly_maps = self.filo_model(items, with_adapter=False)

            gt = batch[2].squeeze().to(self.device)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            loss = 0
            for num in range(len(anomaly_maps)):
                loss += loss_focal(anomaly_maps[num], gt)
                loss += loss_dice(anomaly_maps[num][:, 1, :, :], gt)
                loss += loss_dice(anomaly_maps[num][:, 0, :, :], 1 - gt)

            training_args.optimizer_main_part.zero_grad()
            loss.backward()
            training_args.optimizer_main_part.step()

        else:
            text_probs, anomaly_maps = self.filo_model(
                items, only_train_adapter=True, with_adapter=True
            )

            text_probs = text_probs[:, 0, ...] / 0.07
            loss = F.cross_entropy(
                text_probs.squeeze(), label[0].to(self.device)
            )

            training_args.optimizer_adapter.zero_grad()
            loss.backward()
            training_args.optimizer_adapter.step()

        return loss.item()

    def train_epoch(self, epoch: int, train_dataloader, training_args: FiLoArgs):
        """Train the model for one epoch and return the average batch loss."""

        avg_batch_loss = 0.0
        phase = "Adapter" if training_args.train_adapter else "Main"

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch} [{phase}]"):
            avg_batch_loss += self.train_step(batch, training_args)

        avg_batch_loss /= len(train_dataloader)
        return avg_batch_loss

    # ------------------------------------------------------------------
    # Utility methods (unchanged from original)
    # ------------------------------------------------------------------

    def reset_model(self):
        self.filo_model = FiLo(
            obj_list=self.obj_list,
            args=Dict({
                "clip_model": self.clip_model,
                "clip_pretrain": self.clip_pretrain,
                "image_size": self.image_size,
                "n_ctx": self.n_ctx,
                "features_list": self.feature_list,
                "device": self.device,
            }),
            device=self.device,
        ).to(self.device)

        self.dino_model = self.load_dino_model(
            self.dino_config_path, self.dino_checkpoint_path, self.device
        )
        self.memory_bank.clear()

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        checkpoint = {"filo": self.filo_model.state_dict()}
        torch.save(checkpoint, save_path)
        print(f"Model saved at: {save_path}")

    def get_model_size(self) -> dict:
        total_params = sum(p.numel() for p in self.filo_model.parameters())
        trainable_params = sum(
            p.numel() for p in self.filo_model.parameters() if p.requires_grad
        )
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }

    def load_dino_model(self, config_path, checkpoint_path, device):
        args = SLConfig.fromfile(config_path)
        args.device = device
        model = build_model(args)
        if checkpoint_path is not None and os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            state_dict = checkpoint.get("model", checkpoint)
            cleaned_state_dict = {
                (k[7:] if k.startswith("module.") else k): v
                for k, v in state_dict.items()
            }
            load_res = model.load_state_dict(cleaned_state_dict, strict=False)
            print(load_res)
        _ = model.eval()
        return model

    def get_dataset_object_list(self, dataset_name):
        mapping = {
            "mvtec": prompt.mvtec_obj_list,
            "visa":  prompt.visa_obj_list,
            "cps":   prompt.cps_obj_list,
            "miic":  prompt.miic_obj_list,
        }
        if dataset_name not in mapping:
            raise ValueError("Unsupported dataset name: {}".format(dataset_name))
        return mapping[dataset_name]

    def make_text_prompt(self, dataset_name, category):
        if isinstance(category, (tuple, list)):
            cls_name = category[0]
        else:
            cls_name = category

        mapping = {
            "mvtec": prompt.mvtec_anomaly_detail_gpt,
            "visa":  prompt.visa_anomaly_detail_gpt,
            "cps":   prompt.cps_anomaly_detail_gpt,
            "miic":  prompt.miic_anomaly_detail_gpt,
        }
        if dataset_name not in mapping:
            raise ValueError("Unsupported dataset name: {}".format(dataset_name))
        return " . ".join(
            prompt.anomaly_status_general + mapping[dataset_name][cls_name]
        )

    def get_grounding_output(
        self,
        model,
        image,
        caption,
        box_threshold,
        text_threshold,
        category,
        with_logits=True,
        device="cpu",
        area_thr=0.8,
    ):
        caption = caption.lower().strip()
        if not caption.endswith("."):
            caption = caption + "."
        model = model.to(device)
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image, captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]
        boxes = outputs["pred_boxes"].cpu()[0]

        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        boxes_area = boxes_filt[:, 2] * boxes_filt[:, 3]
        filt_mask = torch.bitwise_and(
            (logits_filt.max(dim=1)[0] > box_threshold), (boxes_area < area_thr)
        )

        if torch.sum(filt_mask) == 0:
            filt_mask = torch.argmax(logits_filt.max(dim=1)[0])
            logits_filt = logits_filt[filt_mask].unsqueeze(0)
            boxes_filt = boxes_filt[filt_mask].unsqueeze(0)
        else:
            logits_filt = logits_filt[filt_mask]
            boxes_filt = boxes_filt[filt_mask]

        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)

        pred_phrases = []
        boxes_filt_category = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(
                logit > text_threshold, tokenized, tokenlizer
            )
            if with_logits:
                pred_phrases.append(
                    pred_phrase + f"({str(logit.max().item())[:4]})"
                )
            else:
                pred_phrases.append(pred_phrase)
            boxes_filt_category.append(box)
        boxes_filt_category = torch.stack(boxes_filt_category, dim=0)

        return boxes_filt_category, pred_phrases

    def generate_location_grid(self) -> dict:
        height, width = self.image_size, self.image_size
        x1, x2 = width // 3, (width // 3) * 2
        y1, y2 = height // 3, (height // 3) * 2
        return {
            "top left":     [(0, 0), (x1, y1)],
            "top":          [(x1 + 1, 0), (x2, y1)],
            "top right":    [(x2 + 1, 0), (width, y1)],
            "left":         [(0, y1 + 1), (x1, y2)],
            "center":       [(x1 + 1, y1 + 1), (x2, y2)],
            "right":        [(x2 + 1, y1 + 1), (width, y2)],
            "bottom left":  [(0, y2 + 1), (x1, height)],
            "bottom":       [(x1 + 1, y2 + 1), (x2, height)],
            "bottom right": [(x2 + 1, y2 + 1), (width, height)],
        }

    def load_filo_checkpoint_complete(self, checkpoint_path) -> dict:
        if checkpoint_path is None or not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Specified checkpoint file not found: {checkpoint_path}"
            )

        print(f"Loading FiLo checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "filo" in checkpoint:
            state_dict = checkpoint["filo"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        clip_state_dict = {
            k.replace("clip_model.", ""): v
            for k, v in state_dict.items()
            if k.startswith("clip_model.")
        }

        print("Executing bicubic interpolation on CLIP position embeddings…")
        open_clip.model.resize_pos_embed(
            state_dict=clip_state_dict,
            model=self.filo_model.clip_model,
            interpolation="bicubic",
        )

        for k, v in clip_state_dict.items():
            state_dict[f"clip_model.{k}"] = v

        load_res = self.filo_model.load_state_dict(state_dict, strict=False)
        print("Checkpoint loading completed successfully!")
        print("PyTorch outcome:", load_res)
        return load_res
    
    def visualize_predictions(
        self,
        image: torch.Tensor,
        gt_mask: torch.Tensor,
        anom_map: torch.Tensor,
        boxes_filt: torch.Tensor,
        pred_phrases: list[str],
        save_path: str,
        alpha: float = 0.5,
    ) -> None:
        """
        Salva una figura con 4 pannelli:
        1) Immagine originale
        2) Ground truth mask
        3) Anomaly map (heatmap)
        4) Immagine con bounding boxes e label DINO

        Args:
            image:        (1, 3, H, W) o (3, H, W) – tensor normalizzato.
            gt_mask:      (H, W) o (1, H, W)        – maschera binaria GT.
            anom_map:     (H, W) o (1, H, W)        – mappa anomalia in [0,1].
            boxes_filt:   (N, 4) pixel-space [x1,y1,x2,y2], già scalate.
            pred_phrases: lista di N stringhe (può contenere '#$%' per box scartate).
            save_path:    percorso file output (es. "output/result.png").
            alpha:        trasparenza overlay heatmap (default 0.5).
        """
        import matplotlib.patches as mpatches
        from matplotlib.colors import Normalize

        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

        # --- Normalizzazione tensori → numpy ---
        def to_np_image(t):
            t = t.squeeze().cpu().float()
            if t.dim() == 3:                        # (3, H, W)
                t = t.permute(1, 2, 0)             # → (H, W, 3)
                # de-normalizza se necessario (clip a [0,1])
                t = (t - t.min()) / (t.max() - t.min() + 1e-8)
            return t.numpy()

        img_np   = to_np_image(image)               # (H, W, 3)
        gt_np    = gt_mask.squeeze().cpu().numpy()  # (H, W)
        amap_np  = anom_map.squeeze().cpu().float().numpy()  # (H, W)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # --- Pannello 1: immagine originale ---
        axes[0].imshow(img_np)
        axes[0].set_title("Input Image", fontsize=12)
        axes[0].axis("off")

        # --- Pannello 2: ground truth ---
        axes[1].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Ground Truth", fontsize=12)
        axes[1].axis("off")

        # --- Pannello 3: anomaly map heatmap ---
        axes[2].imshow(img_np)
        im = axes[2].imshow(amap_np, cmap="jet", alpha=alpha,
                            norm=Normalize(vmin=0, vmax=1))
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        axes[2].set_title("Anomaly Map", fontsize=12)
        axes[2].axis("off")

        # --- Pannello 4: bounding boxes + label ---
        axes[3].imshow(img_np)
        colors = plt.cm.get_cmap("tab10").colors
        valid_idx = 0
        for i, (box, phrase) in enumerate(zip(boxes_filt, pred_phrases)):
            if "#$%" in phrase:
                continue
            x1, y1, x2, y2 = box[0].item(), box[1].item(), box[2].item(), box[3].item()
            color = colors[valid_idx % len(colors)]
            rect = mpatches.FancyBboxPatch(
                (x1, y1), x2 - x1, y2 - y1,
                boxstyle="round,pad=2",
                linewidth=2,
                edgecolor=color,
                facecolor="none"
            )
            axes[3].add_patch(rect)
            # Label sopra la box
            label = phrase.replace("#$%", "").strip()
            axes[3].text(
                x1, max(y1 - 5, 0), label,
                color="white", fontsize=8, fontweight="bold",
                bbox=dict(facecolor=color, alpha=0.7, pad=2, edgecolor="none")
            )
            valid_idx += 1
        axes[3].set_title("Detected Anomalies", fontsize=12)
        axes[3].axis("off")

        plt.suptitle(
            f"Anomaly Score: {anom_map.squeeze().max().item():.4f}",
            fontsize=13, fontweight="bold", y=1.01
        )
        plt.tight_layout()
        full_save_path = os.path.join(save_path, f"{self.idx}.png")
        self.idx += 1
        plt.savefig(full_save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved visualization → {full_save_path}")


# ---------------------------------------------------------------------------
# Standalone helpers (unchanged from original)
# ---------------------------------------------------------------------------

def check_elements_in_array(arr1, arr2):
    for elem in arr1:
        if elem in arr2:
            return True
    return False


def download_groundingdino_weights_to_cache(path_or_dataset: str = "original") -> str:
    if os.path.exists(path_or_dataset) and os.path.isfile(path_or_dataset):
        print(f"Using provided path: {path_or_dataset}")
        return os.path.abspath(path_or_dataset)

    cache_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
    os.makedirs(cache_dir, exist_ok=True)

    WEIGHTS_URLS = {
        "mvtec":    "https://huggingface.co/FantasticGNU/FiLo/resolve/main/grounding_train_on_mvtec.pth",
        "visa":     "https://huggingface.co/FantasticGNU/FiLo/resolve/main/grounding_train_on_visa.pth",
        "original": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
    }

    dataset_key = path_or_dataset.lower().strip()
    if dataset_key not in ["mvtec", "visa"]:
        dataset_key = "original"

    url = WEIGHTS_URLS[dataset_key]
    cached_path = os.path.join(cache_dir, os.path.basename(url))

    if not os.path.exists(cached_path):
        try:
            torch.hub.download_url_to_file(url, cached_path, progress=True)
        except Exception as e:
            raise RuntimeError(
                f"Cannot download weights for '{dataset_key}' from {url}. Error: {e}"
            )

    return os.path.abspath(cached_path)


def get_filo_weights_cache(path_or_dataset: str) -> str:
    HF_WEIGHTS_URLS = {
        "mvtec": "https://huggingface.co/FantasticGNU/FiLo/resolve/main/filo_train_on_mvtec.pth",
        "visa":  "https://huggingface.co/FantasticGNU/FiLo/resolve/main/filo_train_on_visa.pth",
    }

    if os.path.exists(path_or_dataset) and os.path.isfile(path_or_dataset):
        return os.path.abspath(path_or_dataset)

    dataset_key = path_or_dataset.lower().strip()
    if dataset_key not in HF_WEIGHTS_URLS:
        raise ValueError(
            f"Argument '{path_or_dataset}' must be a valid path or 'mvtec'/'visa'."
        )

    torch_cache_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
    os.makedirs(torch_cache_dir, exist_ok=True)

    destination_path = os.path.join(
        torch_cache_dir, f"filo_train_on_{dataset_key}.pth"
    )

    if os.path.exists(destination_path):
        return os.path.abspath(destination_path)

    download_url = HF_WEIGHTS_URLS[dataset_key]
    try:
        torch.hub.download_url_to_file(download_url, destination_path, progress=True)
    except Exception as e:
        raise RuntimeError(
            f"Cannot download FiLo weights for {dataset_key}. Error: {e}"
        )

    return os.path.abspath(destination_path)