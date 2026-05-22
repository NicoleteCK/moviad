import os
import torch
import numpy as np
import random
import wandb
from tqdm import tqdm
from torchvision import transforms

from moviad.models.moeclip.moeclip import MoECLIPModel, MoECLIPArgs
from moviad.models.training_args import TrainingArgs
from moviad.trainers.trainer import Trainer
from moviad.datasets.cps_ad2d.cpsad2d_dataset import CPSAD2DDataset
from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.utilities.evaluation.metrics import MetricLvl, RocAuc, AvgPrec, F1, ProAuc
from moviad.utilities.evaluation.evaluator import Evaluator


OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


def test_moe_clip_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    img_size = 336

    transform_list = [
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD)
    ]

    wandb.init(project="moe_clip_test", entity="test_cpsas2d", name="moe_clip", mode="disabled")


    dataset_path = "/Users/nicolaberti/Documents/Datasets/CPS-AD2D"
    dataset_name = "CPS-AD2D"  
    category_name = "CPS-AD2D" 

    args = DatasetArguments(
        dataset_path=dataset_path,
        img_size=(img_size, img_size),
        gt_mask_size=(img_size, img_size),
        image_transform_list=transform_list
    )


    test_dataset = CPSAD2DDataset(args, split="test")
    print(f"Test dataset size: {len(test_dataset)}")

    model = MoECLIPModel(
        device=device,
        clip_model_name="ViT-L-14-336",
        img_size=img_size,
        use_paa=True,
        seg_proj_sharing_strategy="shared",
        image_adapt_weight=0.1,
        moe_r=8,
        use_fofs=True,
        moe_lora_alpha=16,
        moe_num_experts=4,
        moe_top_k=2,
        moe_layers=[5, 11, 17, 23],
        relu=False,
        balance_loss_lambda=0.01,
        etf_loss_lambda=0.01,
        checkpoint_path=None, # Inserisci il path di un eventuale .pt/.pth se vuoi fare test da checkpoint
        dataset=dataset_name,
        category=category_name
    )
    
    model.to(device)
    model.eval()

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4
    ) 

    results = Evaluator.evaluate(
        model=model, 
        dataloader=test_dataloader, 
        metrics=[
            RocAuc(MetricLvl.IMAGE),
            RocAuc(MetricLvl.PIXEL),
            AvgPrec(MetricLvl.IMAGE),
            AvgPrec(MetricLvl.PIXEL),
            F1(MetricLvl.IMAGE),
            F1(MetricLvl.PIXEL),
            ProAuc(MetricLvl.PIXEL),
        ], 
        device=device 
    )
    
    print("\nReport finale MoE-CLIP:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    if wandb and wandb.run.mode != "disabled":
        wandb.log({
            f"test/{metric_name}": value for metric_name, value in results.items()
        })


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    setup_seed(42)
    test_moe_clip_model()