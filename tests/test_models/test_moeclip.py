import os
import torch
import numpy as np
import random
import wandb
from tqdm import tqdm
from torchvision import transforms

from moviad.models.moeclip.moeclip import MoECLIPModel, MoECLIPArgs
from moviad.trainers.trainer_moeclip import MoECLIPTrainer
from moviad.models.training_args import TrainingArgs
from moviad.trainers.trainer import Trainer
from moviad.datasets.cps_ad2d.cpsad2d_dataset import CPSAD2DDataset
from moviad.datasets.miic.miic_dataset import MiicDataset
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.datasets.visa.visa_dataset import VISADataset
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

    wandb.init(project="moe_clip_test", entity="test_cpsad2d", name="moe_clip_miic_aug_518", mode="offline")


    dataset_path = "/home/nicola_berti/big_disk/Datasets/CPS-AD2D"
    dataset_test_name = "CPS-AD2D"  
    category_test_name = "CPS-AD2D" 

    path_miic = "/home/nicola_berti/big_disk/Datasets/MIIC"
    dataset_train_name = "MIIC"
    category_train_name = "MIIC"

    path_mvtec = "/home/nicola_berti/big_disk/Datasets/MVTec"
    dataset_mvtec_name = "MVTec"

    path_visa = "/home/nicola_berti/big_disk/Datasets/Visa"
    dataset_visa_name = "VisA"

    args = DatasetArguments(
        dataset_path=dataset_path,
        img_size=(img_size, img_size),
        gt_mask_size=(img_size, img_size),
        image_transform_list=transform_list
    )

    args_miic = DatasetArguments(
        dataset_path=path_miic,
        img_size=(img_size, img_size),
        gt_mask_size=(img_size, img_size),
        image_transform_list=transform_list
    )

    args_mvtec = DatasetArguments(
        dataset_path=path_mvtec,
        img_size=(img_size, img_size),
        gt_mask_size=(img_size, img_size),
        image_transform_list=transform_list
    )

    args_visa = DatasetArguments(
        dataset_path=path_visa,
        img_size=(img_size, img_size),
        gt_mask_size=(img_size, img_size),
        image_transform_list=transform_list
    )

    train_dataset = MiicDataset(args_miic, split="test")
    '''
    #MVTEC Train dataset
    train_dataset_1 = MVTecDataset(args_mvtec, category="bottle", split="test")
    train_dataset_2 = MVTecDataset(args_mvtec, category="cable", split="test")
    train_dataset_3 = MVTecDataset(args_mvtec, category="capsule", split="test")
    train_dataset_4 = MVTecDataset(args_mvtec, category="carpet", split="test")
    train_dataset_5 = MVTecDataset(args_mvtec, category="grid", split="test")
    train_dataset_6 = MVTecDataset(args_mvtec, category="hazelnut", split="test")
    train_dataset_7 = MVTecDataset(args_mvtec, category="leather", split="test")
    train_dataset_8 = MVTecDataset(args_mvtec, category="metal_nut", split="test")
    train_dataset_9 = MVTecDataset(args_mvtec, category="pill", split="test")
    train_dataset_10 = MVTecDataset(args_mvtec, category="screw",   split="test")
    train_dataset_11 = MVTecDataset(args_mvtec, category="tile",    split="test")
    train_dataset_12 = MVTecDataset(args_mvtec, category="toothbrush", split="test")
    train_dataset_13 = MVTecDataset(args_mvtec, category="transistor", split="test")
    train_dataset_14 = MVTecDataset(args_mvtec, category="wood", split="test")
    train_dataset_15 = MVTecDataset(args_mvtec, category="zipper", split="test")

    
    train_dataset_1 = VISADataset(args_visa, category="candle", split="test")
    train_dataset_2 = VISADataset(args_visa, category="capsules", split="test")
    train_dataset_3 = VISADataset(args_visa, category="cashew", split="test")
    train_dataset_4 = VISADataset(args_visa, category="chewinggum", split="test")
    train_dataset_5 = VISADataset(args_visa, category="fryum", split="test")
    train_dataset_6 = VISADataset(args_visa, category="macaroni1", split="test")
    train_dataset_7 = VISADataset(args_visa, category="macaroni2", split="test")
    train_dataset_8 = VISADataset(args_visa, category="pcb1", split="test")
    train_dataset_9 = VISADataset(args_visa, category="pcb2", split="test")
    train_dataset_10 = VISADataset(args_visa, category="pcb3", split="test")
    train_dataset_11 = VISADataset(args_visa, category="pcb4", split="test")
    train_dataset_12 = VISADataset(args_visa, category="pipe_fryum", split="test")

    
    train_dataset = torch.utils.data.ConcatDataset([
        train_dataset_1,
        train_dataset_2,
        train_dataset_3,
        train_dataset_4,
        train_dataset_5,
        train_dataset_6,
        train_dataset_7,
        train_dataset_8,
        train_dataset_9,
        train_dataset_10,
        train_dataset_11,
        train_dataset_12
    ])
    '''
    

    print(f"Train dataset size: {len(train_dataset)}")
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
        dataset=dataset_train_name
    )
    
    model.to(device)

    training_args = MoECLIPArgs(batch_size=1, epochs=20, evaluation_epoch_interval=21)
    training_args.init_train(model)

    trainer = MoECLIPTrainer(
        training_args,
        model,
        train_dataset,
        eval_dataset=train_dataset,
        metrics=[
            RocAuc(MetricLvl.IMAGE),
            RocAuc(MetricLvl.PIXEL),
            AvgPrec(MetricLvl.IMAGE),
            AvgPrec(MetricLvl.PIXEL),
            F1(MetricLvl.IMAGE),
            F1(MetricLvl.PIXEL),
            ProAuc(MetricLvl.PIXEL),
        ],
        device=device,
        logger=wandb,
        save_path= "/home/nicola_berti/big_disk/MoECLIP_checkpoint/epoch_20_miic_aug.pth",
        saving_criteria=None,
    )

    trainer.train()
    
    adapted_model = MoECLIPModel(
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
        checkpoint_path="/home/nicola_berti/big_disk/MoECLIP_checkpoint/epoch_20_miic_aug.pth",
        dataset=dataset_test_name
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4
    )

    resutls = Evaluator.evaluate(
        model=adapted_model, 
        dataloader=test_loader, 
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
    for metric, value in resutls.items():
        print(f"{metric}: {value:.4f}") 
    
    if wandb:
        wandb.log({
            f"test/{metric_name}": value for metric_name, value in resutls.items()
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