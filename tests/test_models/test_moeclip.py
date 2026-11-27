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
from torch.utils.data import Subset


OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


def test_moe_clip_model():
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Using device: {device}")

    img_size = 336

    transform_list = [
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD)
    ]

    wandb.init(project="moe_clip_test", entity="test_cpsad2d", name="moe_clip", mode="disabled")


    dataset_path = "/Users/nicolaberti/Documents/Datasets/CPS-AD2D"
    dataset_name = "CPS-AD2D"  
    category_test = "CPS-AD2D" 
    category_train = "CPS-AD2D"

    args = DatasetArguments(
        dataset_path=dataset_path,
        img_size=(img_size, img_size),
        gt_mask_size=(img_size, img_size),
        image_transform_list=transform_list
    )

    # IMPORTANT : the actual training dataset cannot be composed only by normal samples.
    # This is why, to test the training, I am using the test dataset (default is 20).
    # To performe Zero-Shot or Few-Shot perform directly the TEST, TRAINING is used only for finetuning.

    test_dataset = CPSAD2DDataset(args, split="test")
    test_dataset = Subset(test_dataset, list(range(0,3))) # use a subset for faster testing
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
        checkpoint_path= "/Users/nicolaberti/Documents/Datasets/Test/test_save_moeclip.pth", # Inserisci il path di un eventuale .pt/.pth se vuoi fare test da checkpoint
        dataset_train=dataset_name,
        category_train=category_train,
        dataset_test=dataset_name,
        category_test=category_test
    )
    
    model.to(device)

    training_args = MoECLIPArgs(batch_size=1, epochs=1, evaluation_epoch_interval=1)
    training_args.init_train(model)

    trainer = Trainer(
        training_args,
        model,
        test_dataset,
        test_dataset,
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
        save_path= "/Users/nicolaberti/Documents/Datasets/Test/test_save_moeclip.pth",
        saving_criteria=None,
    )

    trainer.train()
    
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