import os
import torch
import numpy as np
import random
import wandb
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

from moviad.models.filo.filo import FiLoModel, FiLoArgs
from moviad.models.training_args import TrainingArgs
from moviad.trainers.trainer import Trainer
from moviad.trainers.trainer_filo import FiLoTrainer
from moviad.datasets.fbk_data.fbk_data import FBKDataset
from moviad.datasets.cps_ad2d.cpsad2d_dataset import CPSAD2DDataset
from moviad.datasets.miic.miic_dataset import MiicDataset
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.datasets.visa.visa_dataset import VISADataset
from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.utilities.evaluation.metrics import MetricLvl, RocAuc, AvgPrec, F1, ProAuc
from moviad.utilities.evaluation.evaluator import Evaluator


OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

def test_filo_model():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    img_size = 336

    transform_list = [
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD)
    ]

    wandb.init(project="filo_test", entity="test_cpsad2d", name="miic_wafer_2", mode="offline")


    path_cps = "/home/nicola_berti/example_proj/CPS-AD2D"
    dataset_cps_name = "cps" 

    path_miic = "/home/nicola_berti/example_proj/MIIC"
    dataset_miic_name = "miic"

    path_mvtec = "/home/nicola_berti/big_disk/Datasets/MVTec"
    dataset_mvtec_name = "mvtec"

    path_visa = "/home/nicola_berti/big_disk/Datasets/Visa"
    dataset_visa_name = "visa"

    path_fbk = "/home/nicola_berti/example_proj/SCon_Split_4x4/SCon_Split_4x4"
    fbk_dataset_name = "fbk"

    args = DatasetArguments(
        dataset_path=path_cps,
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

    args_fbk = DatasetArguments(
        dataset_path=path_fbk,
        img_size=(img_size, img_size),
        gt_mask_size=(img_size, img_size),
        image_transform_list=transform_list
    )


    train_dataset = FBKDataset(args_fbk, split="train", wafer_id = "25")
    train_set = MiicDataset(args_miic, split="test")
    test_dataset = FBKDataset(args_fbk, split="test", wafer_id = "25")
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    #train_dataset = CPSAD2DDataset(args, split="train")
    #test_dataset = CPSAD2DDataset(args, split="test")
    #test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)


    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    '''
    model = FiLoModel(
        device=device,
        image_size=img_size,
        feature_list=[6, 12, 18, 24],
        box_threshold=0.25,
        text_threshold=0.25,
        area_threshold=0.7,
        n_ctx=12,
        dataset_name=dataset_miic_name,  
        dino_checkpoint_path="mvtec",          
        filo_checkpoint_path= "/home/nicola_berti/example_proj/filo_miic.pth",
        category="integrated circuit metal layer"           
    )
    model.to(device)

    training_args = FiLoArgs(
        epochs = 15,
        evaluation_epoch_interval = 15,
        train_adapter = False,
        batch_size = 1
    )
    training_args.init_train(model)
    print(f"Verifica finale - train_adapter è: {training_args.train_adapter}")

    print("\n--- Inizio Training  ---")
    
    trainer = FiLoTrainer(
        training_args,
        model,
        train_set,
        test_dataset,
        metrics=[
            RocAuc(MetricLvl.IMAGE),
            #RocAuc(MetricLvl.PIXEL),
            AvgPrec(MetricLvl.IMAGE),
            #AvgPrec(MetricLvl.PIXEL),
            F1(MetricLvl.IMAGE),
            #F1(MetricLvl.PIXEL),
            #ProAuc(MetricLvl.PIXEL),
        ],
        device=device,
        logger=wandb,
        save_path= "/home/nicola_berti/example_proj/filo_miic.pth",
        saving_criteria=None,
    )

    trainer.train()
    '''
    '''
    #Training the adapter
    training_args_adapter = FiLoArgs(
        epochs = 5,
        evaluation_epoch_interval = 15,
        train_adapter = True,
        batch_size = 1
    )

    training_args_adapter.init_train(model)

    trainer_ada = FiLoTrainer(
        training_args_adapter,
        model,
        train_set,
        test_dataset,
        metrics=[
            RocAuc(MetricLvl.IMAGE),
            #RocAuc(MetricLvl.PIXEL),
            AvgPrec(MetricLvl.IMAGE),
            #AvgPrec(MetricLvl.PIXEL),
            F1(MetricLvl.IMAGE),
            #F1(MetricLvl.PIXEL),
            #ProAuc(MetricLvl.PIXEL),
        ],
        device=device,
        logger=wandb,
        save_path= "/home/nicola_berti/example_proj/filo_miic_ada.pth",
        saving_criteria=None,
    )

    trainer_ada.train()
    '''
    k_shots_to_try = [8]

    #train_normal_dataset = CPSAD2DDataset(args, split="train") 

    for tune in ["visa"]:

        print(f"\n--- Inizio Grid Search ---")
        print(f"Valori di K da testare: {k_shots_to_try}")

        if tune == "visa":
            tune_dino = "visa"
        else:
            tune_dino = "mvtec"

        for k in k_shots_to_try:

            model = FiLoModel(
                device=device,
                image_size=img_size,
                category="semiconductor wafer",
                feature_list= [6, 12, 18, 24],
                box_threshold=0.25,
                text_threshold=0.25,
                area_threshold=0.7,
                n_ctx=12,
                dataset_name=fbk_dataset_name,
                dino_checkpoint_path= tune_dino,
                filo_checkpoint_path= "visa",
                few_shot_k = k
            )

            model.to(device)

            #few_shot_indices = list(range(min(k, len(train_normal_dataset))))
            few_shot_indices = random.sample(range(len(train_dataset)), min(k, len(train_dataset)))
            few_shot_subset = Subset(train_dataset, few_shot_indices)
            few_shot_dataloader = DataLoader(few_shot_subset, batch_size=1, shuffle=False, num_workers=2)
            
            if k!= 0:
                model.build_memory_bank(
                    normal_images=few_shot_dataloader,
                    category="semiconductor wafer",
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
                device=device,
                save_dir="/home/nicola_berti/example_proj/FiLo_maps/Wafer_2_k8",
                max_samples_to_save=40,       # Salva i primi 20 campioni
                save_only_anomalous=True 
            )

            print(f"\nReport finale FiLo per few_shot k = {k}, Tuned = {tune}:")
            for metric_name, metric_value in results.items():
                print(f"{metric_name}: {metric_value:.4f}")
            
            if wandb:
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
    test_filo_model()