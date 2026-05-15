def test_model_create_train():
    from moviad.models.anomalyclip.anomalyclip import AnomalyCLIPModel , AnomalyCLIPArgs
    from moviad.models.training_args import TrainingArgs
    from moviad.trainers.trainer import Trainer
    from moviad.trainers.trainer_anomalyclip import AnomalyCLIPTrainer
    from moviad.datasets.cps_ad2d.cpsad2d_dataset import CPSAD2DDataset
    from moviad.datasets.miic.miic_dataset import MiicDataset
    from moviad.datasets.dataset_arguments import DatasetArguments
    from moviad.utilities.evaluation.metrics import MetricLvl, RocAuc, AvgPrec, F1, ProAuc
    import torch
    import wandb
    from tqdm import tqdm
    from moviad.models.anomalyclip import AnomalyCLIP_lib
    import numpy as np
    from moviad.utilities.evaluation.metrics import MetricLvl
    from torchvision import transforms
    from moviad.models.anomalyclip.AnomalyCLIP_lib import constants
    from moviad.utilities.evaluation.evaluator import Evaluator

   

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform_list = [
      transforms.ToTensor(),
      transforms.Resize((224, 224)),
      transforms.Normalize(mean=constants.OPENAI_DATASET_MEAN, std=constants.OPENAI_DATASET_STD)
      ]

    wandb.init(project="anomaly_clip_test", entity = "test_cpsas2d", name="miic_tuned", mode="disabled")


    args = DatasetArguments(
        dataset_path = "/Users/nicolaberti/Documents/Datasets/CPS-AD2D",
        img_size = (224, 224),
        gt_mask_size = (224, 224),
        image_transform_list = transform_list
    )

    args_miic = DatasetArguments(
        dataset_path = "/Users/nicolaberti/Documents/Datasets/MIIC",
        img_size = (224, 224),
        gt_mask_size = (224, 224),
        image_transform_list = transform_list
    )

    train_dataset = MiicDataset(args_miic, split="test")
    test_dataset = CPSAD2DDataset(args, split="test")


    model = AnomalyCLIPModel(
        device=device,
        image_size= 224,
        features_list=[6, 12, 18, 24],
        feature_map_layer=[0, 1, 2, 3],
        dpam_layer=20,
        n_ctx=12,
        depth=9,
        t_n_ctx=4,
        pretrained_model="ViT-L/14@336px",
        checkpoint= None,
        sigma = 4
    )
    model.float()
    model.to(device)

    training_args = AnomalyCLIPArgs(batch_size=8, epochs=15, evaluation_epoch_interval=15)
    training_args.init_train(model)

    trainer = AnomalyCLIPTrainer(
        training_args,
        model,
        train_dataset,
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
        save_path="/Users/nicolaberti/GitHub/ZSAD-Thesis/external/moviad/src/moviad/models/anomalyclip/AnomalyCLIP_lib/epoch_15_miic.pth",
        saving_criteria=None,
    )

    trainer.train()


if __name__ == '__main__':
    test_model_create_train()
    


                                                     