def test_model_create_train():
    from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
    from moviad.models.patchcore.patchcore import PatchCore
    from moviad.models.training_args import TrainingArgs
    from moviad.trainers.trainer import Trainer
    from moviad.datasets.mvtec import MVTecDataset
    from moviad.datasets.cps_ad2d.cpsad2d_dataset import CPSAD2DDataset
    from torch.utils.data import Subset
    from moviad.datasets.dataset_arguments import DatasetArguments
    from moviad.utilities.evaluation.metrics import MetricLvl, RocAuc, AvgPrec, F1, ProAuc
    import torch
    import wandb
    from moviad.utilities.configurations import LabelName

    device = torch.device("cuda::2" if torch.cuda.is_available() else "cpu")

    wandb.init(project="cpsad2d_test", name="patchcore", mode="disabled")

    args = DatasetArguments(
        dataset_path = "/Users/nicolaberti/Documents/Datasets/CPS-AD2D",
        img_size = (270, 480),
        gt_mask_size = (270, 480),
        image_transform_list = None
    )

    train_dataset = CPSAD2DDataset(args, category="cps_1", split="train")
    test_dataset = CPSAD2DDataset(args, category="cps_1", split="test")
    test_dataset = Subset(test_dataset, list(range(0, 100)))  # use a subset for faster testing

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")


    feature_extractor = CustomFeatureExtractor("mobilenet_v2", ["features.3", "features.6", "features.13"], device, frozen=True)    
    model = PatchCore(
        feature_extractor=feature_extractor,
        memory_bank_size=10000,
    )
    model.to(device)

    training_args = TrainingArgs(epochs=2, batch_size=6)

    trainer = Trainer(
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
        logger=None,
        save_path=None,
        saving_criteria=None,
    )

    trainer.train()

if __name__ == '__main__':
    test_model_create_train()
    


                                                     