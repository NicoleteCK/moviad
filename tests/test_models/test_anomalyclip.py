def test_model_create_train():
    from moviad.models.anomalyclip.anomalyclip import AnomalyCLIPModel
    from moviad.models.training_args import TrainingArgs
    from moviad.trainers.trainer import Trainer
    from moviad.datasets.cps_ad2d.cpsad2d_dataset import CPSAD2DDataset
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

    wandb.init(project="anomaly_clip_test", entity = "test_cpsas2d", name="anomaly_clip", mode="disabled")


    args = DatasetArguments(
        dataset_path = "/Users/nicolaberti/Documents/Datasets/CPS-AD2D",
        img_size = (224, 224),
        gt_mask_size = (224, 224),
        image_transform_list = transform_list
    )

    train_dataset = CPSAD2DDataset(args, split="train")
    test_dataset = CPSAD2DDataset(args, split="test")

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")


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
        checkpoint= "mvctec", # or "visa" or None
        sigma = 4
    )
    model.to(device)
    model.eval()

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2
    ) 

    results = Evaluator.evaluate(model, test_dataloader, metrics=[
            RocAuc(MetricLvl.IMAGE),
            RocAuc(MetricLvl.PIXEL),
            AvgPrec(MetricLvl.IMAGE),
            AvgPrec(MetricLvl.PIXEL),
            F1(MetricLvl.IMAGE),
            F1(MetricLvl.PIXEL),
            ProAuc(MetricLvl.PIXEL),
        ], device=device )
    
    print("Report finale:", results)

    if wandb:
        wandb.log({
                f"/test/{metric_name}": value for metric_name, value in results.items()
            })


def setup_seed(seed):
    import torch
    import random
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    setup_seed(42)
    test_model_create_train()
    


                                                     