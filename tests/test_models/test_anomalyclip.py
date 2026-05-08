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

    path_to_checkpoint = "/Users/nicolaberti/GitHub/ZSAD-Thesis/external/moviad/src/moviad/models/anomalyclip/AnomalyCLIP_lib/epoch_15_visa.pth"


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
        checkpoint_path= path_to_checkpoint,
        sigma = 4
    )
    model.float()
    model.to(device)
    model.eval()

    results = Evaluator.evaluate(model, test_dataset, metrics=[
            RocAuc(MetricLvl.IMAGE),
            RocAuc(MetricLvl.PIXEL),
            AvgPrec(MetricLvl.IMAGE),
            AvgPrec(MetricLvl.PIXEL),
            F1(MetricLvl.IMAGE),
            F1(MetricLvl.PIXEL),
            ProAuc(MetricLvl.PIXEL),
        ], device=device )
    
    print("Report finale:", results)

    gt_mask, gt_label, pred_anom_map, pred_anom_score = [], [], [], []

    print(f"Inizio test su {len(test_dataset)} campioni...")

    for image, label, mask, path in tqdm(test_dataset, desc="Eval"):

        if len(image.shape) == 3:
            image = image.unsqueeze(0)  # Aggiungiamo la dimensione del batch se necessario

        with torch.no_grad():
            output = model(image)
            
            # Recuperiamo gli output dal dizionario del tuo AnomalyCLIPModel
            anomaly_maps = output['anomaly_maps'] # [B, H, W]
            anomaly_scores = output['anomaly_scores'] # [B]

        gt_mask.append(mask.cpu().numpy().astype(int))
        gt_label.append(label)
        pred_anom_map.append(anomaly_maps.cpu().numpy())
        pred_anom_score.append(anomaly_scores.cpu().numpy())

    # Calcolo dei valori finali
    gt_mask = np.concatenate(gt_mask)
    gt_label = np.array(gt_label)
    pred_anom_map = min_max_norm(np.concatenate(pred_anom_map))
    pred_anom_score = np.concatenate(pred_anom_score)


    metrics=[
            RocAuc(MetricLvl.IMAGE),
            RocAuc(MetricLvl.PIXEL),
            AvgPrec(MetricLvl.IMAGE),
            AvgPrec(MetricLvl.PIXEL),
            F1(MetricLvl.IMAGE),
            F1(MetricLvl.PIXEL),
            ProAuc(MetricLvl.PIXEL),
    ]
    report = {}
    for metric in metrics:
        if metric.level == MetricLvl.IMAGE:
            gt, pred = gt_label, pred_anom_score

            gt = np.expand_dims(gt, axis=1)    # Diventa (50, 1)
            pred = np.expand_dims(pred, axis=1)
        else:
            gt, pred = gt_mask, pred_anom_map
            gt = np.expand_dims(gt, axis=1)    # Diventa (50, 1, 336, 336)
            pred = np.expand_dims(pred, axis=1)

        report[metric.name] = metric.compute(gt, pred)
    
    print("Report finale:", report)

def min_max_norm(x):
    x_min = x.min()
    x_max = x.max()
    denom = x_max - x_min
    if denom == 0:
        return x - x_min # Restituisce un array di zeri della stessa forma
    return (x - x_min) / denom


if __name__ == '__main__':
    test_model_create_train()
    


                                                     