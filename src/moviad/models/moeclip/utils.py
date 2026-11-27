import os
import random
import numpy as np
import torch
from torch.nn import functional as F
import kornia as K
from torchvision import transforms

import cv2
from kornia.filters import gaussian_blur2d
from moviad.models.moeclip.constants import CLASS_NAMES, REAL_NAMES, PROMPTS
from moviad.backbones.clip.model_load import tokenize
from sklearn.metrics import precision_recall_curve
from moviad.common.common_losses import FocalLoss, BinaryDiceLoss
from scipy.ndimage import gaussian_filter


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def get_rot_mat(theta):
    theta = torch.tensor(theta)
    return torch.tensor(
        [
            [torch.cos(theta), -torch.sin(theta), 0],
            [torch.sin(theta), torch.cos(theta), 0],
        ]
    )


def get_translation_mat(a, b):
    return torch.tensor([[1, 0, a], [0, 1, b]])


def rot_img(x, scale):
    theta = scale
    dtype = torch.FloatTensor
    if x.dim() == 3:
        x = x.unsqueeze(0)
    rot_mat = get_rot_mat(theta)[None, ...].type(dtype).repeat(x.shape[0], 1, 1)
    grid = F.affine_grid(rot_mat, x.size()).type(dtype)
    x = F.grid_sample(x, grid, padding_mode="reflection")
    x = x.squeeze(0)
    return x


def translation_img(x, translation):
    a, b = translation
    dtype = torch.FloatTensor
    if x.dim() == 3:
        x = x.unsqueeze(0)
    rot_mat = get_translation_mat(a, b)[None, ...].type(dtype).repeat(x.shape[0], 1, 1)
    grid = F.affine_grid(rot_mat, x.size()).type(dtype)
    x = F.grid_sample(x, grid, padding_mode="reflection")
    x = x.squeeze(0)
    return x


def hflip_img(x, **kwargs):
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = K.geometry.transform.hflip(x)
    x = x.squeeze(0)
    return x


def vflip_img(x, **kwargs):
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = K.geometry.transform.vflip(x)
    x = x.squeeze(0)
    return x


def add_gaussian_noise(x, scale=0.05):
    std = scale
    noise_mask = torch.randn(x.shape[-2:]) > 3
    noise = torch.randn_like(x) * std  # mean = 0
    noised_img = (x + noise) * noise_mask
    noise_img = torch.where(noised_img > 0, noised_img, x)
    return noise_img


def cos_sim(a_norm, b_norm):
    if len(a_norm.shape) == 2:
        sim_mt = b_norm @ a_norm.transpose(1, 0)
    elif len(a_norm.shape) == 1:
        sim_mt = b_norm @ a_norm
    else:
        raise NotImplementedError
    return sim_mt

class AddGaussianNoise(object):
    def __init__(self, std=1.0, p=0.5):
        self.std = std
        self.p = p

    def __call__(self, x):

        if random.random() < self.p:
            return x
        if not isinstance(x, torch.Tensor):
            x = transforms.ToTensor()(x)
        noise_mask = (torch.randn(x.shape[-2:]) > 3).int()
        noise = torch.randn_like(x) * self.std  # mean = 0
        noised_img = (1 - noise_mask) * x + noise * x * noise_mask
        noised_img = torch.clamp(noised_img, 0.0, 1.0)
        return transforms.ToPILImage()(noised_img)

    def __repr__(self):
        return self.__class__.__name__ + f"(std={self.std}, p={self.p})"


# FORWARD UTILS

# ================================================================================================
# The following code is used to get adapted text embeddings
prompt = PROMPTS
prompt_normal = prompt["prompt_normal"]
prompt_abnormal = prompt["prompt_abnormal"]
prompt_state = [prompt_normal, prompt_abnormal]
prompt_templates = prompt["prompt_templates"]


def get_adapted_single_class_text_embedding(model, dataset_name, class_name, device):
    if class_name == "object":
        real_name = class_name
    else:
        assert class_name in CLASS_NAMES[dataset_name], (
            f"class_name {class_name} not found; available class_names: {CLASS_NAMES[dataset_name]}"
        )
        real_name = REAL_NAMES[dataset_name][class_name]
    text_features = []
    for i in range(len(prompt_state)):
        prompted_state = [state.format(real_name) for state in prompt_state[i]]
        prompted_sentence = []
        for s in prompted_state:
            for template in prompt_templates:
                prompted_sentence.append(template.format(s))
        prompted_sentence = tokenize(prompted_sentence).to(device)
        class_embeddings = model.encode_text(prompted_sentence)
        class_embeddings = class_embeddings / class_embeddings.norm(
            dim=-1, keepdim=True
        )
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding = class_embedding / class_embedding.norm()
        text_features.append(class_embedding)
    text_features = torch.stack(text_features, dim=1).to(device)
    return text_features


def get_adapted_single_sentence_text_embedding(model, dataset_name, class_name, device):
    assert class_name in CLASS_NAMES[dataset_name], (
        f"class_name {class_name} not found; available class_names: {CLASS_NAMES[dataset_name]}"
    )
    real_name = REAL_NAMES[dataset_name][class_name]
    text_features = []
    for i in range(len(prompt_state)):
        prompted_state = [state.format(real_name) for state in prompt_state[i]]
        prompted_sentence = []
        for s in prompted_state:
            for template in prompt_templates:
                prompted_sentence.append(template.format(s))
        prompted_sentence = tokenize(prompted_sentence).to(device)
        class_embeddings = model.encode_text(prompted_sentence)
        class_embeddings = F.normalize(class_embeddings, dim=-1)
        text_features.append(class_embeddings)
    text_features = torch.cat(text_features, dim=0).to(device)
    return text_features


def get_adapted_text_embedding(model, dataset_name, device):
    ret_dict = {}
    for class_name in CLASS_NAMES[dataset_name]:
        text_features = get_adapted_single_class_text_embedding(
            model, dataset_name, class_name, device
        )
        ret_dict[class_name] = text_features
    return ret_dict


# ================================================================================================
def calculate_similarity_map(
    patch_features, epoch_text_feature, img_size, test=False, domain="Medical"
):
    patch_anomaly_scores = 100.0 * torch.matmul(patch_features, epoch_text_feature)
    B, L, C = patch_anomaly_scores.shape
    H = int(np.sqrt(L))
    patch_pred = patch_anomaly_scores.permute(0, 2, 1).view(B, C, H, H)
    if test:
        assert C == 2
        sigma = 1 if domain == "Industrial" else 1.5
        kernel_size = 7 if domain == "Industrial" else 9
        patch_pred = (patch_pred[:, 1] + 1 - patch_pred[:, 0]) / 2
        patch_pred = gaussian_blur2d(
            patch_pred.unsqueeze(1), (kernel_size, kernel_size), (sigma, sigma)
        )
    patch_preds = F.interpolate(
        patch_pred, size=img_size, mode="bilinear", align_corners=True
    )
    if not test and C > 1:
        patch_preds = torch.softmax(patch_preds, dim=1)
    return patch_preds


focal_loss = FocalLoss()
dice_loss = BinaryDiceLoss()


def calculate_seg_loss(patch_preds, mask):
    loss = focal_loss(patch_preds, mask)
    loss += dice_loss(patch_preds[:, 0, :, :], 1 - mask)
    loss += dice_loss(patch_preds[:, 1, :, :], mask)
    return loss


# ================================================================================================

def extract_variant_from_path(path: str) -> str:
    parts = path.strip().split('/')
    return parts[-1] if len(parts) == 1 else parts[-1] if '.png' in parts[-1] and len(parts) == 1 else parts[0] if parts[0] else parts[1] if parts[0]=='' else parts[0]

def group_indices_by_variant(file_names: list) -> dict:
    variant_map = {}
    for idx, fn in enumerate(file_names):
        variant = extract_variant_from_path(fn)
        variant_map.setdefault(variant, []).append(idx)
    return variant_map

def normalize(pred):
    if (pred.max() - pred.min()) == 0:
        return np.zeros_like(pred)
    return (pred - pred.min()) / (pred.max() - pred.min())


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)

    scoremap = np.squeeze(scoremap)
    if scoremap.ndim != 2:
        raise ValueError(f"Expected 2D scoremap, got shape {scoremap.shape}")

    scoremap = normalize(scoremap) 
    scoremap = (scoremap * 255).astype(np.uint8)

    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)

    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)


def he_cheng(img_list, size=256):
    h, w, c = img_list[0].shape
    pad = np.ones((h, 10, 3), dtype=np.uint8) * 255
    vis = img_list[0]
    for i in range(1, len(img_list)):
        vis = np.concatenate([vis, pad, img_list[i]], axis=1)
    vis = cv2.resize(vis, (size * len(img_list) + 10 * (len(img_list) - 1), size))
    return vis.astype(np.uint8)


def visualization(save_root, pic_name, raw_image, raw_anomaly_map, raw_gt, the=0.5, size=518):
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    raw_gt = np.squeeze(raw_gt)
    raw_anomaly_map = np.squeeze(raw_anomaly_map)
    
    if raw_anomaly_map.ndim > 2:
        raw_anomaly_map = raw_anomaly_map[0]
    if raw_gt.ndim > 2:
        raw_gt = raw_gt[0]
    
    img = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    map_norm = normalize(raw_anomaly_map)
    gt_norm = normalize(raw_gt)

    map_binary = np.array(raw_anomaly_map > the, dtype=np.uint8)
    map_binary = np.squeeze(map_binary)
    map_crop = map_norm * map_binary

    ground_truth_contours, _ = cv2.findContours(np.array(raw_gt * 255, dtype = np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    vis_map = apply_ad_scoremap(img, map_norm)
    vis_gt = apply_ad_scoremap(img, gt_norm)
    vis_map_binary = apply_ad_scoremap(img, map_binary)
    vis_map_crop = apply_ad_scoremap(img, map_crop)

    vis_map = cv2.cvtColor(vis_map, cv2.COLOR_RGB2BGR)
    vis_gt = cv2.cvtColor(vis_gt, cv2.COLOR_RGB2BGR)
    vis_map_binary = cv2.cvtColor(vis_map_binary, cv2.COLOR_RGB2BGR)
    vis_map_crop = cv2.cvtColor(vis_map_crop, cv2.COLOR_RGB2BGR)

    if len(ground_truth_contours) > 0:
        vis_map_binary = cv2.drawContours(vis_map_binary, ground_truth_contours, -1, (0, 255, 0), 2)

    merged_img = he_cheng([raw_image, vis_map, vis_map_crop, vis_gt])
    save_path = os.path.join(save_root, pic_name.replace('bmp', 'png').replace('jpg', 'png'))
    cv2.imwrite(save_path, merged_img)

def calcuate_metric_pixel(results, obj_list, args, sigma=8):

    for obj in obj_list:
        gt_px = results['imgs_masks']
        pr_px = results["anomaly_maps"]
        paths = results["path"]

        gt_px = np.array(gt_px)
        pr_px = np.array(pr_px)

        if pr_px.size == 0 or gt_px.size == 0:
            print(f"[WARN] No samples for class '{obj}', skipping.")
            continue

        if sigma != 0:
            pr_px = gaussian_filter(pr_px, sigma=sigma, axes=(1, 2))

        precisions, recalls, thresholds = precision_recall_curve(gt_px.ravel(), pr_px.ravel())
        f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
        best_th = thresholds[np.argmax(f1_scores)]
        f1_best = np.max(f1_scores)
        #print(f"[{obj}] threshold={best_th:.4f}, F1_best={f1_best:.4f}, mean_pred={pr_px.mean():.4f}")

        gt_px = gt_px.squeeze()
        pr_px = pr_px.squeeze()

        for i in range(len(paths)):
            #print("len path: ", len(paths))
            img = cv2.resize(cv2.imread(paths[i]), (args.img_size, args.img_size))

            anomaly_type = os.path.normpath(paths[i]).split(os.sep)[-2]
            
            save_dir = os.path.join(args.visual_path, "visualization", args.dataset, obj, anomaly_type)
            os.makedirs(save_dir, exist_ok=True)

            pic_name = os.path.basename(paths[i])

            visualization(
                save_root=save_dir,
                pic_name=pic_name,
                raw_image=img,
                raw_anomaly_map=np.squeeze(pr_px[i]),
                raw_gt=np.squeeze(gt_px[i]),
                the=best_th,
                size=args.img_size,
            )
