import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
from torchvision.models.inception import inception_v3
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os.path as osp
from glob import glob
import re
import logging
import argparse
from scipy import linalg
import json
from multiprocessing import Pool

logger = logging.getLogger(__name__)


class ImageListDataset(Dataset):
    def __init__(self, img_list, transform):
        self.img_list = img_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_np = self.img_list[idx]        
        img_pil = Image.fromarray(img_np)        
        return self.transform(img_pil)


def load_image(path, size=None):
    img = Image.open(path).convert("RGBA")
    if size:
        img = img.resize(size, Image.LANCZOS)
    return np.array(img)

def get_predict_layers(pred_root, layer_boxes):
    all_pngs = glob(osp.join(pred_root, "*.png"))

    background = [p for p in all_pngs if osp.basename(p) == "background_rgba.png"]

    layer_pngs = []
    for p in all_pngs:
        name = osp.basename(p)
        match = re.match(r"layer_(\d+)_rgba\.png", name)
        if match:
            idx = int(match.group(1))
            layer_pngs.append((idx, p))
    layer_pngs = [p for _, p in sorted(layer_pngs, key=lambda x: x[0])]

    ordered_paths = background + layer_pngs

    layers = []
    for i, p in enumerate(ordered_paths):
        img = Image.open(p).convert("RGBA")
        if i < len(layer_boxes):
            x1, y1, x2, y2 = layer_boxes[i]
            img = img.crop((x1, y1, x2, y2))
        layers.append(img)

    return layers

def get_gt_layers(pred_root, layer_boxes):
    all_pngs = glob(osp.join(pred_root, "*.png"))

    layer_pngs = []
    for p in all_pngs:
        name = osp.basename(p)
        match = re.match(r"(\d{2})\.png", name)
        if match:
            idx = int(match.group(1))
            layer_pngs.append((idx, p))
    layer_pngs = [p for _, p in sorted(layer_pngs, key=lambda x: x[0])]
    
    layers = []
    for i, p in enumerate(layer_pngs):
        img = Image.open(p).convert("RGBA")
        if i < len(layer_boxes):
            x1, y1, x2, y2 = layer_boxes[i]
            img = img.crop((x1, y1, x2, y2))
        layers.append(img)

    return layers

def rgba_to_rgb_masked(rgba_img: np.ndarray) -> np.ndarray:
    rgba = rgba_img.astype(np.float32)
    if rgba.max() <= 1.0:
        rgba = rgba * 255.0

    rgb = rgba[..., :3]
    alpha = rgba[..., 3:] / 255.0

    background = np.ones_like(rgb) * 128.0

    # Alpha blending: output = rgb * alpha + background * (1 - alpha)
    out = rgb * alpha + background * (1 - alpha)

    return out.astype(rgba_img.dtype)

def extract_mask(img):
    if img.shape[-1] == 4:
        mask = img[..., 3] / 255.0
    else:
        mask = np.ones(img.shape[:2])
    return mask

def compute_iou(pred_mask, gt_mask, thresh=0.0):
    pred_bin = pred_mask > thresh
    gt_bin = gt_mask > thresh
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    return intersection / (union + 1e-8)

def compute_mask_metrics(pred_mask, gt_mask, thresh=0.0):
    pred_bin = (pred_mask > thresh).astype(np.uint8).flatten()
    gt_bin = (gt_mask > thresh).astype(np.uint8).flatten()
    prec = precision_score(gt_bin, pred_bin, zero_division=0)
    rec = recall_score(gt_bin, pred_bin, zero_division=0)
    f1 = f1_score(gt_bin, pred_bin, zero_division=0)
    iou = compute_iou(pred_mask, gt_mask, thresh)
    return iou, prec, rec, f1

@torch.no_grad()
def get_inception_features(img_list, model, device, batch_size=32, num_workers=0):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    dataset = ImageListDataset(img_list, transform)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    feats = []
    for batch in tqdm(dataloader, desc="Calculating Inception Features", leave=False):
        batch = batch.to(device)
        feat = model(batch)
        feats.append(feat.cpu().numpy())
        
    return np.concatenate(feats, axis=0)

def calculate_fid(feats1, feats2):
    mu1, sigma1 = feats1.mean(axis=0), np.cov(feats1, rowvar=False)
    mu2, sigma2 = feats2.mean(axis=0), np.cov(feats2, rowvar=False)
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)

def process_case(args):
    path, gt_root = args
    alpha_thresh = 0.1
    
    try:
        basename = osp.basename(path)
        match = re.search(r"\d+", basename)
        fn = int(match.group())
        layer_boxes_path = os.path.join(gt_root, f"case_{fn}", "layer_boxes.json")
        with open(layer_boxes_path, "r") as f:
            layer_boxes = json.load(f)
        layer_boxes = layer_boxes[1:]

        pred_layers = get_predict_layers(path, layer_boxes)
        gt_layers = get_gt_layers(os.path.join(gt_root, f"case_{fn}"), layer_boxes)

        gt_comp = load_image(os.path.join(gt_root, f"case_{fn}", "composite.png"))
        pred_comp = load_image(os.path.join(path, f"case_{fn}.png"))

        gt_comp_RGB = rgba_to_rgb_masked(gt_comp)
        pred_comp_RGB = rgba_to_rgb_masked(pred_comp)

        case_layer_psnr, case_layer_ssim = [], []
        case_mask_metrics = []
        case_gt_imgs_fid, case_pred_imgs_fid = [], []

        for g, p in zip(gt_layers, pred_layers):
            gt_img_RGBA = np.array(g)
            pred_img_RGBA = np.array(p)
            gt_img_RGB = rgba_to_rgb_masked(gt_img_RGBA)
            pred_img_RGB = rgba_to_rgb_masked(pred_img_RGBA)
            
            case_layer_psnr.append(psnr(gt_img_RGB, pred_img_RGB, data_range=255))
            case_layer_ssim.append(ssim(gt_img_RGB, pred_img_RGB, channel_axis=2, data_range=255))

            gt_mask, pred_mask = extract_mask(gt_img_RGBA), extract_mask(pred_img_RGBA)
            case_mask_metrics.append(compute_mask_metrics(pred_mask, gt_mask, thresh=alpha_thresh))

            case_gt_imgs_fid.append(gt_img_RGB)
            case_pred_imgs_fid.append(pred_img_RGB)

        case_comp_psnr = psnr(gt_comp_RGB, pred_comp_RGB, data_range=255)
        case_comp_ssim = ssim(gt_comp_RGB, pred_comp_RGB, channel_axis=2, data_range=255)
        
        return (
            case_layer_psnr,
            case_layer_ssim,
            case_mask_metrics,
            case_comp_psnr,
            case_comp_ssim,
            case_gt_imgs_fid,
            case_pred_imgs_fid,
            gt_comp_RGB,
            pred_comp_RGB
        )
    except Exception as e:
        logger.error(f"!!! Failed to process {path}: {e}")
        return None

def evaluate_layers(gt_root, pred_root, num_processes, fid_batch_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inception = inception_v3(pretrained=True, transform_input=False).to(device).eval()

    all_layer_psnr, all_layer_ssim = [], []
    all_mask_metrics = []
    all_composite_psnr, all_composite_ssim = [], []

    gt_imgs_for_fid, pred_imgs_for_fid = [], []
    gt_comp_for_fid, pred_comp_for_fid = [], []

    pred_paths = sorted(
        [
            p for p in glob(osp.join(pred_root, "*"))
            if osp.isdir(p) and re.match(r"case_\d+$", osp.basename(p))
        ],
        key=lambda x: int(re.search(r"\d+$", osp.basename(x)).group())
    )

    tasks = [(path, gt_root) for path in pred_paths]

    logger.info(f"Starting parallel computation of PSNR/SSIM using {num_processes} processes…")

    with Pool(processes=num_processes) as pool:
        results_iter = pool.imap_unordered(process_case, tasks)
        
        for result in tqdm(results_iter, total=len(tasks), desc="Evaluating cases"):
            if result is None:
                continue
            
            (
                case_layer_psnr,
                case_layer_ssim,
                case_mask_metrics,
                case_comp_psnr,
                case_comp_ssim,
                case_gt_imgs_fid,
                case_pred_imgs_fid,
                gt_comp_RGB,
                pred_comp_RGB
            ) = result
            
            all_layer_psnr.extend(case_layer_psnr)
            all_layer_ssim.extend(case_layer_ssim)
            all_mask_metrics.extend(case_mask_metrics)
            
            all_composite_psnr.append(case_comp_psnr)
            all_composite_ssim.append(case_comp_ssim)
            
            gt_imgs_for_fid.extend(case_gt_imgs_fid)
            pred_imgs_for_fid.extend(case_pred_imgs_fid)
            gt_comp_for_fid.append(gt_comp_RGB)
            pred_comp_for_fid.append(pred_comp_RGB)

    logger.info("Calculating Layer FID...")
    gt_feats = get_inception_features(gt_imgs_for_fid, inception, device, batch_size=fid_batch_size)
    pred_feats = get_inception_features(pred_imgs_for_fid, inception, device, batch_size=fid_batch_size)
    fid_layers = calculate_fid(gt_feats, pred_feats)

    logger.info("Calculating Composite FID...")
    gt_feats_comp = get_inception_features(gt_comp_for_fid, inception, device, batch_size=fid_batch_size)
    pred_feats_comp = get_inception_features(pred_comp_for_fid, inception, device, batch_size=fid_batch_size)
    fid_composite = calculate_fid(gt_feats_comp, pred_feats_comp)

    mask_metrics = np.array(all_mask_metrics)
    results = {
        "Layer PSNR": np.mean(all_layer_psnr),
        "Layer SSIM": np.mean(all_layer_ssim),
        "Layer FID": fid_layers,
        "Mask IoU": np.mean(mask_metrics[:, 0]),
        "Mask Precision": np.mean(mask_metrics[:, 1]),
        "Mask Recall": np.mean(mask_metrics[:, 2]),
        "Mask F1": np.mean(mask_metrics[:, 3]),
        "Composite PSNR": np.mean(all_composite_psnr),
        "Composite SSIM": np.mean(all_composite_ssim),
        "Composite FID": fid_composite,
    }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate layer decomposition results")
    parser.add_argument("--pred-dir", type=str, default="Path_to_decomposition_results", help="Directory with predicted layer decompositions")
    parser.add_argument("--gt-dir", type=str, default="Path_to_ground_truth", help="Directory with ground truth layer decompositions")
    parser.add_argument("--output-dir", type=str, default="Path_to_save_eval_result", help="Directory to save evaluation results")
    parser.add_argument("--num-processes", type=int, default=64, help="Number of processes to use")
    parser.add_argument("--fid-batch-size", type=int, default=512, help="Batch size for FID calculation")
    args = parser.parse_args()

    result = evaluate_layers(args.gt_dir, args.pred_dir, args.num_processes, args.fid_batch_size)
    os.makedirs(args.output_dir, exist_ok=True)
    result_path = os.path.join(args.output_dir, "evaluation_results.txt")
    with open(result_path, "w") as f:
        for k, v in result.items():
            f.write(f"{k}: {v}\n")
    print(f"Evaluation results saved to {result_path}")
