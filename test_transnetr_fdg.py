import os
import torch
import torch.nn as nn
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

from model import Model as TransNetR  # import your TransNetR model
import numpy as np
from scipy.spatial.distance import directed_hausdorff



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
#                 Kvasir-SEG Dataset Loader
# ============================================================

class KvasirSEG(Dataset):
    def __init__(self, root):
        self.img_dir = os.path.join(root, "images")
        self.mask_dir = os.path.join(root, "masks")

        self.images = sorted(os.listdir(self.img_dir))
        self.masks = sorted(os.listdir(self.mask_dir))

        self.transform = v2.Compose([
            v2.Resize((256, 256)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")   # grayscale mask

        img = self.transform(img)
        mask = self.transform(mask)

        mask = (mask > 0.5).float()  # binarize

        return img, mask, self.images[idx][0] if isinstance(self.images[idx], tuple) else self.images[idx]


# ============================================================
#                          METRICS
# ============================================================

def dice_score(pred, target, eps=1e-6):
    pred = pred.flatten()
    target = target.flatten()

    inter = (pred * target).sum()
    denom = pred.sum() + target.sum() + eps
    return (2.0 * inter + eps) / denom

def iou_score(pred, target, eps=1e-6):
    pred = pred.flatten()
    target = target.flatten()

    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter + eps
    return inter / union

# DICE and DSC are the same metric
def dsc_score(pred, target, eps=1e-6):
    return dice_score(pred, target, eps)

def precision_score(pred, target, eps=1e-6):
    pred = pred.flatten()
    target = target.flatten()
    
    tp = (pred * target).sum()
    fp = pred.sum() - tp
    return tp / (tp + fp + eps)

def recall_score(pred, target, eps=1e-6):
    pred = pred.flatten()
    target = target.flatten()
    
    tp = (pred * target).sum()
    fn = target.sum() - tp
    return tp / (tp + fn + eps)

def f2_score(pred, target, eps=1e-6):
    precision = precision_score(pred, target, eps)
    recall = recall_score(pred, target, eps)
    return (5 * precision * recall) / (4 * precision + recall + eps)

def hausdorff_distance(pred, target):
    pred = pred.flatten().cpu().numpy()
    target = target.flatten().cpu().numpy()
    
    # Get coordinates of non-zero pixels
    pred_points = np.array(np.where(pred > 0)).T
    target_points = np.array(np.where(target > 0)).T
    
    if len(pred_points) == 0 or len(target_points) == 0:
        return 0.0
    
    # Calculate Hausdorff distance
    hd1 = directed_hausdorff(pred_points, target_points)[0]
    hd2 = directed_hausdorff(target_points, pred_points)[0]
    return max(hd1, hd2)


# ============================================================
#                          TEST LOOP
# ============================================================

@torch.no_grad()
def evaluate(model, loader, save_pred=True):
    model.eval()

    os.makedirs("predictions/masks", exist_ok=True)
    os.makedirs("predictions/overlays", exist_ok=True)

    dice_scores = []
    iou_scores = []
    dsc_scores = []
    precision_scores = []
    recall_scores = []
    f2_scores = []
    hd_distances = []

    for img, mask, name in loader:
        img = img.to(DEVICE)
        mask = mask.to(DEVICE)

        logits = model(img)
        probs = torch.sigmoid(logits)
        pred = (probs > 0.5).float()

        dice = dice_score(pred.cpu(), mask.cpu())
        iou = iou_score(pred.cpu(), mask.cpu())
        dsc = dsc_score(pred.cpu(), mask.cpu())
        precision = precision_score(pred.cpu(), mask.cpu())
        recall = recall_score(pred.cpu(), mask.cpu())
        f2 = f2_score(pred.cpu(), mask.cpu())
        hd = hausdorff_distance(pred.cpu(), mask.cpu())

        dice_scores.append(dice)
        iou_scores.append(iou)
        dsc_scores.append(dsc)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f2_scores.append(f2)
        hd_distances.append(hd)

        # ---- save prediction ----
        if save_pred:
            # Clean filename
            clean_name = name[0] if isinstance(name, (list, tuple)) else name
            clean_name = clean_name.strip("',\"")  # Remove quotes
            
            pred_img = pred.squeeze().cpu().numpy() * 255
            pred_img = Image.fromarray(pred_img.astype(np.uint8))
            pred_img.save(f"predictions/masks/{clean_name}.png")

            # ---- overlay ----
            overlay = img.squeeze().cpu().permute(1, 2, 0).numpy()
            overlay = (overlay * 255).astype(np.uint8)

            overlay_mask = np.zeros_like(overlay)
            overlay_mask[:, :, 1] = pred.squeeze().cpu().numpy() * 255  # green = prediction
            overlay_mask[:, :, 2] = mask.squeeze().cpu().numpy() * 255  # blue = ground truth

            blended = (0.7 * overlay + 0.3 * overlay_mask).astype(np.uint8)
            overlay_img = Image.fromarray(blended)
            overlay_img.save(f"predictions/overlays/{clean_name}.png")

    print("\n================ RESULTS ================")
    print(f"Mean Dice: {np.mean(dice_scores):.4f}")
    print(f"Mean IoU : {np.mean(iou_scores):.4f}")
    print(f"Mean DSC : {np.mean(dsc_scores):.4f}")
    print(f"Mean Precision: {np.mean(precision_scores):.4f}")
    print(f"Mean Recall: {np.mean(recall_scores):.4f}")
    print(f"Mean F2: {np.mean(f2_scores):.4f}")
    print(f"Mean Hausdorff Distance: {np.mean(hd_distances):.4f}")
    print("=========================================\n")


# ============================================================
#                          MAIN
# ============================================================

def main():
    print("Loading model...")
    model = TransNetR().to(DEVICE)
    model.load_state_dict(torch.load("saved_models/global_final.pth", map_location=DEVICE))
    print("Model loaded successfully.")

    test_dataset = KvasirSEG("Kvasir-SEG")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print("Running evaluation on Kvasir-SEG...")
    evaluate(model, test_loader, save_pred=True)


if __name__ == "__main__":
    main()