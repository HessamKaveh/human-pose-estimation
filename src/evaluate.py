import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter

from dataset import get_dataloaders, KEYPOINT_NAMES
from model import SimplePoseNet

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODEL_PATH = os.path.join(RESULTS_DIR, "best_pose_model.pt")


def extract_keypoints(heatmaps):
    """Extract keypoint coordinates from heatmaps"""
    keypoints = []
    for i in range(heatmaps.shape[0]):
        hm = heatmaps[i].cpu().numpy()
        y, x = np.unravel_index(np.argmax(hm), hm.shape)
        keypoints.append([x, y])
    return np.array(keypoints)


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = get_dataloaders(batch_size=4)

    model = SimplePoseNet(num_keypoints=17).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    images, gt_heatmaps = next(iter(test_loader))
    images = images.to(device)

    with torch.no_grad():
        pred_heatmaps = model(images)

    n_samples = min(3, images.size(0))
    fig, axes = plt.subplots(n_samples, 4, figsize=(12, 3 * n_samples))

    for i in range(n_samples):
        img = images[i].cpu().permute(1, 2, 0).numpy()
        gt_sum = gt_heatmaps[i].sum(dim=0).cpu().numpy()
        pred_sum = pred_heatmaps[i].sum(dim=0).cpu().numpy()

        axes[i, 0].imshow(img); axes[i, 0].set_title("Input Image"); axes[i, 0].axis("off")
        axes[i, 1].imshow(gt_sum, cmap="jet"); axes[i, 1].set_title("GT Heatmaps (sum)"); axes[i, 1].axis("off")
        axes[i, 2].imshow(pred_sum, cmap="jet"); axes[i, 2].set_title("Pred Heatmaps (sum)"); axes[i, 2].axis("off")

        # draw keypoints on the image
        img_with_kps = img.copy()
        pred_kps = extract_keypoints(pred_heatmaps[i])
        for kp in pred_kps:
            cv2.circle(img_with_kps, tuple(map(int, kp)), 3, (0, 255, 0), -1)
        axes[i, 3].imshow(img_with_kps); axes[i, 3].set_title("Predicted Keypoints"); axes[i, 3].axis("off")

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "pose_predictions.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved pose predictions to {out_path}")


if __name__ == "__main__":
    import cv2
    evaluate()
