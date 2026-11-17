import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# 17 keypoints COCO format
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


class SyntheticPoseDataset(Dataset):
    """Synthetic dataset for demonstrating pose estimation"""
    def __init__(self, num_samples=200, img_size=256):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_keypoints = 17

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # white image (demo only)
        img = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8) * 200

        # 17 random joints
        keypoints = np.random.rand(self.num_keypoints, 2)
        keypoints[:, 0] *= self.img_size
        keypoints[:, 1] *= self.img_size

        # draw small circles for the keypoints
        for kp in keypoints:
            cv2.circle(img, tuple(map(int, kp)), 3, (0, 255, 0), -1)

        # build the heatmaps
        heatmap = np.zeros((self.num_keypoints, self.img_size, self.img_size), dtype=np.float32)
        sigma = 10
        for k, kp in enumerate(keypoints):
            x, y = map(int, kp)
            if 0 <= x < self.img_size and 0 <= y < self.img_size:
                for i in range(max(0, x - sigma), min(self.img_size, x + sigma)):
                    for j in range(max(0, y - sigma), min(self.img_size, y + sigma)):
                        heatmap[k, j, i] = np.exp(-((i - x) ** 2 + (j - y) ** 2) / (2 * sigma ** 2))

        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        heatmap_tensor = torch.from_numpy(heatmap).float()

        return img_tensor, heatmap_tensor


def get_dataloaders(batch_size=16):
    train_set = SyntheticPoseDataset(num_samples=200)
    test_set = SyntheticPoseDataset(num_samples=50)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader
