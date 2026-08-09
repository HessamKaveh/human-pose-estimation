import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from dataset import get_dataloaders
from model import SimplePoseNet

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders(batch_size=16)

    model = SimplePoseNet(num_keypoints=17).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10
    history = {"train_loss": [], "test_loss": []}
    best_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, heatmaps in train_loader:
            images, heatmaps = images.to(device), heatmaps.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, heatmaps)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for images, heatmaps in test_loader:
                images, heatmaps = images.to(device), heatmaps.to(device)
                outputs = model(images)
                loss = criterion(outputs, heatmaps)
                test_loss += loss.item() * images.size(0)

        test_loss /= len(test_loader.dataset)

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "best_pose_model.pt"))

    with open(os.path.join(RESULTS_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    plt.figure(figsize=(6, 4))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["test_loss"], label="Test Loss")
    plt.legend(); plt.title("Pose Estimation Training")
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "training_curves.png"))
    print(f"\nBest test loss: {best_loss:.6f}")


if __name__ == "__main__":
    train()
