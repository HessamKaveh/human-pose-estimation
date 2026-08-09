# 2D Human Pose Estimation

Convolutional neural network for detecting 17 keypoints (COCO format)
on human bodies in images.

## Architecture
Encoder-decoder network that predicts per-keypoint heatmaps.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python src/train.py
python src/evaluate.py
```

## Author
Hessam Kaveh — Research Fellow, Italian Institute of Technology
