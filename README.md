# Autonomous Pick-and-Place System

This project enables real-time **autonomous object detection, 3D localization, and orientation estimation** for robotic pick-and-place applications. It uses the **Intel RealSense D435i** depth camera and **Meta's Segment Anything Model (SAM)** for robust object segmentation. It computes the **3D position**, **orientation**, and **approach direction** of segmented objects, making it suitable for robotic manipulation tasks in industrial and research settings.

---

## Key Features

- Real-time object segmentation using Meta's Segment Anything Model (SAM)
- 3D localization of object centers from RGB + depth data (Intel RealSense)
- Orientation estimation using PCA (Principal Component Analysis)
- Robot approach direction calculated using rotation matrices
- Live visualization:
  - Annotated RGB frame with object info
  - Segmentation + contours overlay
  - Pseudo-colored depth map

---

## Requirements

- Python 3.8+
- NVIDIA GPU (for SAM inference)
- [Intel RealSense SDK 2.0](https://github.com/IntelRealSense/librealsense)
- Segment Anything Model (SAM)

### Install Dependencies

```bash
pip install -r requirements.txt
```

Note: Install SAM and its dependencies from Segment Anything GitHub

### Segment Anything Setup

1. Install the segment-anything-py library.

2. Download the **ViT-Large** model checkpoint from the official [Segment Anything release page](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file).

3. Update the sam_checkpoint path in the code:
  ```python
  sam_checkpoint = "path/to/sam_vit_l_0b3195.pth"
  ```

---

## How It Works

### 1. RealSense Frame Capture
Streams RGB and depth frames from Intel RealSense D435i.

### 2. Object Segmentation with SAM
Segments objects using `SamAutomaticMaskGenerator`.

### 3. Mask Processing
- Extracts one object mask
- Converts it to a grayscale image
- Draws object contour

### 4. 3D Position Estimation
- Converts the 2D center to 3D world coordinates using RealSense intrinsics.

### 5. Orientation Estimation
- Uses PCA on depth points to estimate object orientation
- Converts to Euler angles → Quaternion
- Calculates robot’s approach direction using combined rotation matrix

---

## Visual Outputs

- Segmentation overlay on RGB frame
- Mask with object contours
- Colorized depth map
- Real-time console output:
    ```bash
    World point: [-0.03, 0.12, 0.67]
    Quaternion: [0.12, 0.45, 0.21, 0.87]
    Approach direction: [0.92, -0.01, 0.39]
    ```

---

## 🎥 Demo

[▶️ Watch the demo video](./docs/Final%20Segmentation%20Video-TMNM.gif)

You may download the mp4 video to get the better quality version.

---

## Credits

- [Meta AI – Segment Anything](https://github.com/facebookresearch/segment-anything)
- [Intel RealSense](https://www.intelrealsense.com/)

