# SeamCarving-GUI: 基于内容感知图像缩放算法的实现与多能量函数比较

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 🚀 项目简介

这是一个使用 Python 和 Tkinter 实现的 Seam Carving 内容感知图像缩放工具。它允许用户智能地调整图像尺寸，在缩放过程中尽可能地保留图像中重要内容（如人物、主体物体）的结构和比例，而主要修改图像中不那么重要的区域（如背景、天空）。

项目不仅实现了 Seam Carving 的核心算法，还集成了多种能量函数（包括传统的梯度、熵、显著性，以及基于深度学习的 U2-Netp 显著性检测）进行效果对比，并提供了一个直观的图形用户界面（GUI）进行交互。

## ✨ 主要特性

* **内容感知缩放**: 智能地移除或插入像素缝合线 (seam)，而非简单裁剪或拉伸。
* **多种能量函数**:
    * **Sobel / Laplacian**: 基于图像梯度的边缘信息。
    * **Entropy (熵)**: 基于像素邻域的信息复杂度。
    * **Saliency (显著性)**: 基于谱残差的视觉显著性检测。
    * **Deep (U2-Netp)**: 基于深度学习模型的语义显著性检测，效果通常更优。
* **交互式 GUI**: 基于 Tkinter 构建的用户界面，操作简便。
* **动画演示**: 可视化 Seam Carving 过程中的每一条 seam。
* **批量 Seam 插入**: 加速图像放大过程。
* **多能量函数对比模式**: 一次性展示不同能量函数处理后的结果，方便直观比较。
* **结果导出**: 支持将处理后的图像保存为多种格式。
* **多线程处理**: 后台执行耗时任务，确保 GUI 响应流畅。

## ⚙️ 安装与运行

### 环境要求

* Python 3.8+
* 推荐使用 `pip` 和 `venv` (或 `conda`) 进行环境管理。

### 依赖库

```bash
# 核心依赖
pip install numpy opencv-python Pillow

# 可选依赖 (用于高级功能)
pip install scikit-image
pip install torch torchvision
```

### 下载预训练模型 (U2-Netp)

为了使用“Deep”能量函数，您需要下载预训练的 U2-Netp 模型权重文件。

1.  访问 U2-Net 的 GitHub 仓库: [https://github.com/xuebinqin/U-2-Net](https://github.com/xuebinqin/U-2-Net)
2.  在 readme.md 的 "Download the pre-trained model" 后面可以找到 `u2netp.pth` 模型文件的下载链接。
3.  在项目根目录下创建一个名为 `models` 的文件夹。
4.  将下载的 `u2netp.pth` 文件放入 `models` 文件夹中。
    （即文件路径应为 `models/u2netp.pth`）

### 运行程序

```bash
python main.py
```

## 📸 使用截图

![SeamCarvingGUI](https://github.com/user-attachments/assets/ef46e5e9-1099-4852-b57b-820fad3c948b)

## 💡 算法原理简述

本项目基于 Shai Avidan 和 Ariel Shamir 在 SIGGRAPH 2007 年会上提出的 **Seam Carving** 算法。该算法通过迭代地寻找并移除（或插入）图像中能量最低的“缝合线”（seam）来调整图像尺寸。能量函数用于衡量每个像素的重要性，通常边缘和主体区域的能量较高。动态规划用于高效地找到最优 seam。

本项目在传统的 Seam Carving 基础上，引入了多种能量函数，尤其是基于深度学习的显著性检测，以更好地感知图像的语义内容，从而在缩放时保持图像的视觉质量和内容完整性。

## 许可证

本项目采用 MIT 许可证。
