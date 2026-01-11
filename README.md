# PyTorch Semantic Segmentation Toolbox

这是一个基于 PyTorch 实现的多功能语义分割训练框架。项目集成了多种主流的分割网络（如 UNet, DeepLabV3+, SegFormer, HRNet 等），支持多波段输入（适用于遥感图像），并提供了灵活的多 GPU 训练策略和完善的指标评估。

## 📝 作者信息
* **Author**: Lecheng Wang
* **Time**: 2026/1/11

## ✨ 主要特性

* **多模型支持**：集成了 CNN (UNet, DeepLab, HRNet...) 和 Transformer (SegFormer, SETR, SegNeXt...) 系列共 10+ 种模型。
* **多波段支持**：默认支持多通道输入（代码默认为 10 通道），非常适合遥感或多光谱图像分割任务。
* **灵活的训练模式**：支持单卡 (Single)、指定多卡 (Multi-list) 和全卡 (All) 并行训练。
* **多种 Loss 函数**：支持 CrossEntropy 和 Focal Loss。
* **完善的日志系统**：自动保存训练日志 (`.csv`)，记录 mIoU, Kappa, F1-score, Precision, Recall 等详细指标。
* **学习率策略**：支持 Poly, Step, Cosine, Exponential 等多种衰减策略。

## 🏗️ 支持的模型 (Model Zoo)

可以在 `--MODEL_TYPE` 参数中指定以下模型：

* `unet`
* `deeplab` (DeepLabV3+)
* `pspnet`
* `hrnet` (HRNetV2)
* `segnet`
* `fcn` (FCN16s, 8s, 32s)
* `enet`
* `refinenet`
* `segformer`
* `setr` (SETR)
* `upernet`
* `ocrnet` (HRNet + OCR)
* `segnext`
* `mask2former` (Experimental)

## 📂 目录结构

请按照以下结构组织你的项目和数据：

```text
├── datasets/
│   ├── annotations/
│   │   ├── train.txt    # 训练集列表 (格式: 图片路径 标签路径)
│   │   └── val.txt      # 验证集列表
│   ├── images/          # 图片存放目录
│   └── labels/          # 标签存放目录
├── nets/                # 模型定义文件
├── utils/               # 工具类 (dataset, metrics, losses 等)
├── pth_files/           # (自动生成) 存放训练好的模型权重
├── train.py             # 训练主程序
└── ...
```

## 🚀 快速开始 (Quick Start)

** 1. 环境准备确保安装了 PyTorch 和必要的依赖库
\```bash
pip install torch torchvision numpy tqdm
\```

** 2. 准备数据列表
在 datasets/annotations/ 下创建 train.txt 和 val.txt。每一行包含图像的文件名。
```示例：
   train.txt: 001
              002
              003
              ...
   val.txt :  001
              002
              003
              ...
```
** 3. 启动训练
**基础运行 (默认 Unet):**
\```bash
python train.py
\```

** 4.指定模型与参数:**
例如使用 SegFormer，Batch Size 为 8，训练 100 轮：
\```bash
python train.py --MODEL_TYPE segformer --BATCH_SIZE 8 --EPOCHS 100
\```

** 5.多 GPU 并行训练:**
例如指定使用 GPU 0 和 GPU 1：
\```bash
python train.py --MODE muti --GPU_LIST "0,1" --BATCH_SIZE 16
\```

## ⚙️ 参数说明 (Arguments)
你可以在命令行中调整以下参数：

| 参数名 | 默认值 | 说明 | 可选值 |
|---|---|---|---|
| --DATASET_PATH | ./datasets/ | 数据集根路径 | - |
| --MODE | all | 训练模式 | single, muti, all |
| --GPU_LIST | 1,2 | 多卡模式下的设备 ID | 如 "0,1,2,3" |
| --MODEL_TYPE | unet | 模型架构 | 见 Model Zoo |
| --BACKBONE_TYPE | resnet50 | 骨干网络 | resnet50, vgg16, b0... |
| --BANDS | 10 | 输入图片通道数 | 根据数据修改 (RGB=3) |
| --NUM_CLASS | 3 | 类别总数 (含背景) | - |
| --BATCH_SIZE | 4 | 批处理大小 | - |
| --EPOCHS | 1 | 训练轮数 | - |
| --LOSS_TYPE | ce | 损失函数 | ce, focal |
| --OPTIMIZER_TYPE | adam | 优化器 | adam, sgd |
| --LR_SCHEDULER | poly | 学习率策略 | poly, step, cos, exp |
| --PRETRAIN_MODEL | None | 预训练权重路径 | .pth 文件路径 |

## 📊 结果记录
程序运行后会生成：

**训练日志 ({MODEL_TYPE}_training_log.csv):**
记录每个 Epoch 的 Train Loss, Val Loss。记录详细指标：mIoU, Acc, Kappa, F1-Score, Precision, Recall (包含各类别的详细分数)。

**模型权重 (pth_files/):**
命名格式：{MODEL}-epoch{N}-loss{L}-val_loss{VL}.pth。根据 --MODEL_SAVE_EPOCHS 参数定期保存。

# ⚠️ 注意事项 (Important Note)
代码中 (train.py 第 127 行) 包含硬编码的类别权重：
\```python
weight = np.array([0.03247, 0.25926, 0.70827], np.float32)
\```
请务必在使用前根据你自己数据集的类别分布修改此权重，或者将其注释掉以使用默认的平均权重，否则会严重影响训练效果。
