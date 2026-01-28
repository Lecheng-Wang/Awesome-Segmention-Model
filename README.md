# Awesome Semantic Segmentation Models

该代码是一套功能完备、高灵活性、一站式的多模型语义分割统一训练框架，不仅全面整合了 U-Net、DeepLab、ENet、FCN、HRNet、SegNet、RefineNet、PSPNet、SegFormer、SETR、UperNet、OCRNet、Mask2Former、SegNeXt 共计 14 种主流语义分割模型，还对每个模型的不同复杂度版本进行了完善适配（例如 Swin Transformer 系列的 Swin-T/Swin-S/Swin-B/Swin-L 等高配、低配版本），且在模型调用阶段支持手动灵活修改配置，满足不同显存条件、精度需求下的训练场景，兼顾了研究性与工程落地的实用性。我们针对语义分割这类像素级分类任务的核心训练需求，进行了全方位的定制化配置与性能优化，该框架可直接开箱应用于冰川分割、遥感影像解译、医学影像分割、城市道路场景分割等各类语义分割任务，同时也非常适用于不同分割模型的性能对比、骨干网络选型、改进算法验证等学术研究场景，为相关工作提供了高效、统一的实验基础。
该框架通过命令行参数实现了全训练流程的灵活配置，无需改动核心代码即可完成各类参数调整：在硬件适配层面，支持单 GPU、指定多 GPU、全 GPU 三种训练模式，可根据硬件资源灵活选择，最大化利用计算能力；在训练组件层面，支持自由切换 Adam、SGD 两种主流优化器，交叉熵损失（CE Loss）、焦点损失（Focal Loss）两种针对语义分割的损失函数，Poly、Step、Cos、Exp 四种学习率调度器，以及 Kaiming、Normal、Xavier、Orthogonal 四种经典权重初始化方式，充分适配不同分割模型的训练特性与收敛需求。为保障训练结果的稳定性与可复现性，代码在初始化阶段即固定了全局随机种子，关闭了 cudnn 的随机化优化策略并开启确定性计算模式，训练前还会自动清空 GPU 缓存，避免显存碎片与残留数据对训练过程的干扰；同时对预训练权重加载做了针对性优化处理，自动识别并移除多卡训练过程中产生的module.前缀，且采用非严格匹配模式加载权重，兼容不同训练环境下生成的权重文件与模型层结构的细微差异，当无预训练权重可用时，框架会自动调用对应初始化方法完成模型权重的初始化，确保训练的平稳启动。
整个框架实现了从模型实例化、数据集加载、训练迭代、梯度反向传播，到验证集性能评估、模型权重自动保存、训练日志完整记录的端到端全流程闭环，无需额外补充辅助代码。在数据处理层面，基于 PyTorch 的Dataset抽象类与DataLoader数据加载器，实现了高效的数据批量读取、洗牌与内存锁定，保障了训练过程的数据供给效率；在训练监控层面，通过tqdm库实现了训练与验证过程的实时进度条展示，同步输出当前批次与累计的损失值，方便开发者实时掌握模型收敛状态；在性能评估层面，验证阶段会全面计算 Acc（整体准确率）、Kappa（卡帕系数）、mIoU（平均交并比）、FWIoU（频率加权交并比）、Precision（精确率）、Recall（召回率）、F1-Score（F1 分数）、F2-Score（F2 分数）等全维度性能指标，同时还会输出每个类别的单独细分指标，实现对模型性能的精细化、精准化分析，便于定位模型在特定类别上的不足与优化方向。
针对语义分割任务中普遍存在的数据不平衡问题，该框架引入了自定义类别权重进行损失加权计算，通过对样本量较少的类别赋予更高的损失权重，缓解数据分布不均导致的模型偏向性，提升模型对小众类别的识别能力。在训练结果留存层面，框架会按照指定周期自动将模型权重保存至专属的pth_files目录，权重文件名包含模型类型、训练轮数、训练损失与验证损失等关键信息，方便后续筛选最优模型与断点续训；同时会将每一轮的训练损失、验证损失及所有评估指标完整写入 CSV 格式日志文件，日志结构清晰、字段齐全，可直接通过 Excel、Pandas 等工具进行后续的训练曲线绘制、性能趋势分析与结果复盘。
整体代码结构清晰、模块化程度高，采用面向对象的设计思想将核心训练逻辑与模型配置解耦，仅需修改--MODEL_TYPE命令行参数即可快速切换不同的分割模型，无需改动训练主流程与评估逻辑，大幅提升了实验效率。同时框架兼顾了训练效率与部署兼容性，自动处理多卡训练权重与单卡部署的格式适配问题，无需手动修改权重文件即可直接用于后续推理部署。此外，在net/backbone/目录下，我们还额外整合了近 20 种常用的经典深度学习骨干网络结构文件，涵盖多个主流网络系列：其中卷积神经网络系列包括 ResNet 家族（基础 ResNet、引入分组卷积的 ResNeXt、加入 SE 注意力机制的 SE-ResNet、加入 ECA 注意力机制的 ECA-ResNet、加入 CBAM 注意力机制的 CBAM-ResNet）、MobileNet 家族（v1/v2/v3/v4 多版本迭代的轻量级移动端网络）、Inception 家族（v1/v2/v3/v4 多分支卷积架构，含 GoogLeNet），以及 DenseNet（密集连接卷积网络）、EfficientNet（复合缩放策略的高效网络）、ShuffleNet（引入通道洗牌的轻量级网络）、Xception（深度可分离卷积架构）、VGG（经典卷积 - 池化堆叠架构）等。这些骨干网络均可作为各类分割模型的特征提取器，开发者可根据任务需求，轻松将分割模型的默认主干网络替换为上述任意骨干网络，进一步拓展框架的适用范围与优化空间，为语义分割模型的创新与落地提供了更丰富的选择与更坚实的支撑。


## 📝 作者信息
* **Author**: Lecheng Wang
* **Time**: 2026/1/11 (last updated)

## ✨ 主要特性

* **多模型支持**：集成了 CNN (UNet, DeepLab, HRNet...) 和 Transformer (SegFormer, SETR, SegNeXt...) 系列共 10+ 种模型。
* **多波段支持**：默认支持多通道输入（代码默认为 10 通道），非常适合遥感或多光谱图像分割任务。
* **灵活的训练模式**：支持单卡 (Single)、指定多卡 (Multi-list) 和全卡 (All) 并行训练。
* **多种 Loss 函数**：支持 CrossEntropy 和 Focal Loss。
* **完善的日志系统**：自动保存训练日志 (`.csv`)，记录 mIoU, Kappa, F1-score, Precision, Recall 等详细指标。
* **学习率策略**：支持 Poly, Step, Cosine, Exponential 等多种衰减策略。

## 🏗️ 支持的模型

可以在 `--MODEL_TYPE` 参数中指定以下模型：

* `Unet`       (vgg11/13/16/19、resnet18/34/50/101/152)
* `Deeplab`    (xception/mobilenet/resnet/vggnet/inception)
* `PSPNet`     (mobilenetv2/resnet50)
* `HRNet`      (V1:w18/w32/w48 ; V2:w18/w32/w48)
* `SegNet`     (vgg16)
* `FCN`        (fcn 16s/8s/32s)
* `ENet`       None
* `RefineNet`  (ResNet50/101/152)
* `Segformer`  (b0/b1/b2/b3/b4/b5)
* `SETR`       (Base/Large)
* `UperNet`    (Swin-T)
* `OCRNet`     (V2:w18/w32/w48)
* `SegNext`    (T/S/B/L)
* `Mask2Former` (代码有误，暂未修改，不建议用)

## 📂 目录结构

请按照以下结构组织你的项目和数据：

```text
├── datasets/
│   ├── annotations/
│   │   ├── train.txt    # 训练集列表
│   │   └── val.txt      # 验证集列表
│   ├── images/          # 图片存放目录
│   └── labels/          # 标签存放目录
├── output/              # 预测结果存放目录
├── test_sample/          # 测试样本存放目录
├── nets/                # 模型定义文件
├── utils/               # 工具类 (dataset, metrics, losses 等)
├── pth_files/           # (自动生成) 存放训练好的模型权重
├── train.py             # 训练主程序
├── Structure.py         # 预测阶段用于重构网络结构
├── predict_small.py     # 预测小尺寸(256*256)的结果
└── predict_large.py     # 预测大尺寸(例如1024*2975)的结果
               
```

## 🚀 快速开始 (Quick Start)

**1.环境准备确保安装了 PyTorch 和必要的依赖库**
```bash
pip install torch torchvision numpy tqdm torchinfo thop csv
```

**2.准备数据列表**
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
**3.启动训练**
基础运行 (默认Unet):
bash
```
python train.py
```

**4.指定模型与参数:**
例如使用 SegFormer，Batch Size 为 8，训练 100 轮：
bash
```
python train.py --MODEL_TYPE segformer --BATCH_SIZE 8 --EPOCHS 100
```

**5.多 GPU 并行训练:**
例如指定使用 GPU 0 和 GPU 1：
bash
```
python train.py --MODE muti --GPU_LIST "0,1" --BATCH_SIZE 16
```

## ⚙️ 参数说明 (Arguments)
你可以在命令行中调整以下参数：

| 参数名 | 默认值 | 说明 | 可选值 |
|---|---|---|---|
| --DATASET_PATH | ./datasets/ | 数据集路径 | 可以根据当前位置设置相对路径 |
| --MODE | all | 训练模式 | single, multi, all |
| --GPU_LIST | 1,2 | 多卡模式下的设备 ID | 如 "0,1,2,3" |
| --MODEL_TYPE | unet | 模型架构 | U-Net、DeepLab、ENet、FCN、HRNet、SegNet、RefineNet、PSPNet、SegFormer、SETR、UperNet、OCRNet、Mask2Former、SegNeXt |
| --BACKBONE_TYPE | resnet50 | 骨干网络 | unet:vgg11/13/16/19、resnet18/34/50/101/152   deeplab:xception/mobilenet/resnet/vggnet/inception|
| --BANDS | 10 | 输入图片通道数 | 根据数据修改 如(RGB=3) |
| --NUM_CLASS | 3 | 类别总数 (含背景) | 主要提取的类别数+背景类 |
| --BATCH_SIZE | 4 | 批处理大小 | 4/8/16/32/64/... |
| --EPOCHS | 1 | 训练轮数 | 100/150/200/300 |
| --LOSS_TYPE | ce | 损失函数 | ce, focal |
| --OPTIMIZER_TYPE | adam | 优化器 | adam, sgd |
| --LR_SCHEDULER | poly | 学习率策略 | poly, step, cos, exp |
| --PRETRAIN_MODEL | None | 预训练权重路径 | ***.pth 文件路径 |

## 📊 结果记录
程序运行后会生成：

**训练日志 ({MODEL_TYPE}_training_log.csv):**
记录每个 Epoch 的 Train Loss, Val Loss。记录详细指标：mIoU, Acc, Kappa, F1-Score, Precision, Recall (包含各类别的详细分数)。

**模型权重 (pth_files/):**
命名格式：{MODEL}-epoch{N}-loss{L}-val_loss{VL}.pth。根据 --MODEL_SAVE_EPOCHS 参数定期保存。

# ⚠️ 注意事项 (Important Note)
代码中 (train.py 第 127 行) 包含硬编码的类别权重：
```
weight = np.array([0.03247, 0.25926, 0.70827], np.float32)
```
***请务必在使用前根据你自己数据集的类别分布修改此权重，或者将其注释掉以使用默认的平均权重，否则会严重影响训练效果。***
