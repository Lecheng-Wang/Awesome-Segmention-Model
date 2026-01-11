# Awesome Semantic Segmentation Model

该代码是一套功能完备、高灵活性、一站式的多模型语义分割统一训练框架，全面整合了U-Net、DeepLab、ENet、FCN、HRNet、SegNet、RefineNet、PSPNet、SegFormer、SETR、UperNet、OCRNet、Mask2Former、SegNeXt共14种主流语义分割模型，针对语义分割任务的训练需求做了全方位的配置与优化，可直接应用于冰川分割、遥感影像解译、医学影像分割等各类像素级分类场景，同时也适用于不同分割模型的性能对比与选型研究。
框架通过命令行参数实现了全流程的灵活配置，支持单GPU、指定多 GPU、全GPU三种训练模式，可自由切换 Adam/SGD优化器、交叉熵损失函数(ce loss)/焦点损失函数(focal loss)、Poly/Step/Cos/Exp学习率调度器，以及Kaiming/Normal/Xavier/Orthogonal多种权重初始化方式，适配不同模型的训练特性；为保障训练的稳定性与复现性，代码初始化时固定了随机种子，关闭了cudnn的随机化策略，训练前会自动清空GPU缓存，同时对预训练权重加载做了优化处理，自动移除多卡训练产生的module.前缀并设置非严格匹配，无预训练权重时则自动执行权重初始化。框架实现了从模型实例化、数据集加载、训练迭代到验证评估、模型保存、日志记录的端到端流程，基于自定义Labeled_Model_Dataset数据集类和DataLoader完成高效的数据批量加载，训练与验证过程通过tqdm实现实时进度与损失监控，验证阶段完成Acc、Kappa、mIoU、FWIoU、Precision、Recall、F1-Score、F2-Score等全维度性能指标的计算，同时输出各类别的单独指标，实现模型性能的精准分析。
针对语义分割任务中常见的数据不平衡问题，框架引入了自定义类别权重进行损失加权；训练过程中会按指定周期自动保存模型权重至专属目录，同时将每轮的训练损失、验证损失及所有评估指标写入CSV格式日志文件，方便后续的训练曲线绘制与性能复盘。整体代码结构清晰、模块化程度高，核心训练逻辑与模型配置解耦，通过修改--MODEL_TYPE参数即可快速切换不同分割模型，无需改动训练主流程，同时兼顾了训练效率与部署兼容性，自动处理多卡训练权重的格式问题，为语义分割模型的训练与研究提供了开箱即用的解决方案，仅需配置数据集路径与少量训练参数即可启动训练，大幅降低了多模型语义分割的训练门槛。

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
├── predict_large.py     # 预测大尺寸(例如1024*2975)的结果
└──               
```

## 🚀 快速开始 (Quick Start)

** 1. 环境准备确保安装了 PyTorch 和必要的依赖库
\```bash
pip install torch torchvision numpy tqdm
\```

**2. 准备数据列表**
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
**3. 启动训练**
基础运行 (默认 Unet):
\```bash
python train.py
\```

**4.指定模型与参数:**
例如使用 SegFormer，Batch Size 为 8，训练 100 轮：
\```bash
python train.py --MODEL_TYPE segformer --BATCH_SIZE 8 --EPOCHS 100
\```

**5.多 GPU 并行训练:**
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
| --MODEL_TYPE | unet | 模型架构 | U-Net、DeepLab、ENet、FCN、HRNet、SegNet、RefineNet、PSPNet、SegFormer、SETR、UperNet、OCRNet、Mask2Former、SegNeXt |
| --BACKBONE_TYPE | resnet50 | 骨干网络 | unet:vgg11/13/16/19、resnet18/34/50/101/152   deeplab:xception/mobilenet/resnet/vggnet/inception|
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
