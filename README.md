# VSNet 项目架构说明

代码目录结构规划与各个模块的详细说明：

## 📁 目录结构概览

```text
d:\VSNet-main\
├── config.yaml          # 全局超参数与配置文件
├── scripts\             # 可执行脚本目录
│   └── train.py         # 统一的主训练脚本
├── dataset\             # 数据集相关目录
├── models\              # 核心网络架构定义目录
│   ├── __init__.py      # 模型工厂 (Model Factory)
│   ├── VSNet.py         # 现有的 VSNet 主干
│   ├── unet.py          # 基于 MONAI 构建的标准 U-Net
│   ├── vnet.py          # 基于带残差版 U-Net 构建的 VNet
│   └── attention_unet.py# 基于 MONAI 构建的 Attention U-Net
└── utils\               # 核心训练系统工具箱
    ├── __init__.py
    ├── data_utils.py    # 数据加载 (DataLoader) 与增强编排 (Transforms)
    ├── losses.py        # 多类 Loss 函数适配器与融合逻辑
    └── engine.py        # 核心训练控制台 (Trainer 类)
```

---

## ⚙️ 1. 全局配置模块 (`config.yaml`)
这是整个工程的中枢大脑，用于接管原先 `argparse` 中散落各处的硬编码参数。

- **优势**：保证了所有基线模型（UNet, VNet, Attention UNet, VSNet）在同一套硬件环境、学习率或数据增广标准下进行**绝对公平**的对比实验。
- **动态解析**：通过指明 `model_name: "unet"` 等字符串，脚本将自动向下游模块广播信息，令它们调整读取的数据流、加载的模型及损失函数处理。

---

## 🚀 2. 训练主入口 (`scripts/train.py`)
经过重构后，此文件只有极度干练的 100 余行代码，专注于**流程编排**而非实现细节。

**核心流程：**
1. 读取 `config.yaml` 并处理用户从命令行附加的临时覆盖参数（例如 `python scripts/train.py --batch_size 8`）。
2. 调用 `utils.data_utils.get_dataloader` 构建数据流（Loader）。
3. 调用 `models.__init__.build_model` 构建特定的深度学习模型。
4. 调用 `utils.losses.build_loss` 装配兼容对应模型的损失函数。
5. 初始化由 `utils.engine.py` 驱动的 `Trainer` 并运行 `.fit()`，屏蔽掉底层反向传播等无关逻辑。

---

## 🧠 3. 模型库模块 (`models/`)
存放所有的模型网络定义。

- **工场模式 (`models/__init__.py`)**：这是模型的核心暴露口。它实现了一个名为 `build_model(config)` 的静态函数。当 `train.py` 请求模型时，它会根据配置里的字符串（如 `"vsnet"` 或 `"attention_unet"`）从特定的 py 文件里将模型拉取出来，并统一分配到 GPU 显存上。这代表如果您未来写了一个 `SwinUNETR.py`，只需在此文件加两行映射代码就能直接参与训练。
- **VSNet (`VSNet.py`)**：由于是双头/多输出网络，里面包含了 `GDT`, `SwinLayer` 等自定义算子，其最终输出为由 5 个特征 Tensor 组成的元组。

---

## 🛠️ 4. 多功能工具箱 (`utils/`)
最具有重用价值的模块，包含了具体的前处理、评估和训练细节。

### `data_utils.py` (数据流水线)
维护 `LoadImaged` 等组成的 MONAI Pipeline，负责从磁盘切图喂给显卡。
- **自适应加载**：当它侦测到当前的 `model_name` 是 `vsnet` 时，Transforms 会被设置为同时读取 `image`, `label`, `edge` 和 `reg`（中心点），并执行基于双线性或最近邻的空间联合变换。如果是 `unet`，它会智能掐断后两路数据的读取与运算，加快 I/O 吞吐节奏并降低 RAM 开销。
- **防坍塌种子**：内部在 DataLoader 塞入 `worker_init_fn`，确保 Windows/Linux 多子进程下的几何增强结果具有完美的随机性。

### `losses.py` (融合损失函数)
处理不同网络架构形态不一导致 Loss 崩溃的问题。
- **对于 `unets`**：调用标准的包装器套壳 `DiceCELoss` 提供单一梯度下降支持。
- **对于 `vsnet`**：使用 `VSNetLoss` 进行解包，分别处理主干分类、边缘检测 (edge)、中心回归 (reg) 和 `[epoch2, epoch3]` 层级的深监督，然后把基于 `alpha`, `beta`, `gamma` 系数乘积后的多任务混合 Loss 打包返回。

### `engine.py` (训练引擎主控台)
这是将原先混乱的各种循环压缩进 `Trainer` 类的最终位置。
- **自动混合精度 (AMP)**：维护着 `GradScaler` 和上下文，极大压缩显存占用并提升提速。
- **梯度累积与裁剪 (Gradient Accumulation & Clipping)**：提供基于小物理显存模拟大 Batch Size 的完美实现。如果在小 Batch 尺寸上出现了跳变，会自动触发内置的 `clip_grad_norm_` 以防止模型突然崩坏或 Loss 出 NaN。
- **验证功能集成**：在周期性 Epoch 中，自动接管 `val_loader`，将 `sliding_window_inference`、Dice 的度量聚合以及第一张图片的热力图叠加输出可视化保存到 `vis_check`。最后对 `mean_dice` 进行超越判断，若破纪录则覆盖存放 `best.pth`。
