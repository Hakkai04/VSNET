### 关于VSNet的研究

医学影像深度学习的研究
Github仓库地址：https://github.com/XXYZB/VSNet

#### 库

- MONAI：医学深度学习“官方标准库”


#### 数据集

VSNet使用的医学影像的数据集

医学影像数据格式：NIfTI(形如xxx.nii.gz)

##### 关于reannotated数据集的解释

- reannotated里的是已经标注好的数据集，需要去 http://medicaldecathlon.com/ 里下载未标注的原始数据集。

- 以2mm层厚为阈值将原始数据集划分为Thin-slice(层厚 < 2mm)和Thick-slice(层厚 > 2mm), 分别为61例和242例。(此处论文中将薄层和厚层的区分弄反了)

- 论文仅用Thin-slice验证模型方法有效性(?)

#### 模型

##### 训练参数和环境
- AdamW优化器
- 初始学习率：10⁻³
- 权重衰减：10⁻⁵
- batch: 16
- 验证集Dice分数在2000个周期内不再提升时终止
- Pytorch, 8块40GB显存的A100 GPU

#### 验证指标

- Dice系数：通过计算模型预测结果与真实标签之间的重叠度来度量模型的准确性。Dice系数越接近1，表示模型预测结果与真实标签的重叠度越高，即模型性能越好。

