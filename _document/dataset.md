### 数据集

#### 关于NIfTI

- 医学影像数据格式：NIfTI(形如xxx.nii.gz)
  
#### 关于reannotated

- reannotated是作者基于 MSD (Medical Segmentation Decathlon) 挑战赛中 Task08 (肝脏血管分割) 的训练集进行的重新标注

- 该标注区分了肝静脉 (HV) 和门静脉 (PV) 的掩膜 (Mask)、

- 包含303个CT卷，以2mm层厚为阈值将原始数据集划分为Thin-slice(层厚 < 2mm)和Thick-slice(层厚 > 2mm), 分别为61例和242例。

- (PS：VSNet论文仅用Thin-slice验证模型方法有效性)

#### 关于本实验数据集dataset

dataset里目前包含imagesTr、labelsTr、preprocessed

**imagesTr** : VSNet 训练需要的原始CT影像，共303例

**labelsTr** : 上面原始CT影像的标注，即上面的reannotated

**preprocessed** : 包含reg和edge, 使用dataPre.py计算得来

- reg : 中心线距离回归图 ($M_d$)
- edge : 扩展边缘掩码 ($C_{exp}$) / 轮廓图

#### 后期可能工作（关于数据集

1. 需要根据 CT 的层厚属性筛选出那 61 例用于特定的训练设置