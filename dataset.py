import os
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    ToTensord,
)
from monai.data import Dataset

def get_dataset(image_dir, label_dir):
    data = []

    for name in os.listdir(image_dir):
        data.append({
            "image": os.path.join(image_dir, name),
            "label": os.path.join(label_dir, name)
        })

    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),
        ToTensord(keys=["image", "label"]),
    ])

    dataset = Dataset(data=data, transform=transforms)
    return dataset
