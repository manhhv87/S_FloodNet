import torch 
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
from pathlib import Path
from typing import Tuple
from PIL import Image
from torchvision.transforms import functional as TF


class FloodNet(Dataset):
    """FloodNet Dataset
    https://ieeexplore.ieee.org/document/9460988 

   https://github.com/BinaLab/FloodNet-Supervised_v1.0
    num_classes: 9 + background
    1445 train images
    450 val images
    448 test images with labels
    """

    CLASSES = [
        'background', 'building flooded', 'building non-flooded', 'road flooded',
        'road non-flooded', 'water', 'tree', 'vehicle', 'pool', 'grass'
    ]

    PALETTE = torch.tensor([
        [0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 255, 120], [0, 0, 255],
        [255, 0, 255], [70, 70, 220], [102, 102, 156], [190, 153, 153],
        [180, 165, 180]
    ])

    def __init__(self, root: str, split: str = 'train', transform=None) -> None:
        super().__init__()
        assert split in ['train', 'val', 'test']
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255

        img_path = Path(root) / split / 'images'
        self.files = list(img_path.glob("*.jpg"))

        if not self.files:
            raise Exception(f"No images found in {img_path}")

        print(f"Found {len(self.files)} {split} images.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        img_path = str(self.files[index])
        mask_path = str(self.files[index]).replace(
            'images', 'masks').replace('.jpg', '_lab.png')

        image = io.read_image(img_path)
        label = TF.pil_to_tensor(Image.open(mask_path).convert('RGB'))

        if self.transform:
            image, label = self.transform(image, label)

        return image, self.encode(label).long()

    def encode(self, label: Tensor) -> Tensor:
        label = label.permute(1, 2, 0)
        mask = torch.zeros(label.shape[:-1])

        for index, color in enumerate(self.PALETTE):
            bool_mask = torch.eq(label, color)
            class_map = torch.all(bool_mask, dim=-1)
            mask[class_map] = index

        return mask


if __name__ == '__main__':
    from semseg.utils.visualize import visualize_dataset_sample
    visualize_dataset_sample(FloodNet, '/content/S_FloodNet/data/FloodNet')
