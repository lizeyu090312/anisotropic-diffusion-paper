from training.dataset import ImageFolderDataset 
from torch.utils.data import DataLoader
import torch
import torchvision as tv

def afhqv2_loader(batch, workers=2, path='datasets/afhqv2-64x64', down_to32=False, xflip=True):
    ds = ImageFolderDataset(path=path, resolution=64, use_labels=False, xflip=xflip, random_seed=0)

    def _collate(samples):
        imgs = []
        for s in samples:
            img_np = s[0]
            x = torch.from_numpy(img_np)
            if x.ndim == 3 and x.shape[0] in (1,3):
                pass
            else:                                         # HWC -> CHW
                x = x.permute(2,0,1)
            x = x.float()/127.5 - 1.0                     # [-1,1]
            imgs.append(x)
        x = torch.stack(imgs, 0)
        if down_to32:
            x = torch.nn.functional.interpolate(x, size=(32,32), mode='bilinear', align_corners=False)
        return x, None

    while True:
        loader = DataLoader(
            ds, batch_size=batch, shuffle=True, drop_last=True,
            num_workers=workers, pin_memory=True,
            persistent_workers=False, prefetch_factor=2 if workers>0 else None,
            collate_fn=_collate,
        )
        for x, y in loader:
            yield x, y

def ffhq_loader(batch, workers=2, path='datasets/ffhq-64x64', down_to32=False, xflip=True):
    ds = ImageFolderDataset(path=path, resolution=64, use_labels=False, xflip=xflip, random_seed=0)

    def _collate(samples):
        imgs = []
        for s in samples:
            img_np = s[0]
            x = torch.from_numpy(img_np)
            if x.ndim == 3 and x.shape[0] in (1,3):
                pass
            else:                                         # HWC -> CHW
                x = x.permute(2,0,1)
            x = x.float()/127.5 - 1.0                     # [-1,1]
            imgs.append(x)
        x = torch.stack(imgs, 0)
        if down_to32:
            x = torch.nn.functional.interpolate(x, size=(32,32), mode='bilinear', align_corners=False)
        return x, None

    while True:
        loader = DataLoader(
            ds, batch_size=batch, shuffle=True, drop_last=True,
            num_workers=workers, pin_memory=True,
            persistent_workers=False, prefetch_factor=2 if workers>0 else None,
            collate_fn=_collate,
        )
        for x, y in loader:
            yield x, y


def cifar10_loader(batch, workers=2):
    ds = tv.datasets.CIFAR10(
        '.', train=True, download=True,
        transform=tv.transforms.Compose([
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor()
        ]))
    while True:
        loader = torch.utils.data.DataLoader(
            ds, batch, shuffle=True, num_workers=workers, drop_last=True)
        for x, y in loader:
            y_hot = torch.eye(10)[y]          # one-hot (N,10)
            yield x, y_hot