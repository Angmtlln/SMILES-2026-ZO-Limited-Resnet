import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.datasets as datasets
from augmentation import get_transforms


def get_train_dataset_loader(data_dir, batch_size, generator_train):
    dataset = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=get_transforms(train=True),
    )

    targets = dataset.targets
    class_counts = [0] * 100
    for t in targets:
        class_counts[t] += 1

    sample_weights = [1.0 / class_counts[t] for t in targets]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True,
        generator=generator_train,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
    )

    return dataset, loader