from .dataset.Remote import RemoteDataset
from torch.utils.data import DataLoader


def build_dataloader(args, split="train"):
    dataset = RemoteDataset(root_dir=args.root_dir, split=split, in_channels=args.in_channels)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True if split == "trian" or split=="train_all" else False,
        pin_memory=True,
    )

    return dataloader
