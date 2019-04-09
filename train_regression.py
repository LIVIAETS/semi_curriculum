#!/usr/bin/env python3.7

import argparse
from typing import Any, Dict, List, Tuple
from pathlib import Path
from functools import partial
from operator import itemgetter

import torch
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch import einsum, Tensor
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader

from networks import resnext101
from dataloader import SliceDataset
from utils import class2one_hot, tqdm_, map_, id_, str2bool


def main(args: argparse.Namespace) -> None:
    print("\n>>> Setting up")
    cpu: bool = args.cpu or not torch.cuda.is_available()
    device = torch.device("cpu") if cpu else torch.device("cuda")

    cudnn.benchmark = True

    if args.weights:
        print(f">> Loading weights from {args.weights}")
        net = torch.load(args.weights)
    elif args.pretrained:
        print(">> Starting from pre-trained network")
        net = models.resnet101(pretrained=True)
        print("> Recreating its last FC layer")
        in_, out_ = net.fc.in_features, net.fc.out_features
        print(f"> Going from shape {(in_, out_)} to {(8192, args.n_class)}")
        net.fc = nn.Linear(8192, args.n_class)  # Change only the last layer
    else:
        print(">> Using a brand new netwerk")
        net = resnext101(baseWidth=args.base_width, cardinality=args.cardinality, n_class=args.n_class)
    net.to(device)

    lr: float = args.lr
    criterion = torch.nn.MSELoss(reduction="sum")
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Dataloaderz and shitz
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    png_transform = transforms.Compose([
        lambda img: img.convert('L'),
        lambda img: np.array(img)[np.newaxis, ...],
        lambda nd: nd / 255,  # max <= 1
        lambda nd: torch.tensor(nd, dtype=torch.float32),
        normalize if args.pretrained else id_
    ])
    gt_transform = transforms.Compose([
        lambda img: img.convert('L'),
        lambda img: np.array(img)[np.newaxis, ...],
        lambda nd: torch.tensor(nd, dtype=torch.int64),
        partial(class2one_hot, C=args.n_class),
        itemgetter(0)
    ])

    gen_dataset = partial(SliceDataset,
                          transforms=[png_transform, gt_transform],
                          are_hots=[False, True],
                          debug=args.debug,
                          C=args.n_class,
                          in_memory=args.in_memory,
                          bounds_generators=[],
                          ignore_norm=args.pretrained)

    data_loader = partial(DataLoader,
                          num_workers=args.batch_size + 5,
                          pin_memory=True)

    train_filenames: List[str] = map_(lambda p: str(p.name), Path(args.data_root, args.train_subfolder, "img").glob("*"))
    train_folders: List[Path] = [Path(args.data_root, args.train_subfolder, f) for f in ["img", "gt"]]

    val_filenames: List[str] = map_(lambda p: str(p.name), Path(args.data_root, args.val_subfolder, "img").glob("*"))
    val_folders: List[Path] = [Path(args.data_root, args.val_subfolder, f) for f in ["img", "gt"]]

    train_set: Dataset = gen_dataset(train_filenames, train_folders, augment=args.augment)
    val_set: Dataset = gen_dataset(val_filenames, val_folders)

    train_loader: DataLoader = data_loader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader: DataLoader = data_loader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False)

    print()

    best_perf: float = -1
    best_epc: int = -1

    metrics: Dict[str, Tensor] = {"tra_loss": torch.zeros((args.epc, len(train_loader)),
                                                          dtype=torch.float32, device=device),
                                  "tra_diff": torch.zeros((args.epc, len(train_set), args.n_class),
                                                          dtype=torch.float32, device=device),
                                  "val_loss": torch.zeros((args.epc, len(val_loader)),
                                                          dtype=torch.float32, device=device),
                                  "val_diff": torch.zeros((args.epc, len(val_set), args.n_class),
                                                          dtype=torch.float32, device=device)}
    for i in range(args.epc):
        sizes: Tensor
        predicted_sizes: Tensor
        loss: Tensor

        if not args.no_training:
            net, train_metrics = do_epc(i, "train", net, train_loader, device, criterion, args, optimizer)
            for k in train_metrics:
                metrics["tra_" + k][i] = train_metrics[k][...]

        with torch.no_grad():
            net, val_metrics = do_epc(i, "val", net, val_loader, device, criterion, args)
            for k in val_metrics:
                metrics["val_" + k][i] = val_metrics[k][...]

        epc_perf: float = float(metrics["val_diff"][i, ..., args.idc].mean())
        if epc_perf < best_perf or i == 0:
            best_perf = epc_perf
            best_epc = i

            print(f"> Best results at epoch {best_epc}: diff: {best_perf:12.2f}")
            print(f"> Saving network weights to {args.save_dest}")
            Path(args.save_dest).parent.mkdir(parents=True, exist_ok=True)
            torch.save(net, args.save_dest)

        if i in [args.epc // 2, 3 * args.epc // 4]:
            for param_group in optimizer.param_groups:
                lr *= 0.5
                param_group['lr'] = lr
                print(f'> New learning Rate: {lr}')


def do_epc(epc: int, mode: str, net: Any, loader: DataLoader, device, criterion, args,
           optimizer: Any = None) -> Tuple[Any, Dict[str, Tensor]]:
    assert mode in ["train", "val"]

    desc: str
    if mode == "train":
        net.train()
        desc = f">> Training   ({epc})"
    elif mode == "val":
        net.eval()
        desc = f">> Validation ({epc})"

    total_iteration: int = len(loader)  # U
    total_images: int = len(loader.dataset)  # D

    metrics = {"loss": torch.zeros((total_iteration), dtype=torch.float32, device=device),
               "diff": torch.zeros((total_images, args.n_class), dtype=torch.float32, device=device)}

    tq_iter = tqdm_(total=total_iteration, desc=desc)
    done_img: int = 0
    for j, data in enumerate(loader):
        data[1:] = [e.to(device) for e in data[1:]]  # Move all tensors to device
        # filenames, images, targets = data[:3]
        filenames, images, targets = data
        assert len(filenames) == len(images) == len(targets)
        B: int = len(images)

        sizes = einsum("bcwh->bc", targets).type(torch.float32)

        if optimizer:
            optimizer.zero_grad()

        if args.pretrained:
            b, c, w, h = images.shape
            assert c == 1
            viewed = images.view((b, w, h))
            new_img = torch.stack([viewed, viewed, viewed], dim=1)
            assert new_img.shape == (b, 3, w, h), new_img.shape
            images = new_img

        predicted_sizes = net(images)
        assert sizes.shape == predicted_sizes.shape

        loss = criterion(predicted_sizes[:, args.idc], sizes[:, args.idc])

        if optimizer:
            loss.backward()
            optimizer.step()

        metrics["loss"][j] = loss.detach().item()
        metrics["diff"][done_img:done_img + B, ...] = torch.abs(predicted_sizes.detach() - sizes.detach())[...]

        stat_dict: Dict = {"loss": metrics["loss"][:j].mean(),
                           "diff": metrics["diff"][:done_img + B, args.idc].mean()}
        nice_dict: Dict = {k: f"{v:12.2f}" for (k, v) in stat_dict.items()}

        done_img += B
        tq_iter.set_postfix(nice_dict)
        tq_iter.update(1)
    tq_iter.close()
    print(f"{desc} " + ', '.join(f"{k}={v}" for (k, v) in nice_dict.items()))

    return net, metrics


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Hyperparams')

    parser.add_argument("--n_class", type=int, required=True)
    parser.add_argument("--save_dest", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--epc", type=int, required=True)
    parser.add_argument("--train_subfolder", type=str, required=True)
    parser.add_argument("--val_subfolder", type=str, required=True)
    parser.add_argument("--idc", type=int, nargs='+')

    parser.add_argument('--cardinality', type=int, default=32, help='ResNet cardinality (group).')
    parser.add_argument('--base-width', type=int, default=4, help='ResNet base width.')
    parser.add_argument('--lr', '--learning-rate', default=0.0000005, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--augment', type=str2bool, default=False)
    parser.add_argument('--weights', type=str, default='')

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--no_training", action="store_true", help="Trick to rerun evaluation a trained network.")
    parser.add_argument("--cpu", action='store_true')
    parser.add_argument("--in_memory", action='store_true')

    args = parser.parse_args()
    print("\n", args)

    return args


if __name__ == "__main__":
    main(get_args())
