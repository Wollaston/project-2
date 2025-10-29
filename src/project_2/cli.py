from typing import Callable

import click
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> float:
    pass


def test_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> float:
    pass


@click.command()
@click.argument("train_file", type=click.Path(exists=True))
@click.argument("dev_file", type=click.Path(exists=True))
@click.argument("test_file", type=click.Path(exists=True))
def cli(train_file: str, dev_file: str, test_file: str):
    pass
