from typing import Callable

import click
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from project_2.dataset import SeqPairDataset
from project_2.model import EncoderDecoder
from project_2.tokenizer import SPECIAL_TOKENS, Tokenizer, TokenizerConfig

device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader[SeqPairDataset],
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> float:
    model.train()
    model.to(device)

    total_loss = 0.0
    torch.autograd.set_detect_anomaly(True)

    for batch, (encoder_input_ids, decoder_input_ids, labels) in enumerate(
        tqdm(dataloader, "Training")
    ):
        encoder_input_ids = encoder_input_ids.to(device)
        decoder_input_ids = decoder_input_ids.to(device)
        labels = labels.to(device)

        logits = model(encoder_input_ids, decoder_input_ids)

        batch_size = logits.shape[0]
        tgt_seq_len = logits.shape[1]
        vocab_size = logits.shape[2]

        logits = torch.reshape(logits, (batch_size * tgt_seq_len, vocab_size))
        labels = torch.reshape(labels, (batch_size * tgt_seq_len,))

        loss = loss_fn(logits, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        if batch % 100 == 0:
            print(f"loss: {loss.item():>7f} [{batch}/{len(dataloader)}]")

    return total_loss / len(dataloader)


def test_epoch(
    model: nn.Module,
    dataloader: DataLoader[SeqPairDataset],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> float:
    model.eval()
    model.to(device)

    total_loss = 0.0

    with torch.no_grad():
        for encoder_input_ids, decoder_input_ids, labels in tqdm(
            dataloader, "Evaluating"
        ):
            encoder_input_ids = encoder_input_ids.to(device)
            decoder_input_ids = decoder_input_ids.to(device)
            labels = labels.to(device)

            logits = model(encoder_input_ids, decoder_input_ids)

            batch_size = logits.shape[0]
            tgt_seq_len = logits.shape[1]
            vocab_size = logits.shape[2]

            logits = torch.reshape(logits, (batch_size * tgt_seq_len, vocab_size))
            labels = torch.reshape(labels, (batch_size * tgt_seq_len,))

            loss = loss_fn(logits, labels)

            total_loss += loss.item()

    return total_loss / len(dataloader)


@click.command()
@click.argument("train_file", type=click.Path(exists=True))
@click.argument("dev_file", type=click.Path(exists=True))
@click.argument("test_file", type=click.Path(exists=True))
@click.option("--epochs", type=click.INT)
@click.option("--learning_rate", type=click.FLOAT)
@click.option("--batch_size", type=click.INT)
@click.option("--max_src_len", type=click.INT)
@click.option("--max_tgt_len", type=click.INT)
@click.option("--d_model", type=click.INT)
@click.option("--num_heads", type=click.INT)
@click.option("--d_ff", type=click.INT)
@click.option("--num_enc_layers", type=click.INT)
@click.option("--num_dec_layers", type=click.INT)
@click.option("--dropout", type=click.FLOAT)
@click.option("--strategy", default="greedy")
@click.option("--beam_width", default=5)
def cli(
    train_file: str,
    dev_file: str,
    test_file: str,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    max_src_len: int,
    max_tgt_len: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
    num_enc_layers: int,
    num_dec_layers: int,
    dropout: float,
    strategy: str,
    beam_width,
):
    tokenizer = Tokenizer(config=TokenizerConfig()).from_file(train_file)

    train_dataset = SeqPairDataset(train_file, tokenizer, max_src_len, max_tgt_len)
    dev_dataset = SeqPairDataset(dev_file, tokenizer, max_src_len, max_tgt_len)
    test_dataset = SeqPairDataset(test_file, tokenizer, max_src_len, max_tgt_len)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = EncoderDecoder(
        src_vocab_size=tokenizer.src_vocab_size,
        tgt_vocab_size=tokenizer.tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_enc_layers=num_enc_layers,
        num_dec_layers=num_dec_layers,
        dropout=dropout,
    )

    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        train_loss = train_epoch(model, train_dataloader, optimizer, loss_fn)
        dev_loss = test_epoch(model, dev_dataloader, loss_fn)

        print(f"Train Loss: {train_loss}\tDev Loss: {dev_loss}")

    total_bleu, num_samples = 0.0, 0.0

    with torch.no_grad():
        model.to(device)
        for encoder_input_ids, _, labels in tqdm(test_dataloader, "Testing"):
            [print(tokenizer.decode(ids)) for ids in encoder_input_ids]
            sequences = model.generate(
                src_ids=encoder_input_ids,
                bos_id=tokenizer.bos_id,
                eos_id=tokenizer.eos_id,
                max_len=max_src_len,
                strategy=strategy,
                beam_width=beam_width,
            )

            print("SEQ: ", sequences)
            predictions = [tokenizer.decode(sequence) for sequence in sequences]
            print("PREDS: ", predictions)
            cleaned_predictions: list[list[str]] = []
            for prediction in predictions:
                cleaned_predictions.append(
                    [tok for tok in prediction if tok not in SPECIAL_TOKENS]
                )
            print("CLEANED PREDS: ", cleaned_predictions)

            ground_truth = [tokenizer.decode(label.tolist()) for label in labels]
            cleaned_ground_truth: list[list[str]] = []
            for gold in ground_truth:
                cleaned_ground_truth.append(
                    [tok for tok in gold if tok not in SPECIAL_TOKENS]
                )

            laplace = SmoothingFunction()
            for pred, gold in zip(cleaned_predictions, cleaned_ground_truth):
                bleu = sentence_bleu(
                    [gold],
                    pred,
                    weights=(1.0, 0.0, 0.0, 0.0),
                    smoothing_function=laplace.method2,
                )
                print(f"PRED: {pred}\nGOLD: {gold}\nBLEU: {bleu}")
                total_bleu += bleu
                num_samples += 1

    print(f"Average BLEU Score: {total_bleu / num_samples}")


if __name__ == "__main__":
    cli()
