"""Kitaptaki karakter düzeyinde GRU yorum üretiminin güncel Keras sürümü."""
from argparse import ArgumentParser
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

HERE = Path(__file__).resolve().parent
MODELS = HERE / "outputs"
DATASET = HERE.parent / "dataset.csv"


def train(sentiment, csv_path, output, limit, epochs):
    keras.utils.set_random_seed(42)
    data = pd.read_csv(csv_path, usecols=["Yorum", "Duygu"]).dropna()
    reviews = data.loc[data.Duygu == sentiment, "Yorum"].astype(str).drop_duplicates()
    if limit and len(reviews) > limit:
        reviews = reviews.sample(limit, random_state=42)
    text = "\n".join(re.sub(r"https?://\S+", "", row.lower()).strip() for row in reviews)
    vocab = sorted(set(text))
    if len(text) < 130 or len(vocab) < 2:
        raise ValueError("Bu etiket için yeterli yorum yok.")
    index = {c: i for i, c in enumerate(vocab)}
    ids = np.array([index[c] for c in text], dtype=np.int32)
    seq_length = 64
    blocks = np.lib.stride_tricks.sliding_window_view(ids, seq_length + 1)[::seq_length]
    model = keras.Sequential([
        keras.Input((None,), dtype="int32"),
        keras.layers.Embedding(len(vocab), 32),
        keras.layers.GRU(64, return_sequences=True),
        keras.layers.Dense(len(vocab)),
    ])
    model.compile(optimizer="adam", loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    model.fit(blocks[:, :-1], blocks[:, 1:], epochs=epochs, batch_size=64)
    output.mkdir(parents=True, exist_ok=True)
    model.save(output / f"gru-{sentiment}.keras")
    (output / f"chars-{sentiment}.json").write_text(
        json.dumps(vocab, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Etiket {sentiment}: {len(reviews)} yorum, {len(blocks)} dizi; model kaydedildi.")


def train_cli(sentiment):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, default=MODELS)
    parser.add_argument("--limit", type=int, default=350, help="0: tüm veri")
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    if args.epochs < 1 or args.limit < 0:
        parser.error("Epoch pozitif ve limit negatif olmamalıdır.")
    try:
        train(sentiment, args.csv, args.output, args.limit, args.epochs)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
