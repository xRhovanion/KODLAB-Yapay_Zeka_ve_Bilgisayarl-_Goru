"""Kitaptaki CSV ile Word2Vec + LSTM haber sınıflandırma demosu."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

from prediction import BUNDLE, encode, tokenize


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path(__file__).with_name("dataset.csv"))
    parser.add_argument("--limit", type=int, default=800, help="0: tüm veri")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output", type=Path, default=BUNDLE)
    args = parser.parse_args()
    if args.limit < 0 or 0 < args.limit < 20 or args.epochs < 1:
        parser.error("Limit 0 veya en az 20, epoch en az 1 olmalıdır.")
    tf.keras.utils.set_random_seed(42)
    data = pd.read_csv(args.csv, usecols=["Body", "Label"]).dropna()
    data["Body"] = data["Body"].map(lambda body: " ".join(tokenize(body)))
    data = data[data.Body.ne("")].drop_duplicates("Body", keep=False)
    if set(data.Label.unique()) != {0, 1}:
        parser.error("Label sütununda 0 ve 1 sınıfları olmalıdır.")
    if 0 < args.limit < len(data):
        data, _ = train_test_split(data, train_size=args.limit, stratify=data.Label, random_state=42)
    train, test = train_test_split(data, test_size=0.2, stratify=data.Label, random_state=42)
    corpus = [line.split() for line in train.Body]
    model = Word2Vec(corpus, vector_size=48, window=2, min_count=1, workers=1, seed=42, epochs=10)
    vocab = {word: index for index, word in enumerate(model.wv.index_to_key[:19998], 2)}
    max_length = 160
    embedding = np.zeros((len(vocab) + 2, 48), dtype=np.float32)
    for word, index in vocab.items():
        embedding[index] = model.wv[word]
    net = keras.Sequential([
        keras.Input((max_length,), dtype="int32"),
        keras.layers.Embedding(len(embedding), 48, weights=[embedding],
                               trainable=False, mask_zero=True),
        keras.layers.LSTM(24),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    net.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    x_train = encode(corpus, vocab, max_length)
    x_test = encode([line.split() for line in test.Body], vocab, max_length)
    net.fit(x_train, train.Label.to_numpy(), epochs=args.epochs, batch_size=32)
    result = net.evaluate(x_test, test.Label.to_numpy(), return_dict=True, verbose=0)
    args.output.mkdir(parents=True, exist_ok=True)
    net.save(args.output / "model.keras")
    (args.output / "vocabulary.json").write_text(
        json.dumps({"vocab": vocab, "max_length": max_length}, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Eğitim/test: {len(train)}/{len(test)}; demo doğruluğu: {result['accuracy']:.2%}")
    print(f"Model ve sözlük: {args.output}")


if __name__ == "__main__":
    main()
