"""Tahminde eğitimde kaydedilen sözcük kimliklerini kullan."""
from functools import lru_cache
import json
from pathlib import Path
import re
import unicodedata

import numpy as np

BUNDLE = Path(__file__).resolve().parent / "outputs" / "news"


def tokenize(text):
    text = unicodedata.normalize("NFC", str(text)).translate(str.maketrans({"I": "ı", "İ": "i"})).lower()
    text = re.sub(r"https?://\S+", " ", text)
    return re.findall(r"[^\W_]+", text)


def encode(sentences, vocab, max_length):
    result = np.zeros((len(sentences), max_length), dtype=np.int32)
    for row, words in enumerate(sentences):
        ids = [vocab.get(word, 1) for word in words[:max_length]]
        result[row, :len(ids)] = ids
    return result


@lru_cache(maxsize=2)
def load_bundle(path):
    from tensorflow.keras.models import load_model

    if not (path / "model.keras").is_file():
        raise FileNotFoundError("Önce ai/trainer.py ile modeli eğitin.")
    config = json.loads((path / "vocabulary.json").read_text(encoding="utf-8"))
    return load_model(path / "model.keras", compile=False), config


def score(text, path=BUNDLE):
    tokens = tokenize(text)
    if not tokens:
        raise ValueError("Boş haber metni.")
    model, config = load_bundle(Path(path).resolve())
    x = encode([tokens], config["vocab"], config["max_length"])
    return float(model(x, training=False).numpy()[0, 0])


def tahminEt(text):
    return score(text) >= 0.5


if __name__ == "__main__":
    from ConsolePrediction import main
    main()
