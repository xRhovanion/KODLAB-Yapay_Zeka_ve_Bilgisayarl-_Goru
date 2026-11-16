"""Yerel GRU modeliyle örnek metin üret; çıktıyı herhangi bir siteye gönderme."""
from argparse import ArgumentParser
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

MODELS = Path(__file__).resolve().parent / "outputs"


@lru_cache(maxsize=2)
def load_bundle(path, sentiment):
    from tensorflow.keras.models import load_model

    model_path = path / f"gru-{sentiment}.keras"
    if not model_path.is_file():
        name = "pozitif" if sentiment == 1 else "negatif"
        raise FileNotFoundError(f"Önce ai/train_{name}.py dosyasını çalıştırın.")
    vocab = json.loads((path / f"chars-{sentiment}.json").read_text(encoding="utf-8"))
    return load_model(model_path, compile=False), vocab


def yorum_uret(duygu, token, length=100, model_dir=MODELS, seed=None):
    if duygu not in (0, 1) or not 1 <= length <= 300:
        raise ValueError("Etiket 0/1, uzunluk 1–300 olmalıdır.")
    model, vocab = load_bundle(Path(model_dir).resolve(), duygu)
    index = {c: i for i, c in enumerate(vocab)}
    ids = [index[c] for c in token.lower() if c in index] or [index.get(" ", 0)]
    rng = np.random.default_rng(seed)
    for _ in range(length):
        logits = np.asarray(model(np.array([ids[-64:]], dtype=np.int32), training=False))[0, -1]
        prob = np.exp(logits - logits.max()).astype(float)
        ids.append(int(rng.choice(len(vocab), p=prob / prob.sum())))
    return "".join(vocab[i] for i in ids)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--sentiment", type=int, choices=(0, 1), default=1)
    parser.add_argument("--seed-text", default="kargo")
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--model-dir", type=Path, default=MODELS)
    args = parser.parse_args()
    try:
        print("Yapay olarak üretilen eğitim örneği:")
        print(yorum_uret(args.sentiment, args.seed_text, args.length, args.model_dir))
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
