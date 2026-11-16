"""Etiketli CSV'den eğitim metni çıkar / Export labelled reviews as text."""
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=HERE / "dataset.csv")
    parser.add_argument("--sentiment", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()
    data = pd.read_csv(args.csv, usecols=["Yorum", "Duygu"])
    reviews = data.loc[data.Duygu == args.sentiment, "Yorum"].dropna().astype(str)
    output = HERE / "outputs" / f"reviews-{args.sentiment}.txt"
    output.parent.mkdir(exist_ok=True)
    output.write_text("\n".join(reviews), encoding="utf-8")
    print(f"{len(reviews)} yorum kaydedildi: {output}")
