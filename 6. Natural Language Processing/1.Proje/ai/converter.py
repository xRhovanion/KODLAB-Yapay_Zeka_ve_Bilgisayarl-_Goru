"""Mevcut haber CSV'sinden gerekli sütunları ayrı dosyaya aktar."""
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path(__file__).with_name("dataset.csv"))
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "outputs" / "clean.csv")
    args = parser.parse_args()
    if args.csv.resolve() == args.output.resolve():
        parser.error("Çıktı kaynak CSV ile aynı olamaz.")
    data = pd.read_csv(args.csv, usecols=["Body", "Label"]).dropna().drop_duplicates("Body")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False, encoding="utf-8")
    print(f"{len(data)} satır yazıldı: {args.output}")
