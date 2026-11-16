"""Kaydedilmiş LSTM modeliyle tek haber metnini sınıflandır."""
from argparse import ArgumentParser
from pathlib import Path
from prediction import BUNDLE, score


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?")
    parser.add_argument("--model", type=Path, default=BUNDLE)
    args = parser.parse_args()
    try:
        value = score(args.text or input("Haber metni: "), args.model)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(f"Model etiketi: {int(value >= .5)}, skor: {value:.4f} (olgu doğrulaması değildir)")


if __name__ == "__main__":
    main()
