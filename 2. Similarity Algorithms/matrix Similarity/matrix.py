"""Aynı boyuttaki iki görüntünün kanal başına mutlak farkı."""
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent


def main():
    with Image.open(HERE / "görsel 1.png") as first, Image.open(HERE / "görsel 2.png") as second:
        if first.size != second.size:
            raise SystemExit("İki görüntünün boyutu aynı olmalıdır.")
        a = np.asarray(first.convert("RGB"), dtype=np.int16)
        b = np.asarray(second.convert("RGB"), dtype=np.int16)

    delta = np.abs(a - b)
    difference = float(delta.mean() / 255 * 100)
    marked = np.where(np.any(delta, axis=2, keepdims=True), b, 255).astype(np.uint8)
    Image.fromarray(marked).save(HERE / "farkMatris.png")
    print(f"Fark: %{difference:.2f}; benzerlik: %{100 - difference:.2f}")
    print("Çıktı: farkMatris.png")


if __name__ == "__main__":
    main()
