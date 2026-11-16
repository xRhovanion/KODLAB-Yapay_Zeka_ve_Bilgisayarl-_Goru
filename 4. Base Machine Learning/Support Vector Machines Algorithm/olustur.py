"""Kitaptaki test CSV'sinin ilk 10 rakamını PNG olarak kaydeder."""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    path = HERE / "test.csv"
    if not path.is_file():
        raise SystemExit("test.csv bulunamadı; kaynak depodan bu klasöre ekleyin.")
    rows = pd.read_csv(path, nrows=10)
    output = HERE / "Images"
    output.mkdir(exist_ok=True)
    for i, row in enumerate(rows.to_numpy()):
        Image.fromarray(row.reshape(28, 28).astype(np.uint8)).save(output / f"resim_{i}.png")
    print(f"{len(rows)} görüntü kaydedildi: {output.name}")
