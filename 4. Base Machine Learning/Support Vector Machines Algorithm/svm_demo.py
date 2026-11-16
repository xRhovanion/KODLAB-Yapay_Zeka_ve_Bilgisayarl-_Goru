"""SVM rakam örnekleri: eğitimde ve tahminde aynı ölçekleyici."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent


def main(kernel, example_image=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=3000, help="Örnek üst sınırı; 0 = tüm veri")
    parser.add_argument("--demo", action="store_true", help="Yerleşik 8×8 digits verisi")
    parser.add_argument("--image", type=Path)
    args = parser.parse_args()
    if args.limit < 0 or 0 < args.limit < 100:
        parser.error("--limit 0 veya en az 100 olmalıdır.")
    csv_path = HERE / "train.csv"
    if args.demo or not csv_path.is_file():
        x, y = load_digits(return_X_y=True)
        side = 8
        print("Yerleşik 8×8 digits demosu; kitaptaki 28×28 veri kullanılmıyor.")
    else:
        data = pd.read_csv(csv_path)
        x, y = data.drop(columns="label").to_numpy(), data["label"].to_numpy()
        side = 28
        if x.shape[1] != 784:
            parser.error("train.csv 784 piksel sütunu içermelidir.")
    if 0 < args.limit < len(y):
        x, _, y, _ = train_test_split(x, y, train_size=args.limit, stratify=y, random_state=101)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=101
    )
    options = {"C": 10, "gamma": 0.001} if kernel == "rbf" else {}
    model = make_pipeline(StandardScaler(), SVC(kernel=kernel, **options))
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    print(f"{kernel}: doğruluk {accuracy_score(y_test, prediction):.2%}; "
          f"eğitim/test: {len(y_train)}/{len(y_test)}")
    if args.image or example_image:
        if side == 8 and args.image is None:
            print(f"Test örneği tahmini: {prediction[0]}; etiket: {y_test[0]}")
            return
        path = args.image or HERE / example_image
        with Image.open(path) as image:
            if image.size != (side, side):
                parser.error(f"Görüntü {side}×{side} olmalıdır.")
            pixels = np.asarray(image.convert("L"), dtype=float).reshape(1, -1)
        if side == 8:
            pixels *= 16 / 255
        print(f"Rakam tahmini: {model.predict(pixels)[0]}")
    else:
        print(confusion_matrix(y_test, prediction))
