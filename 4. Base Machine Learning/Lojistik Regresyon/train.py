"""TF-IDF + lojistik regresyon ile yorum kaynağı sınıflandırma."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

HERE = Path(__file__).resolve().parent
MODEL = HERE / "outputs" / "yorum_model.joblib"


def main():
    data = pd.read_csv(HERE / "dataset.csv", usecols=["Body", "Label"]).dropna()
    data["Body"] = data["Body"].str.strip()
    data = data[data["Body"].ne("")].drop_duplicates("Body", keep=False)
    x_train, x_test, y_train, y_test = train_test_split(
        data["Body"], data["Label"], test_size=0.2, stratify=data["Label"], random_state=0
    )
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 3)), LogisticRegression(max_iter=500))
    model.fit(x_train, y_train)
    print(classification_report(y_test, model.predict(x_test), zero_division=0))
    MODEL.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL)
    print(f"Model kaydedildi: {MODEL.name}")


if __name__ == "__main__":
    main()
