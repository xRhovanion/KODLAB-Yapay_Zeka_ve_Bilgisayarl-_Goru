"""Bu klasörde eğitilen TF-IDF ve lojistik regresyon modeliyle tahmin."""
import joblib

from train import MODEL


if __name__ == "__main__":
    if not MODEL.is_file():
        raise SystemExit("Önce train.py ile yerel modeli eğitin.")
    model = joblib.load(MODEL)
    result = model.predict([input("Yorum: ")])[0]
    print("Model etiketi:", int(result), "(0: makine, 1: insan; eğitim demosu)")
