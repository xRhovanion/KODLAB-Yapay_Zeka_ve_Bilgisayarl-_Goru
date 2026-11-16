"""Yerel Flask demosu: girilen metni eğitilmiş LSTM ile sınıflandır."""
from flask import Flask, render_template, request
from ai.prediction import score

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 128 * 1024


@app.get("/")
def main():
    return render_template("root.html")


@app.post("/analiz")
def analyze():
    if request.form.get("tip") != "metin":
        return "Yerel demo yalnızca yapıştırılan haber metnini kabul eder.", 400
    try:
        value = score(request.form.get("veri", ""))
    except FileNotFoundError as error:
        return str(error), 503
    except ValueError as error:
        return str(error), 400
    return f"Model etiketi: {int(value >= .5)}; skor: {value:.4f}. Bu sonuç olgu doğrulaması değildir."


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
