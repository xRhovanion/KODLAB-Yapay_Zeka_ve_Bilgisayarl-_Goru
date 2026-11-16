"""Small offline Word2Vec demo, or a user-supplied vector file."""
from pathlib import Path

from gensim.models import KeyedVectors, Word2Vec


def load_vectors(path=None):
    if path:
        model_path = Path(path)
        return KeyedVectors.load_word2vec_format(str(model_path), binary=model_path.suffix == ".bin")
    print("Küçük yerel eğitim demosu; gerçek Türkçe model değildir.")
    corpus = [
        "çay kahve sıcak içecek fincan",
        "çay bardak sıcak içecek",
        "kahve fincan sıcak içecek",
        "fincan bardak çay kahve",
        "yaprak dal kök ağaç",
        "ağaç orman yaprak dal",
    ]
    return Word2Vec([line.split() for line in corpus] * 20, vector_size=24,
                    window=3, min_count=1, workers=1, seed=42, epochs=20).wv
