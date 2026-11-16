"""Word2Vec: sözcük benzerliği ve yakın sözcükler."""
from argparse import ArgumentParser
from vectors import load_vectors

if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Başlıklı Word2Vec .vec veya .bin dosyası")
    parser.add_argument("--word", default="çay")
    parser.add_argument("--other", default="kahve")
    args = parser.parse_args()
    vectors = load_vectors(args.model)
    try:
        print(f"Kosinüs benzerliği: {vectors.similarity(args.word, args.other):.4f}")
        for word, score in vectors.most_similar(args.word, topn=5):
            print(f"{word}: {score:.4f}")
    except KeyError as error:
        parser.error(str(error))
