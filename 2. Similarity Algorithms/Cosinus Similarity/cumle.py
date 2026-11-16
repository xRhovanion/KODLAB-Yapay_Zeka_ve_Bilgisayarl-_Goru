"""Word Mover's Distance: cümleler arasında en düşük taşıma maliyeti."""
from argparse import ArgumentParser
from collections import Counter

import numpy as np
from scipy.optimize import linprog
from scipy.spatial.distance import cdist
from vectors import load_vectors


def wmd(vectors, first, second):
    counts = [Counter(w for w in sentence.lower().split() if w in vectors)
              for sentence in (first, second)]
    if any(not count for count in counts):
        raise ValueError("Her cümlede model sözlüğünde bulunan en az bir sözcük olmalıdır.")
    a, b = counts
    cost = cdist([vectors.get_vector(w, norm=True) for w in a],
                 [vectors.get_vector(w, norm=True) for w in b])
    m, n = cost.shape
    constraints = np.vstack([np.repeat(np.eye(m), n, axis=1), np.tile(np.eye(n), m)])
    weights = [v / count.total() for count in counts for v in count.values()]
    solved = linprog(cost.ravel(), A_eq=constraints, b_eq=weights, bounds=(0, None), method="highs")
    if not solved.success:
        raise ValueError(solved.message)
    return solved.fun


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("first", nargs="?", default="çay kahve fincan")
    parser.add_argument("second", nargs="?", default="kahve sıcak içecek")
    parser.add_argument("--model")
    args = parser.parse_args()
    try:
        print(f"WMD: {wmd(load_vectors(args.model), args.first, args.second):.4f}")
    except ValueError as error:
        parser.error(str(error))
