from cosine.CosineSimilarity import CosineSimilarity
from cosine.SifCosineSimilarity import SifCosineSimilarity
from embeddings.EmbeddingModelWrapper import EmbeddingModelWrapper
from ensembled.CustomEnsembleSimilarity import CustomEnsembleSimilarity
from jaccard.JaccardSimilarity import JaccardSimilarity
from wmd.WordMoverDistance import WordMoverDistance

import argparse


def main():
    default_algo = "jaccard"
    nkjp_wiki_path = "embeddings/nkjp+wiki-lemmas-all-100-cbow-hs.txt"

    parser = argparse.ArgumentParser()
    parser.add_argument("sentence1", help='Pierwsze zdanie do porównania.', type=str)
    parser.add_argument("sentence2", help='Drugie zdanie do porównania.', type=str)
    parser.add_argument("-a", "--algorithm", help="Użyty algorytm (jeden z: {})".format(algos.keys()),
                        type=str, default=default_algo)
    parser.add_argument("-e", "--embedding", help="Ścieżka do pliku z modelem word2vec.",
                        type=str, default=nkjp_wiki_path)
    args = parser.parse_args()

    embedding = EmbeddingModelWrapper(args.embedding, 100)

    algos = {
        "jaccard": JaccardSimilarity(),
        "wmd": WordMoverDistance(embedding),
        "cosine": CosineSimilarity(embedding),
        "ensemble": CustomEnsembleSimilarity(embedding),
        "sif-cosine": SifCosineSimilarity(embedding)
    }

    result = algos[args.algorithm].normalized_score(args.sentence1, args.sentence2)
    print("Zdanie 1: {}".format(args.sentence1))
    print("Zdanie 2: {}".format(args.sentence2))
    print("Indeks podobieństwa: {}%".format(result))


if __name__ == "__main__":
    main()
