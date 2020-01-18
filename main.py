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

    jaccard = JaccardSimilarity()
    cosine = CosineSimilarity(embedding)
    wmd = WordMoverDistance(embedding)
    sif_cosine = SifCosineSimilarity(embedding)
    ensemble = CustomEnsembleSimilarity([
        (jaccard, 0.1),
        (sif_cosine, 0.3),
        (wmd, 0.6)
    ])

    algos = {
        "jaccard": jaccard,
        "wmd": wmd,
        "cosine": cosine,
        "ensemble": ensemble,
        "sif-cosine": sif_cosine
    }

    result = algos[args.algorithm].normalized_score(args.sentence1, args.sentence2)
    print("Zdanie 1: {}".format(args.sentence1))
    print("Zdanie 2: {}".format(args.sentence2))
    print("Indeks podobieństwa: {}%".format(result))


if __name__ == "__main__":
    main()
