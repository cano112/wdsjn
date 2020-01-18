import numpy as np

from common.SemanticSimilarityAlgorithm import SemanticSimilarityAlgorithm
from cosine.SifCosineSimilarity import SifCosineSimilarity
from jaccard.JaccardSimilarity import JaccardSimilarity
from wmd.WordMoverDistance import WordMoverDistance


class CustomEnsembleSimilarity(SemanticSimilarityAlgorithm):

    def __init__(self, embedding):
        super().__init__()
        self.algos = [
            (WordMoverDistance(embedding), 0.6),
            (SifCosineSimilarity(embedding), 0.3),
            (JaccardSimilarity(), 0.1),
        ]

    def _absolute_score(self, sentence_1, sentence_2):
        return np.average(
            [algo.normalized_score(sentence_1, sentence_2) for algo, weight in self.algos],
            weights=[weight for algo, weight in self.algos])

    def _normalize(self, absolute):
        return absolute
