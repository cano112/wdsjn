from common.SemanticSimilarityAlgorithm import SemanticSimilarityAlgorithm
from common.utils import preprocess


class WordMoverDistance(SemanticSimilarityAlgorithm):

    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding

    def _absolute_score(self, sentence_1, sentence_2):
        splitted_1 = preprocess(sentence_1)
        splitted_2 = preprocess(sentence_2)
        return self.embedding.model.wmdistance(splitted_1, splitted_2)

    def _normalize(self, absolute):
        # We assume that max distance is 1.5
        # TODO: come up with better normalization
        normalized = -(absolute * 200 / 3) + 100
        return min(normalized, 100)
