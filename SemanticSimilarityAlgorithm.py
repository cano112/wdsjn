from abc import abstractmethod


class SemanticSimilarityAlgorithm:

    @abstractmethod
    def analyze(self, sentence_1, sentence_2):
        pass
