import string

from SemanticSimilarityAlgorithm import SemanticSimilarityAlgorithm
from stempel import StempelStemmer

class JaccardSimilarity(SemanticSimilarityAlgorithm):

    def __init__(self):
        super().__init__()
        self.stemmer = StempelStemmer.default()
        self.punct_translator = str.maketrans('', '', string.punctuation)

    def analyze(self, sentence_1, sentence_2):
        stemmed_1 = self._stem(sentence_1.translate(self.punct_translator))
        stemmed_2 = self._stem(sentence_2.translate(self.punct_translator))
        print(stemmed_1, stemmed_2)

        intersection = stemmed_1.intersection(stemmed_2)
        union = stemmed_1.union(stemmed_2)
        return len(intersection) / len(union)

    def _stem(self, sentence):
        res = set()
        for word in sentence.split():
            res.add(self.stemmer.stem(word))
        return res
