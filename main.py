from JaccardSimilarity import JaccardSimilarity

sentence_1 = 'Nazywam się Kamil.'
sentence_2 = 'Nazywam się Julia.'

jaccard = JaccardSimilarity()
result = jaccard.analyze(sentence_1, sentence_2)
print("Jaccard Similarity between '{}' and '{}': {}".format(sentence_1, sentence_2, result))

