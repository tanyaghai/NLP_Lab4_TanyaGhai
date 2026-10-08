import unittest
import spacy
from HmmTagger import *

nlp = spacy.load('en_core_web_sm')

class TestTagger(unittest.TestCase):

    TRAIN_DIR = "/courses/cs159/data/pos/wsj/train"

    def testPennTreebankTags(self):
        tagger = HMMTagger(nlp, alpha=0.1)
        tagger.train(self.TRAIN_DIR)

        test_doc_1 = nlp("These moments will be lost in time like tears in rain.")
        tagger.predict(test_doc_1)
        result_1 = [t.tag_ for t in test_doc_1]
        self.assertEqual(result_1, ['DT', 'NNS', 'MD', 'VB', 'VBN', 'IN', 'NN', 'IN', 'NNS', 'IN', 'NN', '.'])

        test_doc_2 = nlp("Stand clear of the closing doors, please!")
        tagger.predict(test_doc_2)
        result_2 = [t.tag_ for t in test_doc_2]
        self.assertEqual(result_2, ['VB', 'JJ', 'IN', 'DT', 'NN', 'NNS', ',', 'RB', '.'])

        test_doc_3 = nlp("Osmanthus wine tastes the same as I remember, but where are those who share the memory?")
        tagger.predict(test_doc_3)
        result_3 = [t.tag_ for t in test_doc_3]
        self.assertEqual(result_3, ['DT', 'NN', 'VBZ', 'DT', 'JJ', 'IN', 'PRP', 'VBP', ',', 'CC', 'WRB', 'VBP', 'DT', 'WP', 'VBP', 'DT', 'NN', '.'])

    def testUniversalTags(self):
        tagger = HMMTagger(nlp, alpha=0.1, do_universal=True)
        tagger.train(self.TRAIN_DIR)

        test_doc_1 = nlp("These moments will be lost in time like tears in rain.")
        tagger.predict(test_doc_1)
        result_1 = [t.tag_ for t in test_doc_1]
        self.assertEqual(result_1, ['DET', 'NOUN', 'VERB', 'VERB', 'VERB', 'ADP', 'NOUN', 'ADP', 'NOUN', 'ADP', 'NOUN', 'PUNCT'])

        test_doc_2 = nlp("Stand clear of the closing doors, please!")
        tagger.predict(test_doc_2)
        result_2 = [t.tag_ for t in test_doc_2]
        self.assertEqual(result_2, ['VERB', 'ADJ', 'ADP', 'DET', 'NOUN', 'NOUN', 'PUNCT', 'VERB', 'PUNCT'])

        test_doc_3 = nlp("Osmanthus wine tastes the same as I remember, but where are those who share the memory?")
        tagger.predict(test_doc_3)
        result_3 = [t.tag_ for t in test_doc_3]
        self.assertEqual(result_3, ['DET', 'NOUN', 'VERB', 'DET', 'ADJ', 'ADP', 'PRON', 'VERB', 'PUNCT', 'CONJ', 'ADV', 'VERB', 'DET', 'PRON', 'VERB', 'DET', 'NOUN', 'PUNCT'])

if __name__ == '__main__':
    unittest.main()
