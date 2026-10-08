#!/usr/bin/env python3
import argparse
import os
import pickle
import spacy
import sys
from spacy.tokens import Doc

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from HmmTagger import HMMTagger
from read_tags import *


def main(args): 
    nlp = spacy.load("en_core_web_sm")
   
    tagger = pickle.load(args.hmm)
    
    total_right = 0
    total_size = 0 
    for words, tags in parse_dir(args.dir): 
        doc = Doc(nlp.vocab, words=list(words))
        tagger(doc)
        total_right += sum([spacy_token.tag_ == ref_tag 
                       for spacy_token, ref_tag in zip(doc, tags)])
        total_size += len(doc)

    print("Accuracy: {:.2%}".format(total_right/total_size))


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='POS Tag, then evaluate')
    parser.add_argument("--dir", "-d", metavar="DIR", required=True,
                        help="Read data to tag from DIR")
    parser.add_argument("--hmm", metavar="FILE", 
                        type=argparse.FileType('rb'), required=True,
                        help="Read hmm model from FILE")
    args = parser.parse_args()
    main(args)
 
