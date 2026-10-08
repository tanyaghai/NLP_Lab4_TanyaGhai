#!/usr/bin/env python3
import argparse
from collections import defaultdict, Counter
import os
import pickle
import sys

from numpy import argmax, zeros, array, float32, ones, zeros, log
import spacy

import read_tags


class HMMTagger():

    def __init__(self, nlp, alpha=0.1):
        self.tags = ["<<START>>",] + read_tags.ptb_tag_set
        self.vocab = ["<<OOV>>",]
        self.alpha = alpha

    def clean_token(self, text):
        """Convert each token to lower-case text or a special OOV token
        for out of vocabulary words"""
        return text.lower() if text.lower() in self.vocab else "<<OOV>>"

    def word_to_index(self, w):
        """Given a token, find its index in the vocabulary."""
        return self.vocab.index(self.clean_token(w))

    def tag_to_index(self, t):
        """Given a tag, find its index in the tag list."""
        return self.tags.index(t)

    def update_vocab(self, train_dir):
        """Given a directory of files, populate the vocabulary that corresponds
        to all tokens in the files in the directory."""
        token_counter = Counter()
        for words, _ in read_tags.parse_dir(train_dir): 
            token_counter.update([w.lower() for w in words])
        self.vocab += list(token_counter.keys())
    
    def normalize_probabilities(self):
        """Normalize the tag-word and tag-tag probability matrices
        in log space."""
        self.tag_word_probs = self.normalize(self.tag_word)
        self.tag_tag_probs = self.normalize(self.tag_tag)

    def __call__(self, tokens):
        """If invoked as a function call, predicts tags for a list
        of tokens given the trained model."""
        self.predict(tokens)
        
    ## BEGIN DOCUMENTATION HERE ###

    def get_start_costs(self):
        """TODO: docstring"""
        return self.tag_tag_probs[self.tag_to_index("<<START>>"),:]

    def get_token_costs(self, token):
        """TODO: docstring"""
        return self.tag_word_probs[:,self.word_to_index(token.text)]
    
    def normalize(self, m):
        """TODO: docstring"""
        return (log(m).transpose() - log(m.sum(axis=1))).transpose()

    def do_train_sent(self, words, tags): 
        """TODO: docstring"""
        prev_tag = self.tag_to_index("<<START>>")
        for word, tag in zip(words, tags):
            t_i = self.tag_to_index(tag)
            w_i = self.word_to_index(word)
            self.tag_word[t_i][w_i] += 1
            self.tag_tag[prev_tag][t_i] += 1
            prev_tag = t_i

    def train(self, train_dir):
        """TODO: docstring"""
        self.update_vocab(train_dir)

        # TODO: which of these gives transition probabilities,
        # and which gives emission probabilities?
        self.tag_word  = ones((len(self.tags), len(self.vocab))) * self.alpha
        self.tag_tag =  ones((len(self.tags), len(self.tags))) * self.alpha

        for words, tags in read_tags.parse_dir(train_dir):
            self.do_train_sent(words, tags)
        self.normalize_probabilities()

    def predict(self, tokens):
        """TODO: docstring"""
        # Build DP table, which should be |sent| x |tags|
        cost_table = zeros((len(tokens), len(self.tags)), float32)
        bt_table   = zeros((len(tokens), len(self.tags)), int)

        for token_i, token in enumerate(tokens):
            token_costs = self.get_token_costs(token)
            if token_i == 0: 
                cost_table[token_i, :] = self.get_start_costs() + token_costs
                bt_table[token_i, :] = -1
            else:
                costs = self.tag_tag_probs.copy()
                # TODO: Fill in the actual costs matrix to compute
                # the sum of the log probability from the last state,
                # the transition log probability,
                # and the emission log probability
                
                cost_table[token_i, :] = costs.max(axis=0)
                bt_table[token_i,:] = costs.argmax(axis=0)

        # Find the highest-probability tag for last word
        best_last_tag = argmax(cost_table[token_i, :])

        # Trace back through the breadcrumb table
        self.backtrace(bt_table, tokens, best_last_tag)

    def backtrace(self, bt_table, tokens, best_last_tag):
        """TODO: docstring"""
        current_row = len(tokens)-1
        for t in list(tokens)[::-1]:
            t.tag_ = self.tags[best_last_tag]
            best_last_tag = bt_table[current_row, best_last_tag]
            current_row -= 1


def main(args):
    nlp = spacy.load("en_core_web_sm")
    tagger = HMMTagger(nlp, alpha=args.alpha)
    tagger.train(args.dir)
    pickle.dump(tagger, args.output)


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Train (and save) hmm models for POS tagging')
    parser.add_argument("--dir", "-d", metavar="DIR", required=True,
                        help="Read training data from DIR")
    parser.add_argument("--output", "-o", metavar="FILE", 
                        type=argparse.FileType('wb'), required=True,
                        help="Save output to FILE")
    parser.add_argument("--alpha", "-a", default=0.1, 
                        help="Alpha value for add-alpha smoothing")

    args = parser.parse_args()
    main(args)
