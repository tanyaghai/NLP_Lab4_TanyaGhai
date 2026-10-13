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

    def __init__(self, nlp, alpha=0.1, do_universal=False, vocabsize=None):

        self.do_universal = do_universal
        self.vocabsize = vocabsize
        if self.do_universal:
            self.tags = ["<<START>>"] + read_tags.universal_tag_set
        else:
            self.tags = ["<<START>>"] + read_tags.ptb_tag_set
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
        if self.vocabsize is None:
            self.vocab += list(token_counter.keys())
        else:
            self.vocab += [word for word, count in token_counter.most_common(self.vocabsize)]
    
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
        """
        returns the log probabilties of the starting tags from the transition probabilities matrix
        this is the first row of the transition matrix, which corresponds to the <<START>> tag

        """
        return self.tag_tag_probs[self.tag_to_index("<<START>>"),:]

    def get_token_costs(self, token):
        """
        returns the log probabilties of the token from the emission probabilities matrix for all possible tags

        """
        return self.tag_word_probs[:,self.word_to_index(token.text)]
    
    def normalize(self, m):
        """
        normalizes the matrix m in log space by subtracting the log of the sum of each
        """
        return (log(m).transpose() - log(m.sum(axis=1))).transpose()

    def do_train_sent(self, words, tags): 
        """
        for a given sequence of words, the function updates the emission and transition probability matrices by counting the occurrences of 
        each word+tag pair and tag+tag pair in the training data
        """
        prev_tag = self.tag_to_index("<<START>>") #keeping track of the previous tag 
        for word, tag in zip(words, tags): 
            t_i = self.tag_to_index(tag) 
            w_i = self.word_to_index(word)
            self.tag_word[t_i][w_i] += 1 #updating the emission probability matrix
            self.tag_tag[prev_tag][t_i] += 1 #updating the transition probability matrix
            prev_tag = t_i 

    def train(self, train_dir):
        """
        this is the main function that does the training 
        it updates the vocabulay, initializes the emission and transition matrices,
        and then it uses the previous function do_train_sent to update the matrices with the appropriate counts 
        finally it normalizes the matrices 
        """
        self.update_vocab(train_dir) #just populates the vocab with all the words in the training data


        self.tag_word  = ones((len(self.tags), len(self.vocab))) * self.alpha # this is the emission probability matrix
        self.tag_tag =  ones((len(self.tags), len(self.tags))) * self.alpha # this is the transition probability matrix

        for words, tags in read_tags.parse_dir(train_dir, do_universal=self.do_universal):
            self.do_train_sent(words, tags) #update the matrices with the counts of the words and tags in the training data
        self.normalize_probabilities() #normalizing the counts 

    def predict(self, tokens):
        """
        creates 2 tables, one for the probabilities of the tags for each word in the sentence and one to retrace the best path through the DP table so we can return the sequence of tags
        updates the cost_table with the probabilities of the tags for each word in the sentence 
        still yet to finish the implementation 

        """
        # Build DP table, which should be |sent| x |tags|
        cost_table = zeros((len(tokens), len(self.tags)), float32) #this is the probabilities of the tags for each word in the sentence
        bt_table   = zeros((len(tokens), len(self.tags)), int) #this is to retrace the best path through the DP table so we can return the sequence of tags 

        for token_i, token in enumerate(tokens):
            token_costs = self.get_token_costs(token)
            if token_i == 0: 
                cost_table[token_i, :] = self.get_start_costs() + token_costs
                bt_table[token_i, :] = -1
            else:


                # Fill in the actual costs matrix to compute
                # the sum of the log probability from the last state,
                # the transition log probability,
                # and the emission log probability

                costs = self.tag_tag_probs.copy() #making copy of the transition probabilities matrix
                costs = costs +token_costs #adding the emission probabilities to the transition probabilities
                previous_scores = cost_table[token_i - 1, :] #getting the previous scores from the cost_table
                costs += previous_scores.reshape(-1, 1)
                
                cost_table[token_i, :] = costs.max(axis=0)
                bt_table[token_i,:] = costs.argmax(axis=0)

        # Find the highest-probability tag for last word
        best_last_tag = argmax(cost_table[token_i, :])

        # Trace back through the breadcrumb table
        self.backtrace(bt_table, tokens, best_last_tag)

    def backtrace(self, bt_table, tokens, best_last_tag):
        """Trace back through the bt table to find the best sequence of tags for the given tokens"""
        current_row = len(tokens)-1 #starting from the last row of the bt_table 
        for t in list(tokens)[::-1]: #going through the tokens in reverse order
            t.tag_ = self.tags[best_last_tag] #assigning the best tag to the token
            best_last_tag = bt_table[current_row, best_last_tag] #updating the best_last_tag to the previous tag in the sequence
            current_row -= 1 #decrementing the current_row to move up the bt_table


def main(args):
    nlp = spacy.load("en_core_web_sm")
    tagger = HMMTagger(nlp, alpha=args.alpha, do_universal=args.universal,  vocabsize=args.vocabsize)
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

    parser.add_argument( "--universal", action="store_true",
                         help="Use Universal tags instead of PTB tags")

    parser.add_argument("--vocabsize", "-v", type=int, default=None,
            help="Keep only this many of the most frequent words")

    args = parser.parse_args()
    main(args)
