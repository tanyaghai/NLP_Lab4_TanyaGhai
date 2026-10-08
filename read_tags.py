import os
import re
import sys


universal_to_ptb = {'SYM': ('#', '$', 'SYM'),
                    'PUNCT': ('"', ',', '(', ')', '.', ':', 'HYPH', '``', "''"),
                    'ADJ': ('AFX', 'JJ', 'JJR', 'JJS', 'JJSS'), 
                    'CONJ': ('CC',), 
                    'NUM': ('CD',), 
                    'DET': ('DT','PDT', 'PRP$', 'WDT', 'WP$', 'PRP$R'), 
                    'PRON': ('EX', 'PRP', 'WP'), 
                    'X': ('FW', 'LS', 'NIL'), 
                    'ADP': ('IN', 'RP'), 
                    'VERB': ('MD', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'), 
                    'NOUN': ('NN', 'NNS'), 
                    'PROPN': ('NNP', 'NNPS'),
                    'PART': ('TO', 'POS'), 
                    'INTJ': ('UH', ), 
                    'ADV': ('RB', 'RBR', 'RBS', 'WRB',)}

universal_tag_set = list(universal_to_ptb.keys())

ptb_tag_set = [v for vals in universal_to_ptb.values() for v in vals]


def get_files(dirname): 
    """A generator: recursively navigates a directory, yielding each file's
    full filepath instead of just the filename."""
    for root, dirs, files in os.walk(dirname): 
        for fname in files: 
            yield os.path.join(root, fname)


def parse_file(fp, separator="===+", do_universal=False): 
    """A generator: given a file pointer for a file containing
    POS-tagged text, yield pairs of (words, tags) where each
    of words and tags is a list of strings."""
    tagged_text = fp.read()
    sentences = re.split(separator, tagged_text)[1:]

    for sent in sentences: 
        words = [x.rsplit("/", maxsplit=1) for x in sent.split() if "/" in x]
        words = [(word, tag) for word, tag in words if word.strip() and tag.strip()]

        if not(words):
            continue
        words, tags = list(zip(*words))
        tags = [t.split("|")[0] for t in tags]
        yield words, tags


def parse_dir(dirname, do_universal=False):
    """A wrapper for parse_file that runs across all files
    in a particular directory."""
    for fname in get_files(dirname):
        with open(fname) as fp:
            for words, tags in parse_file(fp, do_universal=do_universal):
                yield words, tags


if __name__ == "__main__": 
    # Parses the file and prints the first ten instances in the file.
    with open(sys.argv[1]) as f:
        print(list(parse_file(f))[:10])
