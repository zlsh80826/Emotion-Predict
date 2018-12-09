from collections import defaultdict
import multiprocessing
from tqdm import tqdm
import pandas as pd
import numpy as np
import unicodedata
import argparse
import nltk
import sys
import re

unicode_category = defaultdict(list)
for c in map(chr, range(sys.maxunicode + 1)):
    unicode_category[unicodedata.category(c)].append(c)
so = unicode_category['So']

invalid = ['', ' ', '\t']
separators = ["-", "\u2212", "\u2014", "\u2013", "/", "~", '"', 
              "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0"]

separators.extend(so)
splitter = '([{}])'.format(''.join(map(re.escape, separators)))

def read_df(datadir):
    train_df = pd.read_pickle('{}/train.pkl'.format(datadir))
    test_df = pd.read_pickle('{}/test.pkl'.format(datadir))

    return train_df, test_df

def tokenize(sentence):
    nltk_tokens=[t.replace("''", '"').replace("``", '"') for t in nltk.word_tokenize(sentence.lower())]
    tokens = []
    for token in nltk_tokens:
        tokens.extend([t for t in (re.split(splitter, token)) if t not in invalid])
        
    assert(not any([t == '<NULL>' for t in tokens]))
    assert(not any([' ' in t for t in tokens]))
    assert (not any(['\t' in t for t in tokens]))
    return tokens

def gen_tokenized_sentences(sentences, num_threads):
    with multiprocessing.Pool(num_threads) as pool:
        outputs = list()
        for output in tqdm(pool.imap(tokenize, sentences), total=len(sentences)):
            outputs.append(output)
    return outputs

def gen_w2v_train(train, test, datadir):
    w2v = list()
    w2v.extend(train)
    w2v.extend(test)
    np.save('{}/word2vec_train.npy'.format(datadir), w2v)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tokenize the text')
    parser.add_argument('--datadir', help='Data directory', required=False, default='../data')
    parser.add_argument('--num_threads', help='Number of threads', required=False, default=24, type=int)
    args = parser.parse_args()

    train_df, test_df = read_df(args.datadir)
    
    tokenized_train = gen_tokenized_sentences(train_df['text'], args.num_threads)
    tokenized_test = gen_tokenized_sentences(test_df['text'], args.num_threads)

    print('Number of tokenized training data:', len(tokenized_train))
    print('Number of tokenized testing data:', len(tokenized_test))

    train_df['tokenized_text'] = tokenized_train
    test_df['tokenized_text'] = tokenized_test

    train_df.to_pickle('{}/tokenized_train.pkl'.format(args.datadir))
    test_df.to_pickle('{}/tokenized_test.pkl'.format(args.datadir))

    gen_w2v_train(tokenized_train, tokenized_test, args.datadir)
