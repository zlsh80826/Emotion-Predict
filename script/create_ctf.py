import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import zip_longest
from gensim.models import FastText
from collections import defaultdict
import argparse

def write_ctf(datadir, df, type_, vocabs_dict, emotion_dict):
    word_size = 12
    pad_spec = '{0:<%d.%d}' % (word_size, word_size)
    sanitize = str.maketrans({"|": None, "\n": None})
    with open(datadir + '/{}.ctf'.format(type_), 'w', encoding='utf-8') as file:
        for idx, row in tqdm(enumerate(df.itertuples()), desc='Writing {}.ctf'.format(type_)):
            if type_.find('test') < 0:
                text = row[-1]
                emotion = [row[-2]]
                for t, e in zip_longest(text, emotion):
                    out = [str(idx)]
                    if t is not None:
                        out.append('|# %s' % pad_spec.format(t.translate(sanitize)))
                        out.append('|token {}:{}'.format(vocabs_dict[t], 1))
                    if e is not None:
                        out.append('|# %s' % pad_spec.format(e.translate(sanitize)))
                        out.append('|emotion {}:{}'.format(emotion_dict[e], 1))
                    file.write('\t'.join(out))
                    file.write('\n')
            else:
                text = row[-1]
                for t in text:
                    out = [str(idx)]
                    out.append('|# %s' % pad_spec.format(t.translate(sanitize)))
                    out.append('|token {}:{}'.format(vocabs_dict[t], 1))
                    file.write('\t'.join(out))
                    file.write('\n')

def gen_vocabs_dict(embedding):
    model = FastText.load(embedding)
    vocabs = defaultdict(int)
    vocabs['<UNK>'] = 0
    counter = 1

    for v in model.wv.vocab:
        vocabs[v] = counter
        counter += 1

    return vocabs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create the ctf')
    parser.add_argument('--datadir', help='Data directory', required=False, default='../data')
    parser.add_argument('--embedding', help='Embedding path', required=False, default='../embedding/fasttext.bin')
    parser.add_argument('--num_valid', help='Number of validation', required=False, type=int, default=10000)
    args = parser.parse_args()
    
    vocabs = gen_vocabs_dict(args.embedding)

    labeled_df = pd.read_pickle('{}/tokenized_train.pkl'.format(args.datadir))
    test_df = pd.read_pickle('{}/tokenized_test.pkl'.format(args.datadir))

    uni_emotion = pd.unique(labeled_df['emotion'])
    emotion_dict = dict()
    counter = 0

    for e in uni_emotion:
        emotion_dict[e] = counter
        counter += 1

    valid_df = labeled_df.sample(args.num_valid)
    train_df = labeled_df.drop(valid_df.index)

    write_ctf(args.datadir, valid_df, 'valid', vocabs, emotion_dict)
    write_ctf(args.datadir, train_df, 'train', vocabs, emotion_dict)
    write_ctf(args.datadir, test_df, 'test', vocabs, emotion_dict)
