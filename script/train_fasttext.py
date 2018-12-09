import argparse
import numpy as np
from gensim.models import FastText

sentences = np.load('../data/word2vec_train.npy')
print('Number of training sentences:', len(sentences))

model = FastText(sentences, size=300, workers=24, iter=150, sg=1)
model.save('../embedding/fasttext.bin')
