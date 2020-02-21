# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from tqdm import tqdm

wordfile = 'glove.840B.300d.txt'
weightfile = 'enwiki_vocab_min200.txt'

def getWordmap(textfile, only_these_words):
    words = dict()
    vectors = list()

    if only_these_words is not None:
        print ("Total words: {0}".format(len(only_these_words)))


    with open(wordfile, encoding='utf-8') as fp:
        # for i, line in enumerate(fp):
        for i, line in enumerate(tqdm(fp)):
            try:
                tokens = line.split()
                if only_these_words is not None:
                    if tokens[0] not in only_these_words:
                        continue
                    words[i] = tokens[0]
                    vectors.append(list(map(float, tokens[1:])))
                    only_these_words.remove(tokens[0])
                    if len(only_these_words) == 0:
                        break
                else:
                    words[i] = tokens[0]
                    vectors.append(list(map(float, tokens[1:])))
            except ValueError:
                print ("Error in processing {0}".format(line))

    if only_these_words is not None:
        print ("Missing words: {0}".format(len(only_these_words)))
        with open('missing_words.txt', 'w', encoding='utf-8') as fp:
            for item in only_these_words:
                fp.write("%s\n" % item)

    return words, np.array(vectors)

def getWordWeight(weightfile, a=1e-3):
    if a <= 0:
        a = 1.0

    df = pd.read_csv(weightfile, header=None, delimiter=' ')
    df.columns = ['word', 'count']
    df['weight'] = a / (a + (df['count'] / df['count'].sum()))

    return df

def get_all_unique_words(sentences):
    words = list()
    for s in sentences:
        for w in s.split():
            words.append(w)
    return list(set(words))

def lookupIDX(words, lower_words, w):
    if len(w) > 1 and w[0] == '#':
        w = w.replace("#", "")

    if w in words:
        return words[w]
    elif w.lower() in lower_words:
        return lower_words[w.lower()]
    elif 'UUUNKKK' in words:
        return words['UUUNKKK']
    else:
        return len(words) - 1

def sentences2idx(sentences, words, lower_words, id2weight_table):
    sentences_with_wordidx = list()
    for sent in sentences:
        sentences_with_wordidx.append(
            [lookupIDX(words, lower_words, word) for word in sent.split()])
    return prepare_data(sentences_with_wordidx, id2weight_table)

def prepare_data(list_of_seqs, id2weight_table):
    lengths = list(map(len, list_of_seqs))
    n_samples = len(list_of_seqs)
    maxlen = np.max(lengths)
    x = np.zeros((n_samples, maxlen)).astype('int32') + -1

    for idx, s in enumerate(list_of_seqs):
        x[idx, :lengths[idx]] = s
        # x_mask[idx, :lengths[idx]] = 1.

    x_weights = np.zeros((n_samples, maxlen)).astype('float32')
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i, j] != -1:
                x_weights[i, j] = id2weight_table[x[i, j]]

    return x, x_weights




class params(object):

    def __init__(self):
        self.LW = 1e-5
        self.LC = 1e-5
        self.eta = 0.05

    def __str__(self):
        t = "LW", self.LW, ", LC", self.LC, ", eta", self.eta
        t = map(str, t)
        return ' '.join(t)


import numpy as np
from sklearn.decomposition import TruncatedSVD


def get_weighted_average(We, x, w):
    n_samples = x.shape[0]
    emb = np.zeros((n_samples, We.shape[1]))
    for i in range(n_samples):
        emb[i, :] = w[i, :].dot(We[x[i, :], :]) / np.count_nonzero(w[i, :])
    return emb


def compute_pc(X, npc=1):
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_


def remove_pc(X, npc=1):
    pc = compute_pc(X, npc)
    if npc == 1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX


def SIF_embedding(We, x, w, params):
    emb = get_weighted_average(We, x, w)
    if params.rmpc > 0:
        emb = remove_pc(emb, params.rmpc)
    return emb

def get_embeddings(sentences):
    only_these_words = get_all_unique_words(sentences)
    words, vectors = getWordmap(wordfile, only_these_words)

    df_wordweights = getWordWeight(weightfile)

    df_words = pd.DataFrame(words.values(), columns=['word'])
    df_words['lowercase_word'] = df_words['word'].apply(str.lower)

    df = df_words.merge(df_wordweights, left_on='lowercase_word', right_on='word', how='left')[
        ['word_x', 'lowercase_word', 'count', 'weight']].rename(columns={'word_x': 'word'})
    df = df.fillna(1)

    words2idx_table = pd.Series(df.index, index=df['word']).to_dict()
    lower_words2idx_table = pd.Series(df.index, index=df['lowercase_word']).to_dict()

    idx2weight_table = pd.Series(df['weight'], index=df.index).to_dict()

    x, weights = sentences2idx(sentences, words2idx_table, lower_words2idx_table, idx2weight_table)

    p = params()
    p.rmpc = 1
    encodings = SIF_embedding(vectors, x, weights, p)

    return encodings

