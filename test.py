# -*- coding: utf-8 -*-

import pandas as pd
import os
from nltk.tokenize import sent_tokenize, word_tokenize
import matplotlib.pyplot as plt
from tqdm import tqdm
import nltk
import re
from sim import get_embeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# nltk.download('punkt')

import nltk
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
from nltk.corpus import movie_reviews

def tokenize(text, first_n=200, last_n=100):
    words = word_tokenize(text)
    words = [word for word in words if word not in STOPWORDS]
    n = len(words)
    # Should we remove special symbols after tokenization ?
    if n < first_n + last_n:
        return ' '.join(words)
    else:
        return ' '.join(words[:first_n]) + ' .' + ' '.join(words[-last_n:])


def main():
    df = pd.read_csv("test.csv")[['question1']].head(100)
    df['clean_question'] = df['question1'].apply(lambda x: tokenize(x))

    embeddings = get_embeddings(df['clean_question'].tolist())
    score = cosine_similarity(embeddings)

    n = 5
    top_10 = np.argsort(-score)[:,0:n]
    adf = pd.DataFrame()

    adf['No'] = np.arange(len(df))
    adf['question1'] = df['question1']
    adf['clean_question']  = df['clean_question']

    for i in range(top_10.shape[1]):
        adf["Top_{0}".format(i+1)] = top_10[:, i]

    df = adf.copy()
    for i in range(0,n):
        df = df.merge(adf[['No','question1']], left_on='Top_{0}'.format(i+1),right_on='No')
        df = df.rename(columns={
            'No_x':'No',
            'question1_y':'desc_Top_{0}'.format(i+1),
            'question1_x':'question1'}).drop('No_y',axis=1)

    df.to_csv("result.csv", index=False)
    print (f"Saved to result.csv")


if __name__ == "__main__":
    main()

