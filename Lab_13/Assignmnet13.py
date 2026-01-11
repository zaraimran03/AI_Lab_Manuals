print("\n--- Q1: game-of-thrones-word2vec ---\n")

import numpy as np
import pandas as pd
import os

import gensim
import nltk
from nltk import sent_tokenize
from gensim.utils import simple_preprocess
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.io as pio

nltk.download('punkt')
nltk.download('punkt_tab')

DATA_PATH = r"DataFiles\001ssb.txt"

story = []
try:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        corpus = f.read()
except UnicodeDecodeError:
    with open(DATA_PATH, "r", encoding="cp1252") as f:
        corpus = f.read()

for sent in sent_tokenize(corpus):
    story.append(simple_preprocess(sent))

print(len(story))
print(story[:2])

model = gensim.models.Word2Vec(
    window=10,
    min_count=2
)
model.build_vocab(story)
model.train(story, total_examples=model.corpus_count, epochs=model.epochs)

print(model.wv.most_similar('daenerys'))
print(model.wv.doesnt_match(['jon','rikon','robb','arya','sansa','bran']))
print(model.wv.doesnt_match(['cersei', 'jaime', 'bronn', 'tyrion']))
print(model.wv['king'])
print(model.wv.similarity('arya','sansa'))
print(model.wv.similarity('tywin','sansa'))

vectors = model.wv.get_normed_vectors()
y = model.wv.index_to_key
print(len(y))
print(y)

pca = PCA(n_components=3)
X = pca.fit_transform(vectors)
print(X.shape)

pio.renderers.default = "browser"
df = pd.DataFrame(X[200:300], columns=["x", "y", "z"])
df["label"] = y[200:300]

fig = px.scatter_3d(
    df,
    x="x",
    y="y",
    z="z",
    color="label"
)
fig.show()
