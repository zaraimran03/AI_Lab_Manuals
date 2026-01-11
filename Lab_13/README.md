# Lab 13: Natural Language Processing (NLP)

## Lab Overview
Implementation of Word2Vec on Game of Thrones text data to learn word embeddings.

## Objectives
- Understand distributional semantics
- Implement Word2Vec (Skip-Gram/CBOW)
- Explore word similarity and analogies
- Visualize word vectors

## Tasks Completed

### Word2Vec Implementation
- Downloaded Game of Thrones books dataset
- Preprocessed text using NLTK
- Tokenized sentences
- Built Word2Vec model with window=10, min_count=2
- Trained model on story corpus

### Similarity Analysis
- Found most similar words to 'daenerys'
- Checked similarity between 'arya' and 'sansa'
- Compared 'tywin' and 'sansa'
- Identified odd-one-out in character groups

### Visualization
- Applied PCA to reduce dimensions to 3D
- Plotted word vectors using Plotly
- Analyzed character name clusters
- Visualized semantic relationships

### Lab Questions
- Explained core idea of Word2Vec
- Compared CBOW vs Skip-Gram
- Discussed one-hot encoding inefficiency
- Analyzed character name proximity
- Evaluated window size effects
- Examined rare word embeddings

## Libraries Used
```python
import numpy as np
import pandas as pd
import gensim
import nltk
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.io as pio
```

## Key Learnings
- Distributional hypothesis: words in similar contexts have similar meanings
- Word embeddings capture semantic relationships
- Dense vectors vs sparse one-hot encoding
- Skip-Gram better for rare words
- CBOW faster for large datasets
- Dimensionality reduction for visualization