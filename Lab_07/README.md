# Lab 7: Implementation of Recurrent Neural Network (RNN)

## Lab Overview
Implementation of Recurrent Neural Networks for sequential data processing including text, time series, and music generation.

## Objectives
- Understand RNN architecture for sequences
- Implement RNNs for various tasks
- Process temporal data
- Generate sequences

## Tasks Completed

### Task 1: Next Word Prediction
- Loaded Shakespeare text corpus
- Tokenized and created sequences
- Built SimpleRNN model
- Predicted next word given 3-5 previous words
- Tested with custom prompts

### Task 2: Stock Price Prediction
- Used Google Stock Price dataset
- Prepared 60-day sequences
- Built SimpleRNN for time series
- Plotted actual vs predicted prices
- Evaluated prediction accuracy

### Task 3: Sentiment Analysis (IMDb)
- Loaded IMDb Movie Reviews dataset
- Preprocessed text (tokenize, pad sequences)
- Built RNN with Embedding + SimpleRNN
- Classified positive/negative reviews
- Evaluated test accuracy

### Task 4: Weather Forecasting
- Used daily temperature dataset
- Created time-step sequences
- Trained RNN to predict next day temperature
- Plotted actual vs predicted temperatures
- Analyzed prediction errors

### Task 5: Music Note Generation
- Converted MIDI data to integer sequences
- Trained RNN on note sequences
- Generated new music sequences
- Exported as MIDI file
- Analyzed generated patterns

## Libraries Used
```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from music21 import note, chord, stream
import matplotlib.pyplot as plt
```

## Key Learnings
- RNNs maintain memory of previous inputs
- Sequential data requires special preprocessing
- Embedding layers for text representation
- Time series forecasting techniques
- RNN limitations (vanishing gradient problem)
- Applications in NLP, finance, and music