# Lab 8: Implementation of Long Short-Term Memory (LSTM) Network

## Lab Overview
Implementation of LSTM networks to overcome RNN limitations and handle long-term dependencies.

## Objectives
- Understand LSTM architecture and gates
- Implement LSTM for sequence tasks
- Compare LSTM vs simple RNN
- Apply to text generation

## Tasks Completed

### Question 1: LSTM Text Predictor
- Created LSTM model for next-word prediction
- Used Game of Thrones FAQ text
- Tokenized and padded sequences
- Trained model with categorical crossentropy
- Generated word sequences

### Question 2: Custom Text Dataset
- Loaded custom text dataset
- Built LSTM architecture
- Trained for next-word prediction
- Tested with user input sentences
- Evaluated prediction quality

### Question 3: Generate Multiple Words
- Modified code to generate 5 words sequentially
- Used trained LSTM model
- Created prediction loop
- Displayed generated sequences
- Analyzed coherence

## Libraries Used
```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import pandas as pd
```

## Key Learnings
- LSTM gates (forget, input, output) control information flow
- Cell state enables long-term memory
- LSTM solves vanishing gradient problem
- Better than RNN for long sequences
- Hyperparameters: units, return_sequences
- Applications in language modeling, translation