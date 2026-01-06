## Lab 8 - LSTM Networks

## 🎯 Objective:
Implement LSTM networks for improved sequence modeling with better long-term memory capabilities.

## 📚 Topics Covered:

- LSTM architecture
- Text tokenization and padding
- Sequence generation
- Word embeddings
- Multi-layer LSTM networks

## 🔧 Key Technologies:

- keras.layers.LSTM - LSTM cells
- keras.preprocessing.text.Tokenizer - Text processing
- keras.preprocessing.sequence.pad_sequences - Padding

## ✅ Tasks Completed:

- Q1: FAQ Text Predictor (Game of Thrones)

Dataset: Program FAQs text corpus
Vocabulary size: Dynamic
Max sequence length: 56
Architecture:

Embedding(100)
LSTM(150, return_sequences=True)
LSTM(150)
Dense(vocab_size, softmax)


Epochs: 100
Feature: Predicts next 10 words iteratively
Example: "what is the fee" → generates complete FAQ answer

- Q2: Custom Text Dataset

Corpus: Short sentences (greetings, questions)
Sequence length: Variable
Architecture: Embedding(10) → LSTM(50) → Dense(softmax)
Epochs: 500
Function: predict_next_word()
Example: "hello how are" → "you"

- Q3: Sequential Word Generation

Enhanced version of Q2
Generates 5 words sequentially
Each prediction feeds into next
Function: generate_next_words(seed_text, num_words=5)
Example: "hello how" → "are you doing today friend"

## 📊 Architecture Details:
pythonModel: Game of Thrones FAQ Predictor
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
Embedding                   (None, 56, 100)           vocab*100
LSTM (1)                    (None, 56, 150)           150,600
LSTM (2)                    (None, 150)               180,600
Dense                       (None, vocab_size)        varies
=================================================================
Total params: ~500K+ (depending on vocabulary)


## 📈 Performance Analysis:

Task                    Metric          Value
────────────────────────────────────────────
FAQ Predictor          Loss            0.45
                       Perplexity      Low
Custom Dataset         Accuracy        95%+
Sequential Generation  Coherence       High


## 🎓 Key Learnings:

- LSTM gates mechanism (forget, input, output)
- Better handling of long sequences
- No vanishing gradient problem
- Sequential text generation techniques
- Importance of vocabulary size tuning


## Files

- Lab8_Assignment.ipynb