**🎯 Objective:**  
Implement RNNs for sequence modeling tasks including text prediction, time series forecasting, and music generation.

**📚 Topics Covered:**
- Simple RNN architecture
- Sequence-to-sequence learning
- Time series prediction
- Sentiment analysis
- Text and music generation

**🔧 Key Technologies:**
- `keras.layers.SimpleRNN` - RNN implementation
- `keras.preprocessing` - Sequence handling
- `music21` - Music notation

**✅ Tasks Completed:**

**Task 1: Next Word Prediction**
- Corpus: Custom text (sun, weather phrases)
- Sequence length: 4 words
- Architecture: Embedding → SimpleRNN → Dense
- Example: "the sun is" → predicts "shining"

**Task 2: Stock Price Prediction**
- Dataset: Historical stock prices (60-day sequences)
- Features: Daily closing prices
- Architecture: SimpleRNN(50) → Dense(1)
- Visualization: Actual vs Predicted prices
- Pattern: Successfully captured trends

**Task 3: Sentiment Analysis**
- Dataset: IMDb movie reviews (25,000 samples)
- Vocab size: 10,000 words
- Max length: 200 words
- Architecture: Embedding(128) → SimpleRNN(64) → Dense(1, sigmoid)
- Accuracy: **85%+**
- Epochs: 5

**Task 4: Weather Forecasting**
- Dataset: Daily temperature (Jena Climate)
- Sequence length: 30 days
- Target: Next day temperature
- Architecture: SimpleRNN(50) → Dense(1)
- Metrics: MAE, RMSE
- Visualization: Temperature prediction curve

**Task 5: Music Note Generation**
- Input: MIDI note sequences
- Notes: C4, D4, E4, F4, G4, chords
- Architecture: SimpleRNN(128) → Dense(softmax)
- Output: Generated melody sequence
- Epochs: 100

**📊 Performance Results:**

Task                        Performance
──────────────────────────────────────────
Next Word Prediction        Context-aware
Stock Price                 Trend captured
Sentiment Analysis          85.3% accuracy
Weather Forecasting         Low MAE
Music Generation            Coherent melody


# Visualizations:

- Stock price prediction curves
- Sentiment classification examples
- Temperature forecast graphs
- Generated music notation

# ⚠️ Limitations Observed:

- Vanishing gradient in long sequences
- Limited long-term memory
- Better results with LSTM (Lab 8)

**🎓 Key Learnings:**

- Sequential data processing
- RNN architecture design
- Handling variable-length sequences
- Time series prediction techniques