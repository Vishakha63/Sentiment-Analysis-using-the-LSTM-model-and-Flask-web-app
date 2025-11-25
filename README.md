# Sentiment Analysis using LSTM and Flask Web Application

This project implements an end-to-end **Sentiment Analysis** system using a **Long Short-Term Memory (LSTM)** deep learning model and a **Flask web application** for real-time sentiment prediction.  
Users can enter text, and the system classifies it as **Positive**, **Negative**, or **Neutral**.

---

# Overview

Sentiment Analysis is an NLP task that determines the emotional tone behind text.  
This project combines:

- Deep Learning with **LSTM**
- Text preprocessing (tokenization, embedding, padding)
- Deployment with **Flask**
- A clean and interactive UI

---

# Applications

## Application 1: Sentiment Classifier (Web Interface)

- Users input text directly in a browser  
- The LSTM model processes the text  
- Sentiment output:
  - Positive  
  - Negative  
   

## Application 2: Model Training Module

- Train the LSTM model using custom datasets  
- Modify hyperparameters  
- Save and reload trained models  
- Extend the system with additional NLP techniques  

---

# Features

- **LSTM-based deep learning sentiment analysis**
- **Flask-powered web application**
- **Tokenization, word embedding, and sequence padding**
- **Instant sentiment prediction**
- **Fully modular and extendable codebase**

---

# Technology Stack

- **Python**
- **TensorFlow / Keras**
- **Flask**
- **HTML, CSS (Bootstrap)**
- **ATOM Editor (optional)**

---

# NLP Concepts Used

## Tokenization
Text is split into tokens and converted to integer sequences.

## Word Embedding
Words are mapped to dense vector representations for input into the LSTM network.

## Padding
All sequences are normalized to equal length before training.

---

# LSTM Model Code (Sample)

```python
model = Sequential()
model.add(Embedding(vocab_size, 128, input_length=max_length))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
