# This is a sample Python script.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Activation, MaxPool1D, LSTM, Dropout, Flatten
from tensorflow.python.keras.layers import SpatialDropout1D
from collections import Counter

# load dataset
df = pd.read_csv(r"MoviesDataset.csv")
print(df.head(5))


def plot_wine_quality_histogram(sentiment):  # quality histogram
    unique_vals = df['Sentiment'].sort_values().unique()
    plt.xlabel("Values")
    plt.ylabel("Count")
    plt.hist(sentiment.values, align='left')
    plt.show()
    print(df['Sentiment'].value_counts())
    print(sentiment.values)


plot_wine_quality_histogram(df['Sentiment'])
# create a sequence of words
reviews = df['Summary'].values
all_text = ''.join([c for c in reviews])
reviews_split = all_text.split('\n')


all_text2 = ' '.join(reviews_split)
print('Number of reviews :', len(all_text2))
# create a list of words
words = all_text2.split()  # Count all the words using Counter Method
# Build a dictionary that maps words to integers
count_words = Counter(words)

total_words = len(words)
sorted_words = count_words.most_common(total_words)
print(count_words)
X = df['Summary'].values
Y = df['Sentiment'].values

# Training Own Embedding

# convert them into vectors
tokenizer = Tokenizer(num_words=10000)
# Create the vocabulary index based on word frequency
tokenizer.fit_on_texts(X)
# number of the unique words based on the number of elements in this dictionary
vocab_size = len(tokenizer.word_index) + 1
# assign an integer to each word
x_train = tokenizer.texts_to_sequences(X)
# pad all our reviews to a specific length
padded_sequence = pad_sequences(x_train, maxlen=200)
print(padded_sequence)
maxlen = 200

# build model
model = Sequential()
model.add(Embedding(vocab_size, maxlen, input_length=200))
model.add(SpatialDropout1D(0.25))
model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])


model.summary()
history = model.fit(padded_sequence, Y, epochs=5, validation_split=0.2, verbose=True, batch_size=32)
# test_word = "This is a good movie"
# tw = tokenizer.texts_to_sequences([test_word])
# tw = pad_sequences(tw, maxlen=200)
# prediction = int(model.predict(tw).round().item())
# print(Y[prediction])
