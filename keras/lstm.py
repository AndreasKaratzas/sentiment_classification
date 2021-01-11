from wordcloud import WordCloud, STOPWORDS
import pandas as pd
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Dropout
from tensorflow.python.keras.layers import SpatialDropout1D
from collections import Counter
from keras import backend as K

# load dataset
df = pd.read_csv(r"MoviesDataset.csv")
print(df.head(10))
ps = PorterStemmer()


# create reviews length histogram
def plot_sentiment_histogram(sentiment):
    unique_vals = df['Sentiment'].sort_values().unique()
    plt.xlabel("Values")
    plt.ylabel("Count")
    plt.hist(sentiment.values, align='left')
    plt.show()
    print(df['Sentiment'].value_counts())
    print(sentiment.values)


# plot histogram
plot_sentiment_histogram(df['Sentiment'])

X = df['Summary'].values
Y = df['Sentiment'].values

# split dataset in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# get the max length of a review (number of words in this sentence)
max_length = max([len(s.split()) for s in X])
# get the min length of a review (number of words in this sentence)
min_length = min([len(s.split()) for s in X])
print(max_length)
print(min_length)

# get the length of every review in the dataset
reviews_len = [len(x.split()) for x in X]
# create review length histogram
pd.Series(reviews_len).hist()
# plot histogram
plt.show()
pd.Series(reviews_len).describe()
# create a sequence of words
reviews = df['Summary'].values
all_text = ' '.join([c for c in reviews])
reviews_split = all_text.split('\n')

all_text2 = ' '.join(reviews_split)
print('Number of reviews :', len(all_text2))
# create a list of words
words = all_text2.split()  # Count all the words using Counter Method
# Build a dictionary that maps words to integers
count_words = Counter(words)

total_words = len(words)
sorted_words = count_words.most_common(9000)
print(total_words)
unique = []
for word in words:
    if word not in unique:
        unique.append(word)

# sort
unique.sort()

# print unique words
print(len(unique))
rootWord = []
for word in unique:
    rootWord.append(ps.stem(word))

print(len(rootWord))


# limit the word count and set the stopwords
wordcount = 500
stopwords = set(STOPWORDS)
stopwords.add("br")

# setup, generate and save the word cloud image to a file
wc = WordCloud(scale=5,
               background_color="white",
               max_words=wordcount,
               stopwords=stopwords)
wc.generate(all_text2)
wc.to_file("WordCloud.png")

# show the wordcloud as output
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.figure()
plt.axis("off")
plt.show()

max_seq_length = 30
# Tokenize sentences, keep 10000 most frequent words
tokenizer = Tokenizer(num_words=10000)
# Create the vocabulary index based on word frequency
tokenizer.fit_on_texts(X)
# Get the number of the unique words based on the number of elements in this dictionary
vocab_size = len(tokenizer.word_index) + 1
print('Found %s unique tokens.' % vocab_size)

# Assign an integer to each word and create integer sequences
x_train_tokens = tokenizer.texts_to_sequences(X_train)
x_test_tokens = tokenizer.texts_to_sequences(X_test)

# pad all our reviews to a specific length
X_train = pad_sequences(x_train_tokens, maxlen=max_seq_length)
X_test = pad_sequences(x_test_tokens, maxlen=max_seq_length)

print(len(X_test))
print(X_train.shape)
print(X_test.shape)

MAX_SEQUENCE_LENGTH = 30
EMBEDDING_DIM = 100

# calculate recall
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

# calculate precision
def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

# calculate f1-score
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# build model
model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
model.add(SpatialDropout1D(0.5))
model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy', f1_m, precision_m, recall_m])
model.summary()

history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), verbose=True, batch_size=32)
loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, y_test, verbose=1)
# scores = model.evaluate(X_test, y_test, verbose=1)

# Print metrics
print("F1-score")
print(f1_score)
print("Precision")
print(precision)
print("Recall")
print(recall)

# create loss and accuracy graphs
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('foo.png')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('foo1.png')

# Test our model
test_word = "This is a bad bad movie"
tw = tokenizer.texts_to_sequences([test_word])
tw = pad_sequences(tw, maxlen=30)
print(model.predict(tw))
prediction = int(model.predict(tw).round().item())
print(Y[prediction])

test_word = "This film is terrible"
tw1 = tokenizer.texts_to_sequences([test_word])
tw1 = pad_sequences(tw1, maxlen=30)
print(model.predict(tw1))
prediction1 = int(model.predict(tw1).round().item())
print(Y[prediction1])

test_word = "This film is great"
tw2 = tokenizer.texts_to_sequences([test_word])
tw2 = pad_sequences(tw2, maxlen=30)
print(model.predict(tw2))
prediction2 = int(model.predict(tw2).round().item())
print(Y[prediction2])

test_word = "This film is awesome"
tw3 = tokenizer.texts_to_sequences([test_word])
tw3 = pad_sequences(tw3, maxlen=30)
print(model.predict(tw3))
prediction3 = int(model.predict(tw3).round().item())
print(Y[prediction3])