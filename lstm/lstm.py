""" Sentiment Classification on Movie Reviews using LSTMs and Keras.

In this script, there is an implementation of a Long Short Term Memory(LSTM)
which is a Recurrent Neural Network(RNN) to perform binary sentiment classification
on movie reviews.

See Also
--------
`<https://keras.io/api/>`_

References
----------
The Deep Learning Framework used for the development of the current module is Keras [1]_.
.. [1] Keras: is a deep learning API written in Python, running on top of the machine learning platform
TensorFlow. It was developed with a focus on enabling fast experimentation. Being able to go from idea
to result as fast as possible is key to doing good research. is the high-level API of TensorFlow 2:
an approachable, highly-productive interface for solving machine learning problems, with a focus on modern
deep learning. It provides essential abstractions and building blocks for developing and shipping machine
learning solutions with high iteration velocity.
"""

import sys
import os
import warnings
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Dropout
from tensorflow.python.keras.layers import SpatialDropout1D
from collections import Counter
from keras import backend as K


def plot_sentiment_histogram(sentiment):
    """Plots count of positive and negative reviews histogram.

     This method is used to plot a histogram which shows
     the number of positive and negative reviews in the dataset.

     Parameters
     ----------
     sentiment: numpy.ndarray
              Review column from the dataset.

    """
    unique_vals = df['Sentiment'].sort_values().unique()
    plt.xlabel("Values")
    plt.ylabel("Count")
    plt.hist(sentiment.values, align='left')
    plt.show()
    print(df['Sentiment'].value_counts())
    print(sentiment.values)


def plot_review_length_histogram(X):
    """Plots reviews length histogram.

     This method is used to plot a histogram which shows
     the total number of reviews that have a specific length.

     Parameters
     ----------
     X: pandas.core.series.Series
              Sentiment column from the dataset.

    """
    # get the length of every review in the dataset
    reviews_len = [len(x.split()) for x in X]
    pd.Series(reviews_len).describe()
    plt.xlabel("Values")
    plt.ylabel("Count")
    # create review length histogram
    pd.Series(reviews_len).hist()
    # plot histogram
    plt.show()


def dataset_preprocessing():
    """Dataset preprocessing.

     This method is used to preprocess the reviews.
     More specifically:

     * It concatenates all reviews in one string.
     * It checks if our reviews need stemming
     * It gets number of unique words
     * It calculates the frequency of each word

     Returns
     ----------
     str
        All reviews concatenated in one string.
    """
    # create a sequence of words
    reviews = df['Summary'].values
    all_text = ' '.join([c for c in reviews])
    reviews_split = all_text.split('\n')
    # put all reviews in one string
    all_text2 = ' '.join(reviews_split)
    print('Number of words :', len(all_text2))
    # create a list of words
    words = all_text2.split()
    # build a dictionary that maps words to integers based on frequency
    count_words = Counter(words)
    # count all the words using Counter Method
    total_words = len(words)
    sorted_words = count_words.most_common(9000)
    print(total_words)
    # get unique words
    unique = []
    for word in words:
        if word not in unique:
            unique.append(word)

    # sort
    unique.sort()

    # print unique words
    print(len(unique))
    # check for stemming
    rootWord = []
    for word in unique:
        rootWord.append(ps.stem(word))

    print(len(rootWord))
    return all_text2


def wordcloud_illustration(texts):
    """Plots Wordcloud illustration.

     This method is used to plot a Wordcloud illustration,
     a technique for visualising frequent words in a text
     where the size of the words represents their frequency.

     Parameters
     ----------
     texts: str
              All reviews concatenated in one string.

    """
    # limit the word count
    wordcount = 500
    # set the stopwords
    stopwords = set(STOPWORDS)
    stopwords.add("br")

    # setup, generate and save the word cloud image to a file
    wc = WordCloud(scale=5,
                   background_color="white",
                   max_words=wordcount,
                   stopwords=stopwords)
    wc.generate(texts)
    wc.to_file("WordCloud.png")

    # show the wordcloud as output
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.figure()
    plt.axis("off")
    plt.show()


def tokenize_pad(X, X_train, X_test):
    """Creates and pads sequences.

       This method is used to transform words into sequences of integers.
       First, Keras Tokenizer is used to tokenize sentences to words, keeping only most frequent words.
       Then, we transform each word to an integer (based on frequency) and we create a vocabulary.
       After that, we create integer sequences that we pad to a specific length.

       Parameters
       ----------
       X: numpy.ndarray
               Review column from the dataset.
       X_train: numpy.ndarray
              Used to fit the machine learning model (input).
       X_test: numpy.ndarray
             Used to evaluate the fit machine learning model(input).


       Returns
       -------
       numpy.ndarray
                    Used to fit the machine learning model (input).
       numpy.ndarray
                    Used to evaluate the fit machine learning model(input).
       Tokenizer
             Allows us to vectorize a text corpus, by turning each text into a sequence of integers
       int
          Size of the vocabulary
    """
    max_seq_length = 30
    # tokenize sentences, keep 10000 most frequent words
    tokenizer = Tokenizer(num_words=10000)
    # create the vocabulary index based on word frequency
    tokenizer.fit_on_texts(X)
    # get the number of the unique words based on the number of elements in this dictionary
    vocab_size = len(tokenizer.word_index) + 1
    print('Found %s unique tokens.' % vocab_size)

    # assign an integer to each word and create integer sequences
    x_train_tokens = tokenizer.texts_to_sequences(X_train)
    x_test_tokens = tokenizer.texts_to_sequences(X_test)

    # pad all our reviews to a specific length
    X_train = pad_sequences(x_train_tokens, maxlen=max_seq_length)
    X_test = pad_sequences(x_test_tokens, maxlen=max_seq_length)

    print(len(X_test))
    print(X_train.shape)
    print(X_test.shape)
    return X_train, X_test, tokenizer, vocab_size


def recall_m(y_true, y_pred):
    """Calculates recall metric.

     This method is used to implement a custom recall metric.

     Parameters
     ----------
     y_true: keras_tensor
           Is the true data (or target, ground truth) we pass to the fit method
           (conversion of the numpy array y_train into a tensor).
     y_pred: keras_tensor
           Is the data predicted (calculated, output) by our model.
           (conversion of the numpy array y_train into a tensor).

     Returns
     -------
     float
           Recall metric.

    """
    # calculate true positives
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    # get total actual positives
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    # calculate recall
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    """Calculates precision metric.

    This method is used to implement a custom precision metric.

    Parameters
    ----------
    y_true: keras_tensor
          Is the true data (or target, ground truth) we pass to the fit method
          (conversion of the numpy array y_train into a tensor).
    y_pred: keras_tensor
          Is the data predicted (calculated, output) by our model.

    Returns
    -------
    float
          Precision metric.

    """
    # calculate true positives
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    # get predicted positives (true positives and false positives)
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    # calculate precision
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    """Calculates f1 metric.

     This method is used to implement a custom f1 metric.

     Parameters
     ----------
     y_true: keras_tensor
           Is the true data (or target, ground truth) we pass to the fit method
           (conversion of the numpy array y_train into a tensor).
     y_pred: keras_tensor
           Is the data predicted (calculated, output) by our model.

     Returns
     -------
     float
           F1-score metric.

     """
    # get precision
    precision = precision_m(y_true, y_pred)
    # get recall
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def build_model(vocab_size):
    """Defines and compiles a model.

    This method defines and then compiles a Sequential Keras Model.
    Our Sequential model is a linear stack of these layers:

    * Embedding Layer: a dictionary mapping integer indices to dense vectors, it takes as input a 2D tensor of integers, of shape (samples, sequence_length), where each entry is a sequence of integers.

    * SpatialDropout1D: same function as Dropout, however, it drops entire 1D feature maps instead of individual elements.

    * LSTM

    * Dropout: randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting. Inputs not set to 0 are scaled up by 1/(1 - rate) such that the sum over all inputs is unchanged.

    * Dense: implements the operation: output = activation(dot(input, kernel) + bias), where activation is the element-wise activation function passed as the activation argument, kernel is a weights matrix created by the layer, and bias is a bias vector created by the layer (optional - not used here). We choose Sigmoid as activation function because the output is binary. The optimizer is Adam and the loss function is Binary Crossentropy because the output is binary.

   Parameters
   ----------
   vocab_size: int
         Size of the vocabulary

   Returns
   -------
   Sequential
       The compiled model
    """
    # initialize parameters for Embedding Layer
    MAX_SEQUENCE_LENGTH = 30
    EMBEDDING_DIM = 100

    model = Sequential()
    # add Embedding layer
    model.add(Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
    # add Spatial dropout
    model.add(SpatialDropout1D(0.5))
    # add LSTM layer
    model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
    # add dropout
    model.add(Dropout(0.5))
    # add Dense layer with sigmoid activation function
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy', f1_m, precision_m, recall_m])
    model.summary()

    return model


def train_model(model, X_train, y_train, X_test, y_test):
    """Fits a model.
       This method is used to train the defined LSTM model.

       Parameters
       ----------
       model: Sequential
             The compiled Sequential model.
       X_train: numpy.ndarray
              Used to fit the machine learning model (input).
       y_train: numpy.ndarray
              Used to fit the machine learning model (output).
       X_test: numpy.ndarray
             Used to evaluate the fit machine learning model(input).
       y_test: numpy.ndarray
             Used to evaluate the fit machine learning model(output).

       Returns
       -------
       tensorflow.keras.callbacks.History()
                 A record of training loss values and metrics values at successive epochs,
                 as well as validation loss values and validation metrics values
       Sequential
                The trained model
       """    
    history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), verbose=True, batch_size=32)                        
    loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, y_test, verbose=1)
    # scores = model.evaluate(X_test, y_test, verbose=1)

    # print metrics
    print("F1-score")
    print(f1_score)
    print("Precision")
    print(precision)
    print("Recall")
    print(recall)
    return history, model


def plot_graphs(history):
    """Plots loss and accuracy graphs.

       This method plots loss and accuracy graphs of our model.

       Parameters
       ----------
       history: tensorflow.keras.callbacks.History()
                A record of training loss values and metrics values at successive epochs,
                as well as validation loss values and validation metrics values
    """
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])

    # create accuracy graph
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # save to file
    plt.savefig('foo.png')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    # create loss graph
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # save to file
    plt.savefig('foo1.png')
    plt.show()


def test_model(model, tokenizer):
    """Tests our model.

        This method tests our model with 4 sample reviews to see how it predicts sentiment.
        First, it transforms a sentence into a sequence of integers.
        Then the sequence is given to the trained model, which
        predicts the sentiment of that review.

        Parameters
        ----------
        model: Sequential
            The trained Sequential Model.

        tokenizer: Tokenizer
            Allows us to vectorize a text corpus, by turning each text into a sequence of integers

    """
    # sample review 1
    test_word = "This is a bad bad movie"
    # transform text to a sequence of integers
    tw = tokenizer.texts_to_sequences([test_word])
    # pad review to a specific length
    tw = pad_sequences(tw, maxlen=30)
    print(model.predict(tw))
    # get prediction (0 or 1)
    prediction = 1 if model.predict(tw).item() > 0.5 else 0
    print(prediction)

    # sample review 2
    test_word = "This film is terrible"
    # transform text to a sequence of integers
    tw1 = tokenizer.texts_to_sequences([test_word])
    # pad review to a specific length
    tw1 = pad_sequences(tw1, maxlen=30)
    print(model.predict(tw1))
    # get prediction (0 or 1)
    prediction1 = 1 if model.predict(tw1).item() > 0.5 else 0
    print(prediction1)

    # sample review 3
    test_word = "This film is great"
    # transform text to a sequence of integers
    tw2 = tokenizer.texts_to_sequences([test_word])
    # pad review to a specific length
    tw2 = pad_sequences(tw2, maxlen=30)
    print(model.predict(tw2))
    # get prediction (0 or 1)
    prediction2 = 1 if model.predict(tw2).item() > 0.5 else 0
    print(prediction2)

    # sample review 4
    test_word = "This film is awesome"
    # transform text to a sequence of integers
    tw3 = tokenizer.texts_to_sequences([test_word])
    # pad review to a specific length
    tw3 = pad_sequences(tw3, maxlen=30)
    print(model.predict(tw3))
    # get prediction (0 or 1)
    prediction3 = 1 if model.predict(tw3).item() > 0.5 else 0
    print(prediction3)
    

def main():
    """Driver for LSTM model.

    This function defines the order of execution and binds all other modules.
    """
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

    plot_review_length_histogram(X)
    all_text2 = dataset_preprocessing()
    wordcloud_illustration(all_text2)
    X_train, X_test, tokenizer, vocab_size = tokenize_pad(X, X_train, X_test)
    model = build_model(vocab_size)
    history, model = train_model(model, X_train, y_train, X_test, y_test)
    plot_graphs(history)
    test_model(model, tokenizer)


if __name__ == "__main__":
    # disable tensorflow deprecation warnings
    warnings.filterwarnings("ignore")
    # possible CUBLAS warnings/errors if running on GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # confirm device
    if tf.test.gpu_device_name():
        print('Running on GPU')
    else:
        print("Running on CPU")
    # load dataset
    df = pd.read_csv('..' + os.sep + 'dataset' + os.sep + 'MoviesDataset.csv')
    # print 10 first rows
    print(df.head(10))
    # initialize Stemmer
    ps = PorterStemmer()
    main()
