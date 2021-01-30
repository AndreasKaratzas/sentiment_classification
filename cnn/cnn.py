
"""CNN model for Sentiment Classification.

In this script, there is an implementation of a Convolutional Neural
Network for Sentiment Classification. The sentiments are binary. To
classify the data the model uses an Embedding Layer to convert words
to an arithmetic sequence.

Convolutions are sliding window functions applied to a matrix that
achieve specific results. The sliding window is called a kernel, filter,
or feature detector. By representing each word with a vector of numbers
of a specific length and stacking a bunch of words on top of each other,
we get an image.

See Also
--------
`<https://torchtext.readthedocs.io/en/latest/index.html>`_

References
----------
The Deep Learning Framework used for the development of the current module is Pytorch [1]_.

.. [1] PyTorch: An Imperative Style, High-Performance Deep Learning Library by Paszke, Adam and Gross,
    Sam and Massa, Francisco and Lerer, Adam and Bradbury, James and Chanan, Gregory and Killeen, Trevor and Lin,
    Zeming and Gimelshein, Natalia and Antiga, Luca and Desmaison, Alban and Kopf, Andreas and Yang, Edward and DeVito,
    Zachary and Raison, Martin and Tejani, Alykhan and Chilamkurthy, Sasank and Steiner, Benoit and Fang, Lu and Bai,
    Junjie and Chintala, Soumith, published in "Advances in Neural Information Processing Systems 32",
    "Curran Associates, Inc.", "H. Wallach and H. Larochelle and A. Beygelzimer and F. Buc and E. Fox and R. Garnett",
    pp. 8024-8035, 2019.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchtext import data

from nltk import word_tokenize
from collections import Counter

import re
import os
import sys
import time
import tqdm
import math
import spacy
import numpy
import pandas
import random
import warnings
import matplotlib.pyplot as plt


class CNN(nn.Module):
    """
    Convolutional Neural Network model with Pretrained Embeddings.

    Attributes
    ----------
    vocab_size : int
        Size of the dictionary of embeddings.
    embedding_dim: int
        The size of each embedding vector.
    n_filters: int
        Number of channels produced by the convolution.
    filter_sizes: list
        A list that contains integers that correspond to the amount of channels produced by the convolution.
    output_dim: int
        The size of the output fully connected layer.
    dropout: float
        The probability of an element to be zeroed.
    pad_idx: int
        The numerical identifier mapped to the string token used as padding.

    Methods
    -------
    conv_and_pool(x, conv)
        Applies 1d convolution.
    forward(x)
        Defines the computation performed by the CNN model at every call.
    """

    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):
        """
        Parameters
        ----------
        vocab_size : int
            Size of the dictionary of embeddings.
        embedding_dim: int
            The size of each embedding vector.
        n_filters: int
            Number of channels produced by the convolution.
        filter_sizes: list
            A list that contains integers that correspond to the amount of channels produced by the convolution.
        output_dim: int
            The size of the output fully connected layer.
        dropout: float
            The probability of an element to be zeroed.
        pad_idx: int
            The numerical identifier mapped to the string token used as padding.
        """

        # extends the functionality of this method
        super(CNN, self).__init__()
        # defines an embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        # freezes the embedding layer
        self.embedding.requires_grad = False
        # applies convolution over the input signal
        self.convs_1d = nn.ModuleList([nn.Conv2d(1, n_filters, (k, embedding_dim), padding=(k - 2, 0))
                                       for k in filter_sizes])
        # applies linear transformation to the convolved data
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        # regularizes and prevents the co-adaptation of neurons
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def conv_and_pool(x, conv):
        """Applies 1d convolution.

        The method applies a 2D convolution over the input. It then filters the convolved output
        using Rectified Linear Unit. The result is a tensor of size [32 x 64 x Y x 100] where:
            * 32 is the batch size
            * 64 is the number of filters
            * Y is the sequence length which is equal to the sentence length
            * 100 is size of the second dimension of the kernel of a convolutional layer

        This temporary result is then squeezed to yields a tensor of size [32 x 64 x Y].
        In the last step, the method applies a 1D max pooling over the squeezed tensor with
        a sliding window of size equal to Y. The result is then squeezed again to produce
        a tensor of size [32 x 64].

        Parameters
        ----------
        x: torch.tensor
            This is a tensor of type float that operates as input for each convolution layer.
        conv: torch.nn.Module
            Applies a 2D convolution over a given input.

        Returns
        -------
        torch.tensor
            The 2D tensor to be used in the linear layer.
        """

        x = F.relu(conv(x)).squeeze(3)
        x_max = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x_max

    def forward(self, x):
        """Defines the computation performed by the CNN model at every call.

        This method *forwards* the given input to every single model layer.

        Parameters
        ----------
        x: torch.tensor
            This is a tensor of type int that operates as input for the defined model.

        Returns
        -------
        torch.tensor
            The tensor containing the predictions made by the model.
        """

        # embedded vectors of: (batch_size, seq_length, embedding_dim)
        embeds = self.embedding(x)
        # creates a fourth dimension for the convolutional module list
        embeds = embeds.unsqueeze(1)
        # gets output of each convolutional layer
        conv_results = [self.conv_and_pool(embeds, conv) for conv in self.convs_1d]
        # concatenates results
        x = torch.cat(conv_results, 1)
        # add dropout
        x = self.dropout(x)
        # fully connected layer that yields a float tensor of size equal to the batch size
        logit = self.fc(x)
        return logit


def nlp_preprocessor(text):
    """Defines an NLP preprocessor.

    This method takes some text and filters it. It deletes any
    non - alphanumeric character found. This is a standard
    preprocessing routine in machine learning models for NLP.
    It increases model's performance.

    Parameters
    ----------
    text: str
        This is the string to preprocess.

    Returns
    -------
    str
        The preprocessed - filtered string.
    """

    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?[)(DP]', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text


def dataset_preprocessor(df, column, filepath):
    """Preprocess text in given dataset.

    This method calls the predefined NLP preprocessor to filter out
    any non-alphanumeric character found in the given dataset. The
    method saves afterward the result using the given filepath. The
    filepath can also be relative. An example filepath is provided:

        `./filtered_dataset.csv`

    Parameters
    ----------
    df: pandas.DataFrame
        This is the given dataset.
    column: str
        This is the column with the user reviews
    filepath: str
        This is the filepath to save the preprocessed dataset
    """

    # apply the preprocessor to the dataframe
    df[column] = df[column].apply(nlp_preprocessor)
    # save data
    df.to_csv(filepath, index=False)


def train_validate_test_split(df, seed, train_percent=.7, validate_percent=.1):
    """Splits the given dataset.

    This method splits a dataframe into:
        * A dataframe used to train the model
        * A dataframe used to validate the model
        * A dataframe used to test the model

    The indexes of the given datasets are shuffled.

    Parameters
    ----------
    df: pandas.DataFrame
        This is the given dataset.
    seed: int
        This is the seed used for the NumPy shuffler.
    train_percent: float (optional)
        This is the dataset split ratio to get the sample data to fit the model.
    validate_percent: float (optional)
        This is the dataset split ratio to get the sample data to validate the model.
    """

    # shuffle the given dataframe indexes
    shuffled = numpy.random.RandomState(seed).permutation(df.index)
    # get the number of rows inside the dataframe
    data_length = len(df.index)
    # compute the number of rows for the training dataset
    train_end = int(train_percent * data_length)
    # make the training dataset size divide perfectly the batch size
    train_end = int(train_end/BATCH_SIZE) * BATCH_SIZE + BATCH_SIZE
    # compute the number of rows for the validation dataset
    validate_end = int(validate_percent * data_length) + train_end
    # make the validation dataset size divide perfectly the batch size
    validate_end = int(validate_end / BATCH_SIZE) * BATCH_SIZE + BATCH_SIZE
    # make the test dataset size divide perfectly the batch size
    test_end = int(data_length / BATCH_SIZE) * BATCH_SIZE
    # set the training dataset
    train_df = df.iloc[shuffled[:train_end]]
    # set the validation dataset
    valid_df = df.iloc[shuffled[train_end:validate_end]]
    # set the test dataset
    test_df = df.iloc[shuffled[validate_end:test_end]]
    # save the training dataset
    train_df.to_csv('train_df.csv', index=False)
    # save the validation dataset
    valid_df.to_csv('valid_df.csv', index=False)
    # save the test dataset
    test_df.to_csv('test_df.csv', index=False)


def inspect_vocab(df):
    """Estimates the Vocabulary size after subsampling.

    This method performs a virtual subsampling of the given dataset.
    This is done to increase the context window size of the embedding
    layer. If the computed probability is less than 50%, then the word
    is virtually discarded. The probability is given by the formula:

    .. math::

        p = 1 - \\sqrt{\\frac{t}{f}},

    Where:
        * :math:`p` is the probability of the token to be virtually discarded
        * :math:`t` is a chosen threshold typically around :math:`10^5`
        * :math:`f` the token frequency

    Parameters
    ----------
    df: pandas.DataFrame
        This is the given dataset.

    Returns
    -------
    int
        The vocabulary size after the virtual subsampling.
    int
        The vocabulary size without virtual subsampling.

    See Also
    --------
    `<https://www.mitpressjournals.org/doi/pdf/10.1162/tacl_a_00134>`_

    `<https://arxiv.org/abs/1301.3781>`_
    """

    # initialize vocabulary size register
    unique_count = 0
    # get the text column loaded in a pandas Series
    texts = df.Summary.str.lower()
    # get a dictionary with the count of each token in that pandas Series
    word_counts = Counter(word_tokenize('\n'.join(texts)))
    # get the total token sum
    total_token_count = sum(word_counts.values())
    # get the unique token sum
    final_count = len(word_counts)
    # initialize threshold constant
    threshold = 1e-5
    # use the subsampling formula to estimate the vocabulary size
    for token_freq in word_counts.values():
        if 1 - math.sqrt(threshold / (token_freq / total_token_count)) > 0.5:
            unique_count += 1
    return unique_count, final_count


def compute_vocab_size():
    """Returns vocabulary size

    This method computes vocabulary size. It uses the subsampling
    flag defined in the main thread. If subsampling is activated,
    then the method sets the vocabulary size equal to the subsampled
    estimation of the vocabulary size. Otherwise, it sets the
    vocabulary size equal to the total unique token count.
    """

    if subsampling:
        return vocab_subsampled
    else:
        return token_count


def count_parameters():
    """Counts model trainable parameters.

    Returns
    -------
    int
        The number of trainable parameters.
    """

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def binary_accuracy(preds, y):
    """Computes prediction accuracy.

    This method is used to estimate the models accuracy over binary
    targets. The prediction is rounded to compare it to the true label.

    Parameters
    ----------
    preds: torch.tensor
        These are the predictions returned by the model for an input batch.
    y: torch.tensor
        This is the ground truth tensor for the same input batch.

    Returns
    -------
    torch.tensor
        The model accuracy ratio for the given predictions. The tensor is a single float element container.
    """

    # use the sigmoid to round the predictions
    rounded_preds = torch.round(torch.sigmoid(preds))
    # count the correct predictions by comparing them to the ground truth tensor
    correct = (rounded_preds == y).float()
    # compute the accuracy of the model
    acc = correct.sum() / len(correct)
    # return the accuracy
    return acc


def get_max_length(df):
    """Computes maximum number of tokens given a dataframe.

    This method is used to compute the maximum length found at the text
    column of the given dataframe. The column of the dataframe with the
    text is *Summary*. This function is useful if one decides to use
    padding for tokenization.

    Parameters
    ----------
    df: pandas.DataFrame
        This is the dataset of the model.

    Returns
    -------
    int
        The maximum number of tokens found in a dataframe column.
    """

    # initializes maximum sentence length register
    max_len = 0
    # iterates the Summary column of the given dataframe
    for text in df.Summary:
        # checks length of the "running" sentence
        if len(text.split()) > max_len:
            # update maximum sentence length register
            max_len = len(text.split())
    return max_len


def train(iterator):
    """Fits a model.

    This method is used to fit the defined CNN model. The method provides
    progress context for the user using progressbar.

    Parameters
    ----------
    iterator: torchtext.data.Iterator
        An iterator to load batches of training data from the given dataset.

    Returns
    -------
    float
        The epoch training loss
    float
        The epoch training accuracy
    """

    # initializes epoch loss accumulator
    epoch_loss = 0
    # initializes epoch accuracy accumulator
    epoch_acc = 0
    # sets the module in training mode
    model.train()
    for batch in tqdm.tqdm(iterator):
        # set the gradients to zero
        optimizer.zero_grad()
        # make predictions
        predictions = model(batch.Summary).squeeze(1)
        # compute loss
        loss = criterion(predictions, batch.Sentiment.squeeze(0))
        # compute accuracy
        acc = binary_accuracy(predictions, batch.Sentiment.squeeze(0))
        # store the gradients
        loss.backward()
        # parameter update based on the current gradients
        optimizer.step()
        # update epoch loss accumulator
        epoch_loss += loss.item()
        # update epoch accuracy accumulator
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(iterator):
    """Evaluates a model.

    This method is called either to validate the defined CNN model
    or to test it, by disabling gradient calculation. The method
    provides progress context for the user using progressbar.

    Parameters
    ----------
    iterator: torchtext.data.Iterator
        An iterator to load batches of evaluation data from the given dataset.

    Returns
    -------
    float
        The epoch evaluation loss
    float
        The epoch evaluation accuracy
    """

    # initializes epoch loss accumulator
    epoch_loss = 0
    # initializes epoch accuracy accumulator
    epoch_acc = 0
    # sets the module in evaluation mode
    model.eval()
    # disables gradient calculation
    with torch.no_grad():
        for batch in tqdm.tqdm(iterator):
            # make predictions
            predictions = model(batch.Summary).squeeze(1)
            # compute loss
            loss = criterion(predictions, batch.Sentiment.squeeze(0))
            # compute accuracy
            acc = binary_accuracy(predictions, batch.Sentiment.squeeze(0))
            # update epoch loss accumulator
            epoch_loss += loss.item()
            # update epoch accuracy accumulator
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time():
    """Computes epoch duration.

    This method is called upon the launch of each epoch,
    and upon the termination of each epoch. It then uses
    the checkpoints created to compute the epoch's duration.

    Returns
    -------
    int
        Number of minutes rounded down that represent the running epoch's duration
    int
        The remaining of seconds that represent the running epoch's duration
    """

    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def plot_loss_and_accuracy():
    """Plots model's fitting results.

    This method takes the lists containing the training and the
    validation losses and plots them together. This method is
    useful when detecting an over-fitted model (or an under-fitted).
    The method saves the plot at the project directory.
    """

    plt.plot(train_losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")
    plt.legend()
    plt.title("Losses")
    plt.savefig("model-train_valid_losses.png", dpi=300, bbox_inches='tight', pad_inches=0.1)


def predict_sentiment(sentence, min_len=5):
    """Classifies a custom critic.

    This method converts a sentence into arithmetic tokens.
    The tokens are then given to a trained model. The model
    predicts the sentiment of that *critic*.

    Parameters
    ----------
    sentence: str
        The custom critic to be classified.
    min_len: int (optional)
        The minimum length of tokens of the given sentence.


    Returns
    -------
    float
        The probability of the critic being negative.
    """

    # load natural language processor
    nlp = spacy.load('en_core_web_sm')
    # set the module in evaluation mode
    model.eval()
    # tokenize given text using the defined processor
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    # pad the sentence if it has less tokens than required
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))
    # convert tokens to embeddings using the fit torchtext data field
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    # convert embedding list to torch tensor and load it to the available device
    tensor = torch.LongTensor(indexed).to(device)
    # unsqueeze tensor to make it 2D
    tensor = tensor.unsqueeze(0)
    # filter prediction using sigmoid
    prediction = F.sigmoid(model(tensor))
    return prediction.item()


def filter_prediction(prediction, critic):
    """Provides feedback over a custom prediction.

    This method takes the prediction of the model on a custom critic and
    defines if it was positive or negative. Finally, it prints the proper
    message.

    Parameters
    ----------
    prediction: float
        The probability of a critic being negative
    critic: str
        The word sequence used as a critic
    """

    message = "negative"
    if prediction < 0.5:
        message = "positive"
        prediction = 1 - prediction
    print('Label for critic {:25s}: {:7s}\t-\tPrediction validity probability: {:10f}'.format(
        '\"'+critic+'\"', message, prediction))


def manual_testing():
    """Calls model upon custom critics.

    In this method there are some movie critics
    defined to test the model with custom data.
    """

    x_critic = "This film is terrible"
    y_pred = predict_sentiment(x_critic)
    filter_prediction(y_pred, x_critic)
    x_critic = "This film is great"
    y_pred = predict_sentiment(x_critic)
    filter_prediction(y_pred, x_critic)
    x_critic = "I loved this film"
    y_pred = predict_sentiment(x_critic)
    filter_prediction(y_pred, x_critic)


if __name__ == "__main__":
    # define a seed for the randomizers
    SEED = 42
    # load English package of spacy package
    spacy.load('en_core_web_sm')
    # disable warnings
    warnings.filterwarnings("ignore")
    # seed random package
    random.seed(SEED)
    # seed numpy
    numpy.random.seed(SEED)
    # seed pytorch
    torch.manual_seed(SEED)
    # check for any CUDA device available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # make program controllability easier
    torch.backends.cudnn.deterministic = True
    # define batch size
    BATCH_SIZE = 32
    # define filepath to dave model
    model_filepath = os.getcwd() + os.sep + 'cnn-model.pt'
    # define the input's filepath
    dataset_filepath = '..' + os.sep + 'dataset' + os.sep + 'MoviesDataset.csv'
    # load the dataset
    dataset = pandas.read_csv(dataset_filepath)
    # dataset preprocessed
    dataset_preprocessor(dataset, 'Summary',
                         '..' + os.sep + 'dataset' + os.sep + 'MoviesDatasetPreprocessed.csv')
    # reload dataset after preprocessing
    dataset = pandas.read_csv('..' + os.sep + 'dataset' + os.sep + 'MoviesDatasetPreprocessed.csv')
    # inspect vocabulary
    vocab_subsampled, token_count = inspect_vocab(dataset)
    # set subsampling flag
    subsampling = True
    # set vocabulary size
    vocab_size = compute_vocab_size()
    # split the dataset
    train_validate_test_split(dataset, SEED)
    # define torchtext data text field
    TEXT = data.Field(tokenize='spacy', batch_first=True)
    # define torchtext data label field
    LABEL = data.Field(dtype=torch.float, unk_token=None, pad_token=None)
    # associate defined fields with DataFrame columns
    fields = [('Summary', TEXT), ('Sentiment', LABEL)]
    # define a dataset of columns stored in CSV
    train_data, valid_data, test_data = data.TabularDataset.splits(
        path='./',
        train='train_df.csv',
        validation='valid_df.csv',
        test='test_df.csv',
        format='csv',
        fields=fields,
        skip_header=True
    )
    # construct the Vocab object for the TEXT field
    TEXT.build_vocab(train_data, valid_data, test_data,
                     max_size=vocab_size,
                     vectors="glove.6B.100d",
                     unk_init=torch.Tensor.normal_)
    # construct the Vocab object for the LABEL field
    LABEL.build_vocab(train_data)
    # define an iterator that batches the training dataset object
    train_iterator = data.BucketIterator(
        train_data,
        batch_size=BATCH_SIZE,
        device=device,
    )
    # define an iterator that batches the validation dataset object
    valid_iterator = data.BucketIterator(
        valid_data,
        batch_size=BATCH_SIZE,
        device=device,
    )
    # define an iterator that batches the test dataset object
    test_iterator = data.BucketIterator(
        test_data,
        batch_size=BATCH_SIZE,
        device=device,
    )
    # define the size of the dictionary of embeddings
    INPUT_DIM = len(TEXT.vocab)
    # define the size of each embedding vector
    EMBEDDING_DIM = 100
    # define the number of channels produced by each convolution
    N_FILTERS = 64
    # define the size of the first dimension of the kernel of each convolutional layer
    FILTER_SIZES = [2, 3, 4, 5]
    # define the number of neurons in the output layer of the model
    OUTPUT_DIM = 1
    # define the probability of an element to be zeroed
    DROPOUT = 0.3
    # return the index of the string token used as padding
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    # return the index of the string token used to represent Out-Of-Vocabulary words
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    # define a CNN model
    model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)
    # print model summary
    print(model)
    # print model trainable parameters
    print(f'The model has {count_parameters():,} trainable parameters')
    # extract pretrained vectors
    pretrained_embeddings = TEXT.vocab.vectors
    # copy pretrained vectors to the embedding layer of the defined model
    model.embedding.weight.data.copy_(pretrained_embeddings)
    # set the weight of the <pad> token
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
    # set the weight of the <unk> token
    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # define cost function
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.ones([BATCH_SIZE]))
    # load model to the available device
    model = model.to(device)
    # load cost function to the available device
    criterion = criterion.to(device)
    # define number of epochs for the model's training
    N_EPOCHS = 20
    # initialize a register that holds the best validation cost returned during an epoch
    best_valid_loss = float('inf')
    # declare the train and validation loss lists
    train_losses, val_losses = [], []
    # fit the model
    for epoch in range(N_EPOCHS):
        # initialize an epoch starting time-point
        start_time = time.time()
        # train the model
        train_loss, train_acc = train(train_iterator)
        # update the train loss list
        train_losses.append(train_acc)
        # validate the model
        valid_loss, valid_acc = evaluate(valid_iterator)
        # update the validation loss list
        val_losses.append(valid_acc)
        # initialize an epoch ending time-point
        end_time = time.time()
        # compute epoch duration in minutes and seconds
        epoch_mins, epoch_secs = epoch_time()
        # save the model if validation loss was better than past validation losses
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_filepath)
        # print epoch's progress results
        print(f'\nEpoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%\n')
    # plot model's fitting data
    plot_loss_and_accuracy()
    # load the best evaluated model
    model.load_state_dict(torch.load(model_filepath, map_location=device))
    # test the model
    test_loss, test_acc = evaluate(test_iterator)
    # print test results
    print(f'\nTest Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')
    # test the model over custom critics
    manual_testing()
