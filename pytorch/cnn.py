
# TODO compile `requirements` file
# TODO split into `def`s

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchtext import data

import time
import spacy
import numpy
import pandas
import random
import warnings
import matplotlib.pyplot as plt


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.embedding.requires_grad = False
        self.convs_1d = nn.ModuleList([
            nn.Conv2d(1, n_filters, (k, embedding_dim), padding=(k - 2, 0))
            for k in filter_sizes])
        # 3. fully-connected layer(s) for classification
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        # 4. dropout and sigmoid layers
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def conv_and_pool(x, conv):
        """
        Convolutional + max pooling layer
        """
        # squeeze last dim to get size: (batch_size, num_filters, conv_seq_length)
        # conv_seq_length will be ~ 40
        x = F.relu(conv(x)).squeeze(3)
        # 1D pool over conv_seq_length
        # squeeze to get size: (batch_size, num_filters)
        x_max = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x_max

    def forward(self, x):
        """
        Defines how a batch of inputs, x, passes through the model layers.
        Returns a single, sigmoid-activated class score as output.
        """
        # embedded vectors
        embeds = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        # embeds.unsqueeze(1) creates a channel dimension that conv layers expect
        embeds = embeds.unsqueeze(1)
        # get output of each conv-pool layer
        conv_results = [self.conv_and_pool(embeds, conv) for conv in self.convs_1d]
        # concatenate results and add dropout
        x = torch.cat(conv_results, 1)
        x = self.dropout(x)
        # final logit
        logit = self.fc(x)
        return logit


# TODO add docstrings
def train_validate_test_split(df, train_percent=.8, validate_percent=.1, seed=42):
    numpy.random.seed(seed)
    perm = numpy.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    # columns_titles = ["Sentiment", "Summary"]
    # df = df.reindex(columns=columns_titles)
    train_df = df.iloc[perm[:train_end]]
    valid_df = df.iloc[perm[train_end:validate_end]]
    test_df = df.iloc[perm[validate_end:]]
    train_df.to_csv('train_df.csv', index=False)
    valid_df.to_csv('valid_df.csv', index=False)
    test_df.to_csv('test_df.csv', index=False)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


def get_max_length(df):
    max_len = 0
    for text in df.Summary:
        if len(text.split()) > max_len:
            max_len = len(text.split())
    return max_len + 1


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    # TODO add tqdm iteration
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.Summary).squeeze(1)
        loss = criterion(predictions, batch.Sentiment[0] - 2)
        acc = binary_accuracy(predictions, batch.Sentiment[0] - 2)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        # TODO add tqdm iteration
        for batch in iterator:
            predictions = model(batch.Summary).squeeze(1)
            loss = criterion(predictions, batch.Sentiment[0] - 2)
            acc = binary_accuracy(predictions, batch.Sentiment[0] - 2)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def predict_sentiment(model, sentence, TEXT, device, min_len=5):
    nlp = spacy.load('en')
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()


def plot_loss_and_accuracy(train_losses, val_losses):
    plt.plot(train_losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")
    plt.legend()
    plt.title("Losses")
    plt.show()
    plt.savefig("model-train_valid_losses.png", dpi=300, bbox_inches='tight', pad_inches=0.1)


def filter_prediction(prediction, critic):
    message = None
    if prediction > 0.5:
        message = "negative"
    else:
        message = "positive"
        prediction = 1 - prediction
    print('Prediction for critic {:25s}: {:7s}'.format('\"'+critic+'\"', message))


def main():
    SEED = 42
    spacy.load('en')
    warnings.filterwarnings("ignore")
    random.seed(SEED)
    numpy.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.deterministic = True
    BATCH_SIZE = 32
    train_validate_test_split(pandas.read_csv("../dataset/MoviesDataset.csv"))
    # Fields
    TEXT = data.Field(tokenize='spacy', batch_first=True)
    LABEL = data.Field(dtype=torch.float)

    fields = [('Summary', TEXT), ('Sentiment', LABEL)]

    # TabularDataset
    train_data, valid_data, test_data = data.TabularDataset.splits(
        path='./',
        train='train_df.csv',
        validation='valid_df.csv',
        test='test_df.csv',
        format='csv',
        fields=fields,
        skip_header=True
    )

    TEXT.build_vocab(train_data, valid_data, test_data,
                     max_size=10000,
                     vectors="glove.6B.100d",
                     unk_init=torch.Tensor.normal_)
    LABEL.build_vocab(train_data)

    # Iterators
    train_iterator = data.BucketIterator(
        train_data,
        batch_size=BATCH_SIZE,
        device=device,
    )
    valid_iterator = data.BucketIterator(
        valid_data,
        batch_size=BATCH_SIZE,
        device=device,
    )
    test_iterator = data.BucketIterator(
        test_data,
        batch_size=BATCH_SIZE,
        device=device,
    )
    LABEL.build_vocab(train_data)
    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 100
    N_FILTERS = 64
    FILTER_SIZES = [3, 3, 3, 3]
    OUTPUT_DIM = 1
    DROPOUT = 0.3
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES,
                OUTPUT_DIM, DROPOUT, PAD_IDX)
    print(model)
    print(f'The model has {count_parameters(model):,} trainable parameters')
    pretrained_embeddings = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)
    # TODO refactor names
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    model = model.to(device)
    criterion = criterion.to(device)
    N_EPOCHS = 20
    best_valid_loss = float('inf')
    train_losses, val_losses = [], []
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        train_losses.append(train_acc)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        val_losses.append(valid_acc)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        # TODO improve early stopping condition
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'cnn-model.pt')
        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
    plot_loss_and_accuracy(train_losses, val_losses)
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    print(f'\nTest Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')
    # TODO load saved model and test
    # TODO make custom predict `def`
    critic = "This film is terrible"
    prediction = predict_sentiment(model, critic, TEXT, device)
    filter_prediction(prediction, critic)
    critic = "This film is great"
    prediction = predict_sentiment(model, critic, TEXT, device)
    filter_prediction(prediction, critic)
    critic = "I loved this film"
    prediction = predict_sentiment(model, critic, TEXT, device)
    filter_prediction(prediction, critic)


# TODO move code from `def main()` to thread main
if __name__ == "__main__":
    main()
