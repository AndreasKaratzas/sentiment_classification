
import pandas
import re


def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text


def main():
    df = pandas.read_csv("C:/Users/andreas/Downloads/IMDB_Dataset.csv")
    print(df.head())
    # preprocess data
    df['review'] = df['review'].apply(preprocessor)
    print(df.head())
    df.to_csv('../dataset/IMDB.csv', index=False)


if __name__ == "__main__":
    main()
