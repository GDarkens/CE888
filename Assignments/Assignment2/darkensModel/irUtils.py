from urllib.request import urlopen

import nltk
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk import pos_tag_sents, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer, word_tokenize


def fullyProcess(tweetFrame):
    tweetFrame['processed_tweet'] = tweetFrame['tweet'].str.lower()
    tweetFrame['processed_tweet'] = handleRemover(tweetFrame)
    tweetFrame['processed_tweet'] = punctuationRemover(tweetFrame)
    tweetFrame['processed_tweet'] = stopwordRemoval(tweetFrame)
    tweetFrame['processed_tweet'] = stemmer(tweetFrame)
    tweetFrame['processed_tweet'] = tweetFrame['processed_tweet'].str.strip()
    tweetFrame['processed_tweet'] = tokenizeSequencePadder((tweetFrame))
    
    
    #tweetFrame['length'] = tweetFrame['processed_tweet'].apply(lambda x: len(x))
    #tweetFrame['words'] = tweetFrame['processed_tweet'].apply(lambda x: len(x.split()))

    return tweetFrame


def stopwordRemoval(tweetFrame):
    en_Stopwords = stopwords.words("english")
    result = tweetFrame['processed_tweet'].apply(lambda x: ' '.join(
        [item for item in x.split() if item not in en_Stopwords]))

    return result


# Tokenizer set to drop non-alphanumeric values and leave rest
puncTokenizer = RegexpTokenizer(r'\w+')
def punctuationRemover(tweetFrame):
    result = tweetFrame['processed_tweet'].str.replace(r'[^\w\s]+', '')

    return result


def partOfSpeechTagger(tweetFrame):
    result = pos_tag_sents(tweetFrame['tweet'].apply(word_tokenize).tolist())
    return result


def tokenizeSequencePadder(tweetFrame):
    # Each tweet tokenized, then turned into numeric sequence, then padded
    # or truncated to a set length that all tweets will maintain
    nltkTokenizer = Tokenizer(num_words = 15000)
    nltkTokenizer.fit_on_texts(tweetFrame['processed_tweet'])
    
    tokenizedFrame = nltkTokenizer.texts_to_sequences(tweetFrame['processed_tweet'])
    paddedFrame = pad_sequences(tokenizedFrame, padding = 'post', maxlen = 60)
    
    return paddedFrame


def tokenizer(tweetFrame):
    result = tweetFrame['tweet'].apply(nltk.word_tokenize)
    return result


def stemmer(tweetFrame):
    #Stemmer applies the stem_sentences function to all tweets
    result = tweetFrame['processed_tweet'].apply(stem_sentences)
    return result


def stem_sentences(tweet):
    #Applied by stemmer
    stemmer = SnowballStemmer('english')
    tokenized = tweet.split()
    stemmed = [stemmer.stem(token) for token in tokenized]
    return ' '.join(stemmed)


def handleRemover(tweetFrame):
    #Simple replacement of handles that
    # TweetEval team left after pre-processing
    result = tweetFrame['processed_tweet'].str.replace('@user', '')
    return result
