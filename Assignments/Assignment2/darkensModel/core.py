from pathlib import Path
from urllib.request import urlopen

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import irUtils

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')
# nltk.download('wordnet')


def importEmojiDatasets(dataset_name):
    # Download a given dataset from GitHub CDN
    # Data cast to global values

    baseUrl = "https://raw.githubusercontent.com/GDarkens/CE888/main/Assignments/Assignment1/tweeteval/datasets/"

    MAPPING_URL = baseUrl + dataset_name + "/" + "mapping.txt"
    global mapping
    mapping = urlopen(MAPPING_URL).read().decode('utf-8').split("\n")

    TEST_LABELS_URL = baseUrl + dataset_name + "/" + "test_labels.txt"
    global test_labels
    test_labels = urlopen(
        TEST_LABELS_URL).read().decode('utf-8').split("\n")

    TEST_TEXT_URL = baseUrl + dataset_name + "/" + "test_text.txt"
    global test_text
    test_text = urlopen(
        TEST_TEXT_URL).read().decode('utf-8').split("\n")

    TRAIN_LABELS_URL = baseUrl + dataset_name + "/" + "train_labels.txt"
    global train_labels
    train_labels = urlopen(
        TRAIN_LABELS_URL).read().decode('utf-8').split("\n")

    TRAIN_TEXT_URL = baseUrl + dataset_name + "/" + "train_text.txt"
    global train_text
    train_text = urlopen(
        TRAIN_TEXT_URL).read().decode('utf-8').split("\n")

    VAL_LABELS_URL = baseUrl + dataset_name + "/" + "val_labels.txt"
    global val_labels
    val_labels = urlopen(
        VAL_LABELS_URL).read().decode('utf-8').split("\n")

    VAL_TEXT_URL = baseUrl + dataset_name + "/" + "val_text.txt"
    global val_text
    val_text = urlopen(
        VAL_TEXT_URL).read().decode('utf-8').split("\n")


def dataFramer(tweetVar, labelVar):
    #Suitably frames tweets, pairing them with their given label
    tweet_array = []
    label_array = []
    for i in range(len(tweetVar)):
        tweet_array.append({"tweet": tweetVar[i], "label": labelVar[i]})
        label_array.append(i)
    dataframe = pd.DataFrame(tweet_array, index=label_array)
    dataframe = dataframe[:-1]  # Drop last row, as is a blank
    return dataframe


def main():
    dataset = input(
        "Choose a dataset: [1] = emoji   [2] = hate   [3] = sentiment \n")
    if(dataset == "1"):
        dataset = "emoji"
    elif(dataset == "2"):
        dataset = "hate"
    elif(dataset == "3"):
        dataset == "sentiment"
    else:
        print("Invalid input, enter 1, 2, or 3")
        return

    importEmojiDatasets(dataset)  # emoji  / hate /  sentiment

    trainingTweets = dataFramer(train_text, train_labels)
    testingTweets = dataFramer(test_text, test_labels)
    valTweets = dataFramer(val_text, val_labels)

    train_processedTweets = irUtils.fullyProcess(trainingTweets)
    test_processedTweets = irUtils.fullyProcess(testingTweets)
    #val_processedTweets = irUtils.fullyProcess(valTweets)

    print(trainingTweets.head())

    if(dataset == "hate"):
        hateModelTrainer(trainingTweets, testingTweets)
    elif(dataset == "emoji"):
        return
    elif(dataset == "sentiment"):
        return


def hateModelTrainer(trainingTweets, testingTweets):
    # Trainer for hate model
    # Feeds trained model into testing module
    
    train_y = trainingTweets['label']
    train_x = trainingTweets['processed_tweet']

    x_train, x_test, y_train, y_test = train_test_split(
        train_x, train_y, test_size = 0.33, random_state = 0, shuffle = True)
    
    classifier = Pipeline([("tf-idf", TfidfVectorizer()), ("classr", RandomForestClassifier(
        n_estimators=100, warm_start=True, verbose=1))])
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    print("\n- - Training Classification Report: - -")
    print(classification_report(y_test, y_pred))
    print("- - - - - - - - - - - - - - - - - - - - -")

    hateModelTester(testingTweets, y_pred, classifier)


def hateModelTester(testingTweets, model, classifier):
    # Tester for hate model
    model = classifier.predict(testingTweets['label'])

    print("\n- - Testing Classification Report: - -")
    print(classification_report(testingTweets['label'], model))


def sentimentModelTrainer(trainingTweets, testingTweets):
    # Trainer for hate model
    train_y = trainingTweets['label']
    train_x = trainingTweets['processed_tweet']

    x_train, x_test, y_train, y_test = train_test_split(
        train_x, train_y, test_size=0.33, random_state=0, shuffle=True)
    
    classifier = Pipeline([("tf-idf", TfidfVectorizer()), ("classr", RandomForestClassifier(
        n_estimators=100, warm_start=True, verbose=1,))])
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    print("\n- - Training Classification Report: - -")
    print(classification_report(y_test, y_pred))
    print("- - - - - - - - - - - - - - - - - - - - -")
    

if __name__ == "__main__":
    # Executes only if run as a script
    main()
