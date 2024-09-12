import json
import numpy as np
from constants import *

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def SplitSentences(trainingSize, di) -> list | list:
    return di.sentences[0:trainingSize], di.sentences[trainingSize:]


def SplitLabels(trainingSize, di) -> list | list:
    return di.labels[0:trainingSize], di.labels[trainingSize:]


class DataInstance:
    def __init__(self) -> None:
        self.sentences = []
        self.labels = []
        self.urls = []
        pass


class Data:
    def __init__(self) -> None:
        self.trainingSentences = []
        self.trainingLabels = []
        self.testingSentences = []
        self.testingLabels = []

        self.trainingSeqs = []
        self.trainingSeqsPadded = []
        self.testingSeqs = []
        self.testingSeqsPadded = []

        self.vocabSize = 0
        
        pass


    def PrepareData(self, di):
        # Prepare the training and testing sentences
        self.trainingSentences, self.testingSentences = SplitSentences(TRAINING_SIZE, di)
        self.trainingLabels,    self.testingLabels    = SplitLabels(TRAINING_SIZE, di)

        # Prepare the training and testing labels
        self.trainingLabels = np.array(self.trainingLabels)
        self.testingLabels  = np.array(self.testingLabels)

        # Tokenize it
        tokenizer = Tokenizer(oov_token=OOV_TOK)
        tokenizer.fit_on_texts(self.trainingSentences)
        self.vocabSize = len(tokenizer.word_index) + 1

        # Training data
        self.trainingSeqs = tokenizer.texts_to_sequences(self.trainingSentences)
        self.trainingSeqsPadded = np.array(pad_sequences(self.trainingSeqs, padding='post', maxlen=MAX_LENGTH, truncating='post'))

        # Testing data
        self.testingSeqs = tokenizer.texts_to_sequences(self.testingSentences)
        self.testingSeqsPadded = np.array(pad_sequences(self.testingSeqs, padding='post', maxlen=MAX_LENGTH, truncating='post'))

        return


def LoadData(filePath, instance) -> None:
    # Load the json data
    with open(filePath, 'r') as file:
        datastore = json.load(file)

    # Prepare lists
    for item in datastore:
        instance.sentences.append(item['headline'])
        instance.labels.append(item['is_sarcastic'])
        instance.urls.append(item['article_link'])
    
    return