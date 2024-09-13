import json
import numpy as np
from constants import *

# "type: ignore" here removes the annoying import error that only VSC shows (executes properly though)
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore

def SplitSentences(trainingSize, rawData) -> list | list:
    return rawData.sentences[0:trainingSize], rawData.sentences[trainingSize:]


def SplitLabels(trainingSize, rawData) -> list | list:
    return rawData.labels[0:trainingSize], rawData.labels[trainingSize:]


class RawData:
    def __init__(self) -> None:
        self.sentences = []
        self.labels = []
        self.urls = []
        pass


class ProcessedData:
    def __init__(self) -> None:
        self.trainingSentences = []
        self.trainingLabels    = []
        self.testingSentences  = []
        self.testingLabels     = []

        self.trainingSeqs       = []
        self.trainingSeqsPadded = []
        self.testingSeqs        = []
        self.testingSeqsPadded  = []

        self.vocabSize = 0
        
        pass


    def PrepareData(self, rawData):
        # Prepare the training and testing sentences
        self.trainingSentences, self.testingSentences = SplitSentences(TRAINING_SIZE, rawData)
        self.trainingLabels,    self.testingLabels    = SplitLabels(TRAINING_SIZE, rawData)

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

        return tokenizer


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