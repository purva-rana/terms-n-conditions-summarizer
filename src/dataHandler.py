import json
from numpy import array as nparray
from constants import *

# "type: ignore" here removes the annoying import error that only VSC shows (executes properly though)
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore


# Split the sentences into training and testing sentences
# I/P: Size for training data, and the raw data object
# O/P: Split sentences as list, list
def SplitSentences(trainingSize, rawData) -> list | list:
    return rawData.sentences[0:trainingSize], rawData.sentences[trainingSize:]

# Split the labels into training and testing labels
# I/P: Size for training data, and the raw data object
# O/P: Split labels as list, list
def SplitLabels(trainingSize, rawData) -> list | list:
    return rawData.labels[0:trainingSize], rawData.labels[trainingSize:]


# Object to hold the raw data loaded initially from the dataset
class RawData:
    def __init__(self) -> None:
        self.sentences = [] # Actual sentence
        self.labels = []    # is_sarcastic or not (boolean)
        # self.urls = []      # Link to article, not used in this program 
        pass


# Object to hold the processed data, ready to be used by the model
class ProcessedData:
    def __init__(self) -> None:
        # Sentences and labels
        self.trainingSentences = []
        self.trainingLabels    = []
        self.testingSentences  = []
        self.testingLabels     = []

        # Sequences
        self.trainingSeqs       = []
        self.testingSeqs        = []

        self.vocabSize = 0
        
        pass

    # Convert rawData into a ProcessedData object
    # I/P: RawData object
    # O/P: Tokenizer used on the training sentences
    def PrepareData(self, rawData):

        # Prepare the training and testing sentences
        self.trainingSentences, self.testingSentences = SplitSentences(TRAINING_SIZE, rawData)
        self.trainingLabels,    self.testingLabels    = SplitLabels(TRAINING_SIZE, rawData)

        # Prepare the training and testing labels
        self.trainingLabels = nparray(self.trainingLabels)
        self.testingLabels  = nparray(self.testingLabels)

        # Tokenize it
        tokenizer = Tokenizer(oov_token=OOV_TOK)
        tokenizer.fit_on_texts(self.trainingSentences)
        self.vocabSize = len(tokenizer.word_index) + 1

        # Sequence the training data
        self.trainingSeqs = tokenizer.texts_to_sequences(self.trainingSentences)
        self.trainingSeqs = nparray(pad_sequences(self.trainingSeqs, padding='post', maxlen=MAX_LENGTH, truncating='post'))

        # Sequence the testing data
        self.testingSeqs = tokenizer.texts_to_sequences(self.testingSentences)
        self.testingSeqs = nparray(pad_sequences(self.testingSeqs, padding='post', maxlen=MAX_LENGTH, truncating='post'))

        return tokenizer


# Load the dataset into a RawData object
# I/P: Path to the data set, and the RawData object
# O/P: Nothing
def LoadData(filePath, rawData) -> None:
    # Load the json data
    with open(filePath, 'r') as file:
        datastore = json.load(file)

    # Prepare lists
    for item in datastore:
        rawData.sentences.append(item['headline'])
        rawData.labels.append(item['is_sarcastic'])
        # rawData.urls.append(item['article_link'])
    
    return