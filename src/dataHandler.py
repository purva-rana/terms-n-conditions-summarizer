from constants import *
import json
from numpy import array as nparray
from nltk.tokenize import sent_tokenize
# "type: ignore" here removes the annoying import error that only VSC shows (executes properly though)
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore


def SplitSentences(trainingSize: int, sentences: list) -> list | list:
    """
        Split the sentences into training and testing sentences.

        `trainingSize` - no. of sentences to reserve for training\\
        `sentences` - the sentences to split

        **Returns**\\
        Split sentences
    """
    # return sentences[0:trainingSize], sentences[trainingSize:]
    return sentences, sentences



def SplitLabels(trainingSize: int, labels: list) -> list | list:
    """
        Split the labels into training and testing labels.

        `trainingSize` - no. of labels to reserve for training\\
        `labels` - the labels to split

        **Returns**\\
        Split labels
    """
    # return labels[0:trainingSize], labels[trainingSize:]
    return labels, labels



# Object to hold the raw data loaded initially from the dataset
class RawData:
    def __init__(self) -> None:
        self.sentences = [] # Actual sentence
        self.labels = []    # is_flagged or not (boolean)
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



    def ProcessRawData(self, rawData: RawData):
        """
            Convert `rawData` into a `ProcessedData` object

            `rawData` - A `RawData` object with data loaded from the dataset.

            **Returns**\\
            Tokenizer used on the training sentences
        """

        # Prepare the training and testing sentences
        self.trainingSentences, self.testingSentences = SplitSentences(TRAINING_SIZE, rawData.sentences)
        self.trainingLabels,    self.testingLabels    = SplitLabels(TRAINING_SIZE, rawData.labels)

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



def LoadDataFromJSON(filePath: str) -> RawData:
    """
        Load the dataset into a RawData object.
        
        `filePath` - path to the data set.

        **Returns**\\
        The loaded `RawData()` object
    """

    rawData = RawData()

    # Load the json data
    with open(filePath, 'r') as file:
        datastore = json.load(file)

    # Prepare lists
    for item in datastore:
        rawData.labels.append(item['is_flagged'])
        rawData.sentences.append(item['sentence'])
    
    return rawData



def LoadTextFromFile(filePath: str) -> list[str]:
    """
        The user will give a file path to a TnC/ToS document.
        Load it, and convert the whole document into sentences.

        `filePath` - path to the file.

        **Returns**\\
        The sentences
    """

    with open(filePath, 'r', encoding='utf-8') as file:
        paragraphs = file.read()

    return sent_tokenize(paragraphs)


def SavePredictionsToFile(sentences, predictions):
    # Ask the user where to save the file
    filePath = input("Full path of where to save: ")

    if filePath[-4:] != '.txt':
        filePath += '.txt'

    try:
        with open(filePath, 'w') as file:
            for sentence, prediction in zip(sentences, predictions):
                file.write(f'Sentence: {sentence}\n')
                file.write('Prediction: {:.4f}% important\n\n\n'.format(prediction[0][0] * 100))
                
        print(f"File saved successfully at {filePath}")
        
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")