import numpy as np

from constants import *
from tensorflow.keras.preprocessing.sequence import pad_sequences

def ProcessUserInput(tokenizer):

    inputText = list(input('Sentence: ').split())
    inputSequences = tokenizer.texts_to_sequences(inputText)

    return np.array(pad_sequences(inputSequences, maxlen=MAX_LENGTH, padding='post', truncating='post'))