# What in the python
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import dataHandler as dh
from userInput import *
from constants import *

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def main():

    di = dh.DataInstance()
    data = dh.Data()

    # Load data from the .json dataset
    dh.LoadData(DATA_PATH, di)

    tokenizer = data.PrepareData(di)

    # Length of output vector
    embeddingDim = 16

    # Load the model
    model = tf.keras.models.load_model('models/1.h5')
    print('Model Loaded')

    while True:
        try:
            userInputPaddedSequences = ProcessUserInput(tokenizer)
            prediction = model.predict(userInputPaddedSequences)

            print('Prediction: ', prediction[0][0])
            print()
            print()
        except KeyboardInterrupt:
            break
        
    print('\nExiting...')
    return

    

if __name__ == '__main__':
    main()

# 1 - Summarizer
# 2 - Sentiment analysis to analyze text and flag harmful or potentially malicious statement in the T&C