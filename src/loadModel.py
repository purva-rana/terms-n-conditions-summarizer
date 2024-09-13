# What in the python
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import dataHandler as dh
import modelDiskIO as mdio
from userInput import *
from constants import *

import tensorflow as tf

def main():

    di = dh.RawData()
    data = dh.ProcessedData()

    # Load data from the .json dataset
    dh.LoadData(DATA_PATH, di)

    tokenizer = data.PrepareData(di)

    # Length of output vector
    embeddingDim = 16

    # Load the model
    model = mdio.LoadModel('../models/50.keras')
    print(type(model))
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