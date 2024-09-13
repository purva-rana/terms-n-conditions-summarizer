# What in the python
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import dataHandler as dh
import modelDiskIO as mdio
from userInput import *
from constants import *

import tensorflow as tf
from tensorflow import keras

def main():

    rawData = dh.RawData()
    data = dh.ProcessedData()

    # Load data from the .json dataset
    dh.LoadData(DATA_PATH, rawData)

    tokenizer = data.PrepareData(rawData)

    # Length of output vector
    embeddingDim = 16
    
    # Define and compile the model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(data.vocabSize, embeddingDim),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(1,  activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Iterations over the dataset
    numEpochs = 5

    # Train the model
    model.fit(data.trainingSeqsPadded, data.trainingLabels, epochs=numEpochs, validation_data=(data.testingSeqsPadded, data.testingLabels), verbose=2)

    model.save(f'../models/{numEpochs}.keras')
    print(f'Model saved as {numEpochs}.keras')

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