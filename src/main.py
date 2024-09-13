# What in the python
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from constants import *
import dataHandler as dh
import modelDiskIO as mdio


from numpy import array as nparray
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore


# Get a sentence from the user, and make it into a processable form for the model
# I/P: Tokenizer that was used while preparing the training data
# O/P: Sequence generated from the user input and tokenizer in an np.array()
def ProcessUserInput(tokenizer):

    inputText = list(input('Sentence: ').split())
    inputSequences = tokenizer.texts_to_sequences(inputText)

    return nparray(pad_sequences(inputSequences, maxlen=MAX_LENGTH, padding='post', truncating='post'))


def main():

    # Load data from the dataset
    # DATA_PATH defined in constants.py
    rawData = dh.RawData()
    dh.LoadData(DATA_PATH, rawData)

    # Process the raw data
    data = dh.ProcessedData()
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

    # Save the model to disk
    saveModelOrNot = input('Save model to disk? (y/n): ')
    if (saveModelOrNot.lower() == 'y'):
        mdio.SaveModel(model, f'{numEpochs}epochs.keras')


    # Ask user to enter sentences, and predict if they're sarcastic or not
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