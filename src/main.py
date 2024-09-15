# oneDNN custom operations are on by default.
# May see slightly different numerical results due to floating-point round-off errors from different computation orders.
# To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
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

    inputText = input('Sentence: ')
    inputSequences = tokenizer.texts_to_sequences([inputText])
    
    return nparray(pad_sequences(inputSequences, maxlen=MAX_LENGTH, padding='post', truncating='post'))


# Entry point of program
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
        # # works fine at 20 epochs
        # tf.keras.layers.GlobalAveragePooling1D(),
        
        # works fine at 20 epochs (might be the best here, needs further testing to confirm)
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),

        # tf.keras.layers.LSTM(64), # not as good
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(1,  activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Iterations over the dataset
    numEpochs = 20

    # Train the model
    model.fit(data.trainingSeqs, data.trainingLabels,
              epochs=numEpochs,
              validation_data=(data.testingSeqs, data.testingLabels),
              verbose=2)


    # Save the model to disk
    saveModelOrNot = input('\n\nSave model to disk? (y/n): ')
    customAddition = input('Append any custom name at end (leave blank if no): ')
    if (saveModelOrNot.lower() == 'y'):
        mdio.SaveModel(model, f'{numEpochs}epochs_{customAddition}')
    print('\n')

    model = mdio.LoadModel('../models/20epochs_GAP1D.keras')

    # Ask user to enter sentences, and predict if they're sarcastic or not
    while True:
        try:
            userInputPaddedSequences = ProcessUserInput(tokenizer)
            prediction = model.predict(userInputPaddedSequences)
            
            # Prediction is <class 'numpy.ndarray'>
            print(f'Prediction: {'{:.4f}'.format(prediction[0][0] * 100)}% sarcastic')
            print()
            print()

        except KeyboardInterrupt:
            break
        
    print('\nExiting...')
    return



if __name__ == '__main__':
    main()