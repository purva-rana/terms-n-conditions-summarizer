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



def ProcessUserInput(tokenizer):
    """
        Get a sentence from the user, and make it into a processable form for the model.

        `tokenizer` - Tokenizer that was used while preparing the training data.
        
        **Returns**\\
        Sequence generated from the user input and tokenizer in an np.array().
    """

    inputText = input('Sentence: ')
    inputSequences = tokenizer.texts_to_sequences([inputText])
    
    return nparray(pad_sequences(inputSequences, maxlen=MAX_LENGTH, padding='post', truncating='post'))



def main():

    # Load data from the dataset
    tempDataPath = '../data/discord/discordTOS.json'
    rawData = dh.LoadData(tempDataPath)

    # Process the raw data
    data = dh.ProcessedData()
    tokenizer = data.ProcessRawData(rawData)


    if input('Train new model? (y/n): ').lower() == 'y':
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
        numEpochs = 50

        # Train the model
        model.fit(data.trainingSeqs, data.trainingLabels,
                epochs=numEpochs,
                validation_data=(data.testingSeqs, data.testingLabels),
                verbose=2)

        # Save the model
        saveModelOrNot = input('\n\nSave model to disk? (y/n): ')
        customAddition = input('Append any custom name at end (leave blank if no): ')
        if (saveModelOrNot.lower() == 'y'):
            mdio.SaveModel(model, f'{numEpochs}epochs_{customAddition}')
        print('\n')


    # Load an existing model
    else:
        print('\nLoading new model...')
        modelName = input('Model name: ')
        model = mdio.LoadModel(f'../models/{modelName}.keras')
        # model = mdio.LoadModel('../models/20epochs_BidirectionLSTM.keras')
        # model = mdio.LoadModel('../models/20epochs_GAP1D.keras')
        # model = mdio.LoadModel('../models/50epochs_BidirectionalLSTM.keras')


    # Ask user to enter sentences, and predict if they're sarcastic or not
    print('\n\n', end='')
    while True:
        try:
            userInputPaddedSequences = ProcessUserInput(tokenizer)
            prediction = model.predict(userInputPaddedSequences)

            # print(userInputPaddedSequences)
            
            # Prediction is <class 'numpy.ndarray'>
            print('Prediction: {:.4f}% malicious'.format(prediction[0][0] * 100))
            print('\n\n', end='')

        except KeyboardInterrupt:
            break
        
    print('\nExiting...')
    return



if __name__ == '__main__':
    main()