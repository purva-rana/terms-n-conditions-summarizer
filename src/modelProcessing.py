import dataHandler as dh
from constants import *

from numpy import array as nparray
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore


def TrainModel(data : dh.ProcessedData, numEpochs : int):
    """
       Trains a tf.keras model with the given data

       `data` - The processed data.\\
       `numEpochs` - Number of epochs to fit the model with.
       
       **Returns**\\
       The model
    """
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

    # Train the model
    model.fit(data.trainingSeqs, data.trainingLabels,
            epochs=numEpochs,
            validation_data=(data.testingSeqs, data.testingLabels),
            verbose=2)
    
    return model



def ProcessSingleInput(tokenizer, model):
    """
        Get a single sentence from the user through std input, and predict it
        
        `tokenizer` - Tokenizer used on the dataset.\\
        `model` - Trained AI model.\\
    """
    # Ask user to enter sentences, and predict if they're sarcastic or not
    print('\n', end='')
    try:
        inputText = input('Sentence: ')
        inputSequences = tokenizer.texts_to_sequences([inputText])
    
        userInputPaddedSequences = nparray(pad_sequences(inputSequences, maxlen=MAX_LENGTH, padding='post', truncating='post'))
        prediction = model.predict(userInputPaddedSequences)

        # print(userInputPaddedSequences)
        
        # Prediction is <class 'numpy.ndarray'>
        print('Prediction: {:.4f}% important'.format(prediction[0][0] * 100))
        print('\n\n', end='')

    except Exception:
        print('[ERROR]: An exception occured in modelProcessing.py -> ProcessSingleInput')
        print('         Terminating function')

    return



def ProcessBlockInput(tokenizer, model, filePath : str):
    """
        Makes the model predict every sentence given.

        `tokenizer` - Tokenizer used on the dataset.\\
        `model` - Trained AI model.\\
        `sentences` - List of sentences to predict.
    """

    try:
        # Get user sentences
        print('Loading data...')
        sentences = dh.LoadTextFromFile(filePath)

        sentenceSequences = []
        # Tokenize the loaded sentences
        print('Tokenizing...')
        for sentence in sentences:
            sentenceSequences.append(tokenizer.texts_to_sequences([sentence]))
            sentenceSequences[-1] = nparray(pad_sequences(sentenceSequences[-1], maxlen=MAX_LENGTH, padding='post', truncating='post'))

        predictions = []

        print('Predicting...')
        for sentSeq in sentenceSequences:
            predictions.append(model.predict(sentSeq, verbose=None))
        print('Completed')

    except KeyboardInterrupt:
        print('Stopping block analysis\n')
        return sentences, predictions
    
    except Exception:
        print('[ERROR]: An exception occured in modelProcessing.py -> ProcessBlockInput()')
        print('         Terminating function')
        return
    
    return sentences, predictions