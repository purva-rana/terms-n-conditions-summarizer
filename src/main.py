from constants import *
import dataHandler as dh
import modelDiskIO as mdio
import modelProcessing as mpr

import os
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore


def main():
    # May see slightly different numerical results due to floating-point round-off errors from different computation orders.
    # oneDNN custom operations are on by default.
    # To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    # Load data from the dataset
    tempDataPath = '../data/final/final-data-3863.json'
    rawData = dh.LoadDataFromJSON(tempDataPath)

    # Process the raw data
    data = dh.ProcessedData()
    tokenizer = data.ProcessRawData(rawData)

    if input('Train new model? (y/n): ').lower() == 'y':

        numEpochs = 50
        model = mpr.TrainModel(data, numEpochs)

        # Save the model
        while True:
            saveModelOrNot = input('\n\nSave model to disk? (y/n): ')
            if saveModelOrNot in ['y', 'n']:
                break
                
        if (saveModelOrNot.lower() == 'y'):
            customAddition = input('Append any custom name at end (leave blank if no): ')
            mdio.SaveModel(model, f'{numEpochs}ep_{customAddition}')
        print('\n')


    # Load an existing model
    else:
        print('\nSelect model to load')
        modelName = input('Model name: ')
        model = mdio.LoadModel(f'../models/{modelName}.keras')


    print('\n\n         Options')
    print('  1 | Manual sentence entry.')
    print('  2 | Process .txt file.')
    print('  3 | Exit.')
    userChoice = int(input('    Choice: '))

    while userChoice not in [1, 2, 3]:
        print('Invalid choice, try again')
    

    match userChoice:
        case 1:
            try:
                while True:
                    mpr.ProcessSingleInput(tokenizer, model)
            except KeyboardInterrupt:
                print('Exiting...')

        case 2:
            filePath = input('Enter path of file: ')
            print('\n\n         Options')
            print('  1 | Save to file.')
            print('  2 | Display on terminal.')
            saveChoice = int(input('    Choice: '))
            while saveChoice not in [1, 2]:
                print('Invalid choice, try again')
        
            filePath = '../misc/testing.txt'

            sentences, predictions = mpr.ProcessBlockInput(tokenizer, model, filePath)

            if saveChoice == 1:
                dh.SavePredictionsToFile(sentences, predictions)

            else:
                for i in range(len(predictions)):
                    if predictions[i][0][0] > THRESHOLD_VALUE:
                        print(f'\nSentence: {sentences[i]}')
                        print('Prediction: {:.4f}% important\n'.format(predictions[i][0][0] * 100))

    
    return



if __name__ == '__main__':
    main()