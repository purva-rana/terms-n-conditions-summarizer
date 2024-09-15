import os

from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.models import Sequential # type: ignore


def SaveModel(model, modelName: str) -> None:
    """
        Save the model to disk, to be reused later on.

        `model` - model to save\\
        `modelName` - path and name to save as (ex: '../models/exampleModel')\\
        extension will be '.keras'\\
        The above example will be saved to ../models/exampleModel.keras
    """
    
    basePath = f'../models/{modelName}.keras'
    
    try:
        # Check if file already exists
        if os.path.exists(basePath):
            print(f'The file {modelName}.keras already exists.')
            user_input = input('Do you want to overwrite it? (y/n): ').strip().lower()
            
            # Overwrite the existing file
            if user_input == 'y':
                os.remove(basePath)
                model.save(basePath)
                print(f'Deleting old file {modelName}.keras')
                print(f'Model saved as {modelName}.keras')

            # Find a new name with an incrementing suffix
            else:
                counter = 1
                newModelName = f'{modelName}_{counter}.keras'

                while os.path.exists(f'../models/{newModelName}'):
                    counter += 1
                    newModelName = f'{modelName}_{counter}.keras'
                
                # Save the model with the new name
                model.save(f'../models/{newModelName}')
                print(f'Model saved as {newModelName}')
        
        # Save the model directly if no conflict
        else:
            model.save(basePath)
            print(f'Model saved as {modelName}.keras')
        
    except Exception as e:
        print(f'[ERROR]: An error occurred while saving the model: {str(e)}')
        exit()
    
    return



def LoadModel(modelPath: str) -> Sequential:
    """
        Load the model from disk.

        `modelPath` - path of the model to be loaded.

        **Returns**\\
        Loaded model
        
        Throws a ValueError if path is invalid.\\
        Exits program if extension is not '.keras'.

    """

    if modelPath[-6:] != '.keras':
        print(f'[ERROR]: File extension invalid, should end in ".keras", exiting...')
        exit()
    
    try:
        model = load_model(modelPath)

    except ValueError:
        print(f'[ERROR]: File "{modelPath}" not found, exiting...')
        exit()
        
    except Exception:
        print('[ERROR]: Unknown error occured wihle loading model, exiting...')
        exit()
    
    return model



def main():
    return

if __name__ == '__main__':
    main()