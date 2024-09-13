from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.models import Sequential # type: ignore


# Save the model to disk, to be reused later on.
# I/P: Model to save, and what name to save it with (extension will be .keras, cannot be changed)
# O/P: Nothing
def SaveModel(model, modelName) -> None:
    try:
        model.save(f'../models/{modelName}.keras')
        print(f'Model saved as {modelName}.keras')

    except Exception:
        print('[ERROR]: Unknown error occured while saving model, exiting...')
        exit()


# Load the model from disk
# I/P: Path of model to be loaded, throws a ValueError if path is invalid
#      Exits program if extension is not '.keras'
# O/P: Loaded model
def LoadModel(modelPath) -> Sequential:

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