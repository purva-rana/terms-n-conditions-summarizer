# Terms and Conditions Summarizer

The Terms and Conditions Summarizer uses natural language processing (NLP) techniques to analyze the text of terms and conditions documents. It identifies key points and important clauses, and then classifies each statement to determine if it is potentially malicious.
The highlighted clauses are then presented to the user for review, along with a confidence percentage that indicates the likelihood of the statement being malicious. The user can then make an informed decision about whether to accept or reject the terms and conditions based on the highlighted clauses.

## Features
- Parses terms and conditions documents
- Extracts key points and important clauses
- Classifies statements as malicious or not

- Save highlighted clauses to a file
- Save and load trained models

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python main.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Currently in testing phase
We're working on improving the accuracy of the classification model. The data considered has been custom labelled by us by sifting through various terms and conditions documents.