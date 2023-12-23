from transformers import pipeline
from datasets import list_datasets
import pandas as pd 
import warnings
warnings.filterwarnings("ignore")

print("imports successful.")

classifier = pipeline("text-classification")

class TextClassifier:

    @classmethod
    def test(self):

        sample_text = "what a great day!"

        output = classifier(sample_text)

        print("The statement is {} with a {}% certainty".format(output[0]['label'], int(output[0]['score'] * 100)))


if __name__ == "__main__":
    
    TextClassifier.test()