from transformers import pipeline
from datasets import list_datasets
import pandas as pd 
import warnings
warnings.filterwarnings("ignore")

class Translator:

    @classmethod
    def english_to_spanish(self, input_text):
        
        translator = pipeline("translation_en_to_es", model = "Helsinki-NLP/opus-mt-en-es") 

        output = translator(input_text)

        print(output[0]['translation_text'])


if __name__ == "__main__":

    Translator.english_to_spanish("Hi! What is your name?")