from transformers import pipeline
from datasets import list_datasets
import pandas as pd 
import warnings
warnings.filterwarnings("ignore")

class Generator:

    @classmethod
    def test(self, query, prompt):

        generator = pipeline("text-generation")
        prompt_ = query + "\n\nCUSTOMER SERVICE RESPONSE : \n" + prompt 
        output = generator(prompt_, max_length = 250)

        print(output[0]["generated_text"])


if __name__ == "__main__":

    query = "Hi, I ordered a book from your website which appears to be delayed. Could you please let me know what the latest status is and by when will the book arrive?"
    Generator.test(query = query, prompt = "Hi, I am sorry to hear that your order has been delayed and I sincerely apologize for the inconvenience.")