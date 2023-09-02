import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from datasets import load_dataset
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
class Preprocessing():
    def __init__(self,path):
        self.path=path
        self.data=None
        self.stemmer = PorterStemmer()
        stop_words = set(stopwords.words('english'))
        self.custom_stopwords = stop_words.copy()
        word_to_remove_from_stopwords = ['couldn', "didn't", "wasn't", "no", "haven't",
                                        "doesn't", "shouldn't", "mustn't", "needn't", "aren't", "wouldn't", "mightn't", "hasn't", "not", "don't", 'didnt', 'wasnt', 'havent', 'doesnt', 'shouldnt', 'mustnt', 'neednt', 'arent', 'wouldnt', 'mightnt', 'hasnt', 'dont']
        for i in word_to_remove_from_stopwords:
            self.custom_stopwords.discard(i)

    def load_data(self):
        self.data=load_dataset(self.path)

    def preprocess_data(self):
        self.load_data()
        self.connect_data()
        self.to_pandas()
        self.data['text'] = self.remove_stop_words(self.data['text'])
        self.data['text'] = self.steming_and_tokenization(self.data['text'])
        self.shuffle_data()
        

    def shuffle_data(self):
        self.data=shuffle(self.data)
        self.data.reset_index(inplace=True, drop=True)

    def to_pandas(self):
        self.data=pd.DataFrame(self.data)

    def connect_data(self):
        combined_data = []
        for split in self.data.keys():
            combined_data.extend(self.data[split])
        self.data=combined_data
        
    def get_X(self):
        return self.data['text']
    
    def get_Y(self):
        return self.data['label']
    
    def steming_and_tokenization(self,data):
        def stem_text(text):
            words=word_tokenize(text)
            stemmed_words = [self.stemmer.stem(word) for word in words]
            return ' '.join(stemmed_words)
        data=data.apply(stem_text)
        return data
        
        
    def remove_stop_words(self,data):
        def stop_words_help(text):
            words=text.split()
            filtered_words = [word for word in words if word.lower(
            ) not in self.custom_stopwords]
            return ' '.join(filtered_words)
        data = data.apply(stop_words_help)
        return data
                