import os
import nltk
import pickle
import numpy as np
import re
from nltk.corpus import stopwords
from tqdm import tqdm

#if NLTK library is not available install it
#!pip install nltk

# Set up NLTK stopwords
# nltk.download('stopwords')
#get stop words for English
stop_words = set(stopwords.words('english'))

class HAL:
    def __init__(self, corpus_folder, min_count, window_size, nearest_neighbor, farthest_neighbor):
        self.corpus_folder = corpus_folder
        self.min_count = min_count
        self.window_size = window_size
        self.nearest_neighbor = nearest_neighbor
        self.farthest_neighbor = farthest_neighbor
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = {}
        self.word_vectors = None

    def preprocess_corpus(self):
        # Collect word counts
        for file_name in os.listdir(self.corpus_folder):
            file_path = os.path.join(self.corpus_folder, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in tqdm(file, desc=f'{file_name}', unit=' line'):

                # for line in file:
                    tokens = nltk.word_tokenize(line.lower())
                    #remove punctuations
                    tokens_without_punct = [word for word in tokens if re.match(r'\w+', word)]
                    for token in tokens_without_punct:
                        if token not in stop_words:
                            self.word_counts[token] = self.word_counts.get(token, 0) + 1

        # Filter words based on minimum count frequency
        self.word_counts = {word: count for word, count in self.word_counts.items() if count >= self.min_count}

        # Assign indices to words
        for idx, word in enumerate(self.word_counts.keys()):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    '''Compute the word vectors using HAL algorithm'''
    def compute_word_vectors(self):
        #get the vocab count as word_counts holds the unique set of words 
        # whose frequencies are > min_count
        vocab_size = len(self.word_counts)

        #initialize the word vector matrix or incidence matrix
        self.word_vectors = np.zeros((vocab_size, vocab_size), dtype=np.float32)

        file_idx = 0
        file_count = len(os.listdir(self.corpus_folder))

        #for every file, run the window from left-to-right and right-to-left
        #and capture the co-occurences
        for file_name in os.listdir(self.corpus_folder):
            file_path = os.path.join(self.corpus_folder, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                tokens = nltk.word_tokenize(file.read().lower())
                for i, token in tqdm(enumerate(tokens), desc=f'{file_name}/Tokens', unit=' token'):

                # for i, token in enumerate(tokens):
                    # if token not in stop_words:
                        center_word_idx = self.word2idx.get(token)
                        if center_word_idx is not None:
                            # Left-to-right scanning
                            start_idx = max(0, i - self.window_size)
                            end_idx = min(len(tokens), i + self.window_size + 1)
                            #get the context words that fall within the window of size 10
                            context_words = tokens[start_idx:i] + tokens[i+1:end_idx]
                            for context_word in context_words:
                                context_word_idx = self.word2idx.get(context_word)
                                if context_word_idx is not None:
                                    self.word_vectors[center_word_idx, context_word_idx] += 1

                            # Right-to-left scanning
                            start_idx = max(0, i - self.window_size - 1)
                            end_idx = min(len(tokens), i + self.window_size)
                            context_words = tokens[start_idx:i] + tokens[i+1:end_idx]
                            for context_word in context_words:
                                context_word_idx = self.word2idx.get(context_word)
                                if context_word_idx is not None:
                                    self.word_vectors[center_word_idx, context_word_idx] += 1
            file_idx = file_idx + 1
        
            print("\r {0:5d}/{1:3d} of files processed".format(file_idx,file_count),end='')
    
    def save_word_vectors(self, output_file):
        with open(output_file, 'wb') as file:
            pickle.dump(self.word_vectors, file)
            pass

    def load_word_vectors(self, input_file):
        with open(input_file, 'rb') as file:
            self.word_vectors = pickle.load(file)

    def distance_measure(self, word):
        if word in self.word2idx:
            word_idx = self.word2idx[word]
            word_vector = self.word_vectors[word_idx]
            distances = np.linalg.norm(self.word_vectors - word_vector, axis=1)
            sorted_indices = np.argsort(distances)
            similar_words = [self.idx2word[idx] for idx in sorted_indices[1:self.nearest_neighbor+1]]
            return similar_words
        else:
            return [], []

# Example usage
if __name__ == '__main__':
    corpus_folder = './corpus'
    model_file = './Models/WordVectors.hal'
    hal = HAL(corpus_folder, min_count=5, window_size=10, nearest_neighbor=10, farthest_neighbor=1)

    hal.preprocess_corpus()
    hal.compute_word_vectors()
    hal.save_word_vectors(model_file)
#     # Distance measure example
    while (1):
        word = input('\nInput word:')
        similar_words = hal.distance_measure(word)
        print(f"Similar words to '{word}': {similar_words}")


    #Exercise
    #1. Change parameters such as min_count, window_size, nearest_neighbor, farthest_neighbor
    #   note the changes that you find
    #2. Remove unwanted tokens such as number and try the algorithm
    #3. Change the Window size to 5 and accordingly the   nearest_neighbor and values
    #   in the ramped window and and observe the change. Do you find any change
    #   in the list of similar for a given word?