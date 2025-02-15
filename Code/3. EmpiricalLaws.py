import nltk
#!pip install nltk
#!pip install matplotlib
from nltk import FreqDist
import matplotlib.pyplot as plt
import mplcursors
import os
import numpy as np

class Counting():
    def __init__(self,corpus_path):
        with open(corpus_path, 'r', encoding='utf-8') as file:
            self.corpus = file.read().lower()
        self.tokens = self.get_tokens()
        self.vocabulary = self.vocabulary()

    def get_tokens(self):
        return nltk.word_tokenize(self.corpus)

    def vocabulary(self):
        return set(self.tokens)
    
    def plot_zipf_law(self):
        # Calculate word frequency using NLTK's FreqDist
        freq_dist = FreqDist(self.tokens)

        # Sort the words by frequency in descending order
        sorted_words = sorted(freq_dist.items(), key=lambda x: x[1], reverse=True)

        # Extract the word frequencies and ranks
        frequencies = [freq for _, freq in sorted_words]
        ranks = list(range(1, len(frequencies) + 1))

        # Plot Zipf's law
        plt.figure(figsize=(10, 6))
        plt.loglog(ranks, frequencies, marker='.', linestyle='solid', color='b')
        plt.xlabel('Rank')
        plt.ylabel('Frequency')
        plt.title("Zipf's Law")
        plt.grid(True)
        plt.text(1000, 45000, f'Vocabulary Count: {len(self.vocabulary)}', fontsize=12, color='blue')
        plt.text(1000, 23000, f'Number of tokens: {len(self.tokens)}', fontsize=12, color='green')
        plt.text(1000, 10000, f'Type/Token ratio: {self.type_token_ratio():.2f}', fontsize=12, color='red')

        # Add annotations on mouseover
        cursor = mplcursors.cursor(hover=True)

        @cursor.connect("add")
        def on_add(cursor_position):
            x, y = cursor_position.target[0], cursor_position.target[0]
            cursor_position.annotation.set_text(f'({sorted_words[int(round(x))][0]}, {sorted_words[int(round(x))][1]})')

        # plt.savefig("./Images/zipf.pdf", format="pdf", bbox_inches="tight")

        plt.show()

    def type_token_ratio(self):
        # Calculate the number of unique words (types) and the total number of words (tokens)
        types = len(self.vocabulary)
        tokens = len(self.tokens)

        # Calculate the Type-Token Ratio (TTR)
        ttr = types / tokens
        return ttr


    def on_add(sel):
        x, y = sel.target[0], sel.target[1]
        sel.annotation.set_text(f'({x}, {y})')

    def calculate_heap(this, corpus_folder):
        corpus_files = os.listdir(corpus_folder)
        #This stores the vocabulry of each file
        vocabulary_sizes = []
        #This stores the number of tokens in each file
        corpus_sizes = []

        for file_name in corpus_files:
            file_path = os.path.join(corpus_folder, file_name)

            # Read the file
            with open(file_path, 'r', encoding='utf-8') as file:
                corpus = file.read().lower()

            # Tokenize the corpus into individual words
            tokens = corpus.split()

            # Calculate the vocabulary size and corpus size
            vocabulary_size = len(set(tokens))
            corpus_size = len(tokens)

            # Add the values to the lists
            vocabulary_sizes.append(vocabulary_size)
            corpus_sizes.append(corpus_size)

        return vocabulary_sizes, corpus_sizes


    def fit_heaps_law(self,vocabulary_sizes, corpus_sizes):
        # Convert the lists to arrays for fitting
        vocabulary_sizes = np.array(vocabulary_sizes)
        corpus_sizes = np.array(corpus_sizes)

        # Fit Heap's Law model using log-log transformation
        log_corpus_sizes = np.log(corpus_sizes)
        log_vocabulary_sizes = np.log(vocabulary_sizes)

        # Fit a linear regression line
        coeffs = np.polyfit(log_corpus_sizes, log_vocabulary_sizes, deg=1)
        beta = coeffs[0]
        k = np.exp(coeffs[1])

        return k, beta


    def plot_heaps_law(self, vocabulary_sizes, corpus_sizes, k, beta):
        # Plot vocabulary size against corpus size
        plt.scatter(corpus_sizes, vocabulary_sizes, color='b', alpha=0.5)
        plt.xlabel('Corpus Size')
        plt.ylabel('Vocabulary Size')

        # Plot Heap's Law line
        x = np.array(corpus_sizes)
        y = k * (x ** beta)
        plt.plot(x, y, color='r')

        # Set logarithmic scale for better visualization
        plt.xscale('log')
        plt.yscale('log')
        
        plt.text(3000, 800, f'\u03B2 =  {beta:.2f} and K = {k:.2f}', fontsize=12, color='blue')

        plt.title("Heap's Law")
        plt.grid(True)
        # plt.savefig('Zipfs.pdf')
        plt.show()


    def heaps_law(self):
        vocabulary_sizes, corpus_sizes = self.calculate_heap('./TXT')
        k, beta = self.fit_heaps_law(vocabulary_sizes, corpus_sizes)
        self.plot_heaps_law(vocabulary_sizes, corpus_sizes, k, beta)

if __name__ ==  '__main__':
    corpus = Counting('./corpus/budget_speech_en.txt')

    corpus.plot_zipf_law()
    corpus.heaps_law()


    del corpus

