import os
from gensim.models import Word2Vec
import logging

logging.basicConfig(level=logging.INFO)

class Corpus(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for filename in os.listdir(self.dirname):
            for every_line in open(os.path.join(self.dirname, filename), encoding="utf-8", errors="ignore"):
                yield every_line.lower().split()

def compute_word_vectors():
   # text = Corpus('/Users/ram/nltk_data/corpora/gutenberg')
   text = Corpus('./corpus')
   logging.getLogger('gensim.models.word2vec').setLevel(logging.INFO)

   model = Word2Vec(text, vector_size=300, epochs=20, window=5, min_count=5, workers=4, compute_loss=True)
   model.save('./Models/WordVectors.w2v')

def load_model(model_file_name):
   model = Word2Vec.load(model_file_name)
   return model

def get_vocabulary(model):
    return list(model.wv.index_to_key)

def model_parameters():
   # Get model parameters from a trained a Word2Vec
   # Model parameters are already available in this object 'model'
   model = load_model('./Models/WordVectors.w2v')

   # Get the size of the word vectors
   vector_size = model.vector_size
   print("Vector size:", vector_size)

   # Get the vocabulary size
   vocab_size = len(model.wv.key_to_index)
   print("Vocabulary size:", vocab_size)


   # Get the total number of words in the corpus used for training
   corpus_words = model.corpus_total_words
   print("Total words in the corpus:", corpus_words)

   # Get the training loss value
   model.epochs

   # Get the total number of training epochs
   num_epochs = model.epochs
   print("Number of epochs:", num_epochs)

   # Get the context window size used during training
   window_size = model.window
   print("Window size:", window_size)

   # Get the vocabulary list with frequencies
   # vocab = [(word, model.wv.get_vecattr(word, "count")) for word in model.wv.key_to_index]

   # Sort the vocabulary list by frequency in descending order
   # sorted_vocab = sorted(vocab, key=lambda x: x[1], reverse=True)

   # Print the vocabulary list with frequency
   # for word, freq in sorted_vocab:
   #    print(word, freq)

   del model

def similar_words(word='finance'):
   # Get similar words for a given word
   model = load_model('./Models/WordVectors.w2v')
   similar_words = model.wv.most_similar(word)
   for word, similarity in similar_words:
      print(f"{word:20s}:{similarity:.4f}")
   del model

if __name__ == '__main__':
   build_or_predict = int(input('1. Build\n2. Find Similar words\n(1/2)'))
   if build_or_predict == 1:
      compute_word_vectors()
   else:
      model_parameters()
      while (1):
         word = input('Similar word for: ')
         similar_words(word)
