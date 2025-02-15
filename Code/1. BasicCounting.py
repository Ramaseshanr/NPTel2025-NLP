def read_corpus(filename):
    """Reads a corpus from a file and returns a list of tokens."""
    with open(filename, 'r', encoding='utf-8') as file:
        corpus = file.read().lower().split()
    return corpus

def calculate_frequency_table(corpus):
    """Calculates the frequency table of words in the corpus."""
    frequency_table = {}
    for word in corpus:
        frequency_table[word] = frequency_table.get(word, 0) + 1
    return frequency_table

def get_total_tokens(corpus):
    """Returns the total number of tokens in the corpus."""
    return len(corpus)

def get_vocabulary_count(corpus):
    """Returns the count of unique words in the corpus."""
    vocabulary = set(corpus)
    return len(vocabulary)

# Example usage:
corpus = read_corpus('./corpus/budget_speech_en.txt')
frequency_table = calculate_frequency_table(corpus)
total_tokens = get_total_tokens(corpus)
vocabulary_count = get_vocabulary_count(corpus)

# # Print the frequency table
# print("Frequency Table:")
# for word, count in sorted(frequency_table.items(), key=lambda x: x[1], reverse=True):
#     print(f"{word:20s}: {count:4d}")

# Print the total number of tokens and vocabulary count
print(f"Total Tokens: {total_tokens}")
print(f"Vocabulary Count: {vocabulary_count}")
