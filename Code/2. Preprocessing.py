import nltk
import functools
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
import re
import glob


#A simple functional code - used for preprocessing the corpus
def compose(*functions):
    def compose2(f, g):
        return lambda x: f(g(x))
    return functools.reduce(compose2, functions, lambda x: x)

#Lemmatize every token
def lemmatizer(s):
    lemmatize = WordNetLemmatizer()
    lem_tokens = []
    tokens = nltk.word_tokenize(s)
    for token in tokens:
        lem_tokens.append(lemmatize.lemmatize(token))
    return ' '.join(lem_tokens)

#remove stop words using the list of stop words from NLTK
def remove_stop_words(s):
    result = []
    tokens = nltk.word_tokenize(s)

    #get the list of stop words for english
    en_stops = set(stopwords.words('english'))
    for token in tokens:
        if token not in en_stops:
            result.append(token)

    return result

#Sometimes you may want to remove newlines
def replace_newline_with_space(text):
    return re.sub('\n',' ',text)    

#You may want to hard-code phrases and convert them as a single word
def search_and_replace_phrases(text):
    search_replace_dict = {
        'minister of finance':'finance_minister',
    }
    #for every phrase listed as key in the dictionary, 
    #replace it in the corpus with its value
    for search, replace in search_replace_dict.items():
        text = re.sub(search,replace,text)

    return text

def case_folding(text:str,to='lower')->str:
    if to == 'lower':
        return text.lower()
    else:
        return text.upper()
    
def extract_text(input_file:str)->str:
    with open(input_file, 'r', encoding='utf-8') as file:
        return file.read()
    
def replace_multiple_white_spaces(text):
    # you may also use strip
    return re.sub(r'\s+', ' ', text)

def remove_punctuation(text:str) -> str:
    tokens = nltk.word_tokenize(text)
    return ' '.join(tokens)

def preprocessing(input_files):
    files_processed = 0
    number_files = len(input_files)
    outputfile ='./corpus/budget_speech_en.txt'
    file = open(outputfile, mode='w')
    fc = compose(
                # search_and_replace_phrases,
                replace_multiple_white_spaces,
                case_folding,
                # remove_punctuation,
                # replace_newline_with_space,
                extract_text
                )
    for filename in input_files:
        text = fc(filename)
        # tokens = text.split()
        file.write(text+'\n')
        files_processed = files_processed + 1
        print("\r {0:5d}/{1:3d} of files processed".format(files_processed,number_files),end='')

    file.close()

def main():

    #set the path where the files are located
    path = './TXT'

    #get the list of files fom the folder
    input_file_list = glob.glob(f'{path}/**/*.txt',recursive=True)

    #preprocess and save the final outcome
    preprocessing(input_file_list)


main()