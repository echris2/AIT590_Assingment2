#!/usr/bin/env python
# coding: utf-8

# ### Programming Assignment 2: N-Gram ###
# ![](Eliza.PNG)
# <br>
# **Team 3:**
# <br>
# Srikanth Vadlamani, Jason Drahos, Eric Christiansen, Stephen Andre
#
# Citation:
#       https://rstudio-pubs-static.s3.amazonaws.com/115676_ab6bb49748c742b88127e8b5ce3e1298.html
#       https://www.geeksforgeeks.org/word-prediction-using-concepts-of-n-grams-and-cdf/
#       https://medium.com/@davidmasse8/predicting-the-next-word-back-off-language-modeling-8db607444ba9
#        https://web.stanford.edu/class/archive/cs/cs106b/cs106b.1176/assn/ngrams.html

import sys
import os.path
from os import path
import random
import re
import nltk
import time

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk import FreqDist
from nltk.util import ngrams
import string
punctuations = list(string.punctuation)
import logging


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

log_format = '%(asctime)s %(filename)s: %(message)s'
logging.basicConfig(filename="ngram-log.txt", format=log_format,
                    datefmt='%Y-%m-%d %H:%M:%S')

logger.info("=====N-Gram execution Start==========")


def readCommandLineArguments():
    # print(len(sys.argv))
    if len(sys.argv) < 5:
        print("Error: Incorrect arguments. Please enter command in followsing format:")
        print()
        print("python3 Assignment2-Ngram.py 3 10 pg2554.txt pg2600.txt pg1399.txt")
        print()
        print("where    3 - n-grams")
        print("         10 - number of random sentences")
        print("         pg2554.txt: Represents first text files to train the model.")
        print("         pg2600.txt  Represents second text files to train the model.")
        print("         pg1399.txt: Represents third text files to train the model.")
        sys.exit()
    elif(int(sys.argv[1]) <= 0 ):
        print("Error: Incorrect arguments. Please enter N-Gram value greater than or equal to 1")
        sys.exit()
    elif (int(sys.argv[2]) <= 0):
        print("Error: Incorrect arguments. Please enter total number of sentences a value greater than or equal to 1")
        sys.exit()
    elif not (path.exists(sys.argv[3]) and not os.stat(sys.argv[3]).st_size < 10):
        print("Error: Invalid File: " + sys.argv[3] + "  does not exist or too small to use.")
        sys.exit()
    elif not (path.exists(sys.argv[4]) and not os.stat(sys.argv[4]).st_size < 10):
        print("Error: Invalid File: " + sys.argv[3] + "  does not exist or too small to use.")
        sys.exit()
    elif not (path.exists(sys.argv[5]) and not os.stat(sys.argv[5]).st_size < 10):
        print("Error: Invalid File: " + sys.argv[3] + "  does not exist or too small to use.")
        sys.exit()


    # get number of n-grams to use
    nGram = int(sys.argv[1])

    #Get number of sentences to generate
    mSentences = int(sys.argv[2])

    # Read text corpus
    txtFiles = [str(f) for f in sys.argv[3:]]

    return nGram, mSentences, txtFiles

# use zip function to create N-gram based on tokens
def generateNgrams(N, tokens):
    '''
    

    Parameters
    ----------
    N : TYPE integer
        DESCRIPTION. n-count n-grams to return, eg. 2 is bigrams
    tokens : TYPE list of strings
        DESCRIPTION. list of word tokes

    Returns
    -------
    list
        DESCRIPTION. n-grams in list form

    '''
    # print(N)
    # print(tokens)
    ngrams_list = zip(*[tokens[i:] for i in range(N)])

    return [" ".join(ngram) for ngram in ngrams_list]

# The method will return relative frequency for N ad N-1 gram tokens.
# @N - Number of n-grams
# @inputCorpus - the name of the files to train the model
#
def generateFQDist(N, inputCorpus):
    '''

    Parameters
    ----------
    N : TYPE integer
        DESCRIPTION. n-count n-grams to return, eg. 2 is bigrams
    inputCorpus : TYPE list of text files
        DESCRIPTION.  list of text file from which to generate ngram distribution

    Returns
    -------
    ngram_fqdist : TYPE dictionary
        DESCRIPTION. Key: specific n-gram, Value: Count of ngram
    nm1gram_freq_dist : TYPE
        DESCRIPTION. specific n-1gram, Value: Count of n-1 gram
    sentence_tokens : TYPE list
        DESCRIPTION. list of tokens within each sentence

    '''
    ngram_fqdist, nm1gram_freq_dist, sentence_tokens = [], [], []
    # read all files and create tokens
    for fName in inputCorpus:
        with open(fName, encoding='utf-8') as fp:
            tmpSentenceToken = sent_tokenize(fp.read())
            # Process each sentence
            for sentence in tmpSentenceToken:
                # Process each word in the sentence
                wTkn = re.findall(r"[\w]+|[^\s\w]", "BGN "+sentence+" EOF") #add BGN and EOF tags
                for wtk in wTkn:
                    sentence_tokens.append(wtk)

        logger.info("processing: "+ str(N) +"-gram")
        n1gram = generateNgrams(N, sentence_tokens)
        ngram_fqdist = nltk.FreqDist(n1gram)
        logging.info(len(ngram_fqdist))

        # Only process when N-Gram 2 or more
        if N > 1:
            logger.info("processing: "+ str(N-1) +"-gram")
            '''
                                                        SRI - wouldn't we want BGN and END tags at the 
                                                        start adn finish of end corpus?
            
            see below
            '''
            n1Minus1Gram = generateNgrams(N-1, sentence_tokens[0:-1])
            
            nm1gram_freq_dist = nltk.FreqDist(n1Minus1Gram)
            logging.info(len(nm1gram_freq_dist))

    return ngram_fqdist, nm1gram_freq_dist, sentence_tokens


# get the given word/words
def get_previous_n_words(i, N, my_tokens):
    temp = []
    for j in range(1, N):
        temp.append(my_tokens[i - j])
    # why
    temp.reverse()
    previous_n_words = ' '.join(map(str, temp))
    return previous_n_words

def calculate_relative_frq(N,ngram_fqdist, nm1gram_freq_dist, sentence_tokens):
    '''
    

    Parameters
    ----------
    N : TYPE integer
        DESCRIPTION. n of n-grams to use
        ngram_fqdist : TYPE dictionary
        DESCRIPTION. Key: specific n-gram, Value: Count of ngram
    nm1gram_freq_dist : TYPE
        DESCRIPTION. specific n-1gram, Value: Count of n-1 gram
    sentence_tokens : TYPE list
        DESCRIPTION. list of all tokens, ordered as sentences

    Returns
    -------
    relative_frequency_dictionary : TYPE dictionary
        DESCRIPTION. realtive frequency of the n-grams

    '''
    # variables to store N-Gram relative frequency and N-1 gram Relative Frequency
    dict_ngram, dict_nm1gram = {}, {}

    # print(len(sentence_tokens))
    # print(len(ngram_fqdist))
    # print(len(nm1gram_freq_dist))

    # Check lecture slide 50 in Lecture 4 - Ngrams_rev(1) for explanation
    # Proportion of n-grams of total n-grams
    # Proportionate share of n-gram in corpus
    for key, value in ngram_fqdist.items():
        dict_ngram[key] = value/len(ngram_fqdist)

    # Check lecture slide 50 in Lecture 4 - Ngrams_rev(1) for explanation
    # n-1 gram as proportion of total n-1 grams
    # Proportionate share of n-1 grams in corpus
    for key, value in nm1gram_freq_dist.items():
        dict_nm1gram[key] = value/len(nm1gram_freq_dist)

    # formula for relative frequency freq(Xk-1, Xk)/freq(Xk-1)
    relative_frequency_dictionary = {}
    #get token of n-length to use for single table
    for i in range(N - 1, len(sentence_tokens)):  # iterate all the tokens
        previous_n_words = get_previous_n_words(i, N, sentence_tokens)
        # print(i, condi_token)

        relative_frequency_dictionary[(sentence_tokens[i], 
                                       previous_n_words)] = \
                                                            dict_ngram[previous_n_words +
                                                            " " + sentence_tokens[i]]/dict_nm1gram[previous_n_words]

    return relative_frequency_dictionary
   



def predictNextWord(next_keys, relative_frequency_dictionary):
    total_probability, temp = 0, 0
    word = ''
    select_list = []
 

    # calculate the sum of the probability
    for key in next_keys:
        total_probability += relative_frequency_dictionary[key]

    # normalize and calculate the cumulative relative probability.
    # create a list = [[prediction, given, cumulative relative probability] ... ]
    for key in next_keys:
        temp = temp + relative_frequency_dictionary[key] / total_probability
        select_list.append([key[0], key[1], temp])

    prob = random.uniform(0, 1)  # assign a random number between 0 and 1

    # use binary search to find where "prob" falls
    left = 0
    right = len(select_list) - 1

    while (right - left > 1):
        mid = int((left + right) / 2)

        if select_list[mid][2] > prob:
            right = mid
        elif select_list[mid][2] < prob:
            left = mid
        else:
            word = select_list[mid][0]

    if select_list[left][2] > prob:
        word = select_list[left][0]
    elif select_list[left][2] < prob:
        word = select_list[right][0]
    else:
        word = select_list[left][0]

    return word


def generateRandomSentences_Eric(N, M, relative_frequency_dictionary):
    for m in range(0, M):
        start_keys = []

        # Find the keys that start with BGN. Example, "BGN the" or "BGN every".
        # Also, relative frequency is not 0, which implies the word is used elsewhere.
        start_keys = [key for key in relative_frequency_dictionary.keys() if key.split(" ")[0] == 'BGN' and relative_frequency_dictionary[key] != 0]
        # print("start_keys: ", start_keys)

        # Choose a random key word from above step. 
        start_gram = random.choice(start_keys)
        #extract the last word in that n-gram
        last_word = start_gram[-1]
        # print()
        # print("sentence's first_word: ", first_word)

        notEnd = True

        # Sentnce begins with the start gram"
        sentence = start_gram
        
        while (notEnd):
            tokens = sentence.split()
            given = tokens[-(N - 1):]
            # Get the next word after the first word
            given = ' '.join(map(str, given))  # create given word/words
            # print("given tokens: ", given)

            # find subsequent words with word from previous step and relative frequency is not 0
            next_keys = [key for key in relative_frequency_dictionary.keys() if given == key.split()[0]]
            # print("next_keys: ", next_keys)

            # re-arrange words according the relative frequency  Slide 28 of 40 from W2 review class
            next_word = predictNextWord(next_keys, relative_frequency_dictionary)
            # print("next_word: ", next_word)

            sentence = sentence + " " + next_word  # add the selected word to the sentence

            # if the selecte word includes "EOF", then end the setence creation
            if next_word == 'EOF':
                notEnd = False

        sentence = re.sub(r'BGN', '', sentence)  # remove "BGN" from the sentence
        sentence = re.sub(r'EOF', '', sentence)  # remove "EOF" from the sentence
        print("Sentence", m + 1, ": ", sentence)
        print()

# generate sentences for N-gram
def generateRandomSentences(N, M, relative_frequency_dictionary):
    for m in range(0, M):
        start_keys = []

        # Find the keys that start with BGN. Example, "BGN the" or "BGN every".
        # Also, relative frequency is not 0, which implies the word is used elsewhere.
        start_keys = [key for key in relative_frequency_dictionary.keys() if 'BGN' == key[1].split(" ")[0] and relative_frequency_dictionary[key] != 0]
        # print("start_keys: ", start_keys)

        # Choose a random key word from above step. 
        #Skip BGN Tag to 
        # For example, get keys with key "every person in"
        first_word = random.choice(start_keys)[1]
        # print()
        # print("sentence's first_word: ", first_word)

        notEnd = True

        # The first word is now " every "
        sentence = first_word

        while (notEnd):
            tokens = sentence.split()
            ##tokens without BGN tag
            given = tokens[-(N - 1):]
            # Get the next word after the first word
            given = ' '.join(map(str, given))  # create given word/words
            # print("given tokens: ", given)

            # find subsequent words with word from previous step and relative frequency is not 0
            next_keys = [key for key in relative_frequency_dictionary.keys() if given == key[1]]
            # print("next_keys: ", next_keys)

            # re-arrange words according the relative frequency  Slide 28 of 40 from W2 review class
            next_word = predictNextWord(next_keys, relative_frequency_dictionary)
            # print("next_word: ", next_word)

            sentence = sentence + " " + next_word  # add the selected word to the sentence

            # if the selecte word includes "EOF", then end the setence creation
            if next_word == 'EOF':
                notEnd = False

        sentence = re.sub(r'BGN', '', sentence)  # remove "BGN" from the sentence
        sentence = re.sub(r'EOF', '', sentence)  # remove "EOF" from the sentence
        print("Sentence", m + 1, ": ", sentence)
        print()

def main():

    start_time = time.localtime()  # get start time
    # inputCorpus = ["../twain1.txt"]
    # N = 3
    # M = 10
    N, M, inputCorpus = readCommandLineArguments()

    ngram_fqdist, nm1gram_freq_dist, sentence_tokens = generateFQDist(N, inputCorpus)

    relative_frequency_dictionary = calculate_relative_frq(N, ngram_fqdist, nm1gram_freq_dist,sentence_tokens)

    # logger.info(relative_frequency_dictionary)
    generateRandomSentences(N, M, relative_frequency_dictionary)

    ### log processing
    process_time = time.mktime(time.localtime()) - time.mktime(start_time)  # get process time in seconds

    logging.info("%s minutes, python ngram.py %d", round(process_time / 60, 5), N)  # log process time in minutes

    logging.info('%exit')  # exit logging
    ### log end


if __name__ == "__main__":
    main()