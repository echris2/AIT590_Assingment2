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
#
#

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
    # print(N)
    # print(tokens)
    ngrams_list = zip(*[tokens[i:] for i in range(N)])

    return [" ".join(ngram) for ngram in ngrams_list]

# The method will return relative frequency for N ad N-1 gram tokens.
# @N - Number of n-grams
# @inputCorpus - the name of the files to train the model
#
def generateFQDist(N, inputCorpus):
    ngram_fqdist, nm1gram_freq_dist, sentence_tokens = [], [], []
    # read all files and create tokens
    for fName in inputCorpus:
        with open(fName) as fp:
            tmpSentenceToken = sent_tokenize(fp.read())
            # Process each sentence
            for sentence in tmpSentenceToken:
                # Process each word in the sentence
                wTkn = re.findall(r"[\w]+|[^\s\w]", sentence)
                for wtk in wTkn:
                    sentence_tokens.append(wtk)

    if N > 1 :
        logger.info("processing: "+ str(N) +"-gram")
        n1gram = generateNgrams(N, sentence_tokens)
        logger.info("processing: "+ str(N-1) +"-gram")
        n1Minus1Gram = generateNgrams(N-1, sentence_tokens[0:-1])

        ngram_fqdist = nltk.FreqDist(n1gram)
        nm1gram_freq_dist = nltk.FreqDist(n1Minus1Gram)

        logging.info(len(ngram_fqdist))

        logging.info(len(nm1gram_freq_dist))

    return ngram_fqdist, nm1gram_freq_dist, sentence_tokens


def main():

    start_time = time.localtime()  # get start time

    N, M, inputCorpus = readCommandLineArguments()

    ngram_fqdist, nm1gram_freq_dist, sentence_tokens = generateFQDist(N, inputCorpus)

    generateRandomSentences(ngram_fqdist, nm1gram_freq_dist, sentence_tokens)

    ### log processing
    process_time = time.mktime(time.localtime()) - time.mktime(start_time)  # get process time in seconds

    logging.info("%s minutes, python ngram.py %d", round(process_time / 60, 5))  # log process time in minutes

    logging.info('%exit')  # exit logging
    ### log end


if __name__ == "__main__":
    main()