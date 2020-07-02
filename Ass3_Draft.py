# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 11:11:57 2020

@author: eachr
"""
import sys
import os
import re

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import ConditionalFreqDist, ConditionalProbDist, ELEProbDist

# =============================================================================
# training_data = sys.argv[1]
# testing_data = sys.argv[2]
# my_decision_list = sys.argv[3]
# =============================================================================

training_data = "line-data\line-train.xml"
testing_data = "line-data\line-test.xml"
#my_decision_list = sys.argv[3]

decision_list = []
target ="line"


with open(training_data, 'r') as tdata:
    parse_tdat = BeautifulSoup(tdata, 'html.parser')
 
training_list = []
for instance in parse_tdat.find_all('instance'):
    sent = dict()
    sent ['id'] = instance['id']
    sent ['sense'] = instance.answer['senseid']
    text = ""
    for s in instance.find_all('s'):
        text = text+ " "+ s.get_text()
        #processing sentences
        text = text.lower()
        text = re.sub(r'[Ll]ines?','line',text) #standardize plural lines
        text = re.sub(r'[\.\,\?\!\\\'\-\$\%\("]',' ',text)
        
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in stopwords.words('english')]
        
        
        
        
    sent['text'] = tokens
    training_list.append(sent)
    
    ##Conditional Frequency of each word
cfd = ConditionalFreqDist() #inidtialize conditional freq dist




for item in training_list:
    condition = item['sense']
    for word in item['text']:
        if word not in ['line']:
            cfd[condition][word] += 1
            
def add_condition(cfd, data, n):
    for element in data:
        sense, context = element['sense'], element['text']
        n_word = get_positional_n(n, context)
        if n_word != '' and n > 0:
            condition = str(n) + "_ahead_" + re.sub(r'\_', '', n_word)
            cfd[condition][sense] += 1
        elif n_word != '' and n < 0:
            condition = str(n) + "_behind_" + re.sub(r'\_', '', n_word) 
            cfd[condition][sense] += 1
    return cfd
    
def get_positional_n(n, corpus):
    root_index = corpus.index(target) #position of line
    n_word_index = root_index + n #position of target word
    if len(corpus) > n_word_index and n_word_index >= 0:
        return corpus[n_word_index]
    else:
        return ""

cfd = ConditionalFreqDist()
Window = range(-5,5)
for i in Window:
    if i != 0:
        cfd = add_condition(cfd, training_list, i)    
        

cpd = ConditionalProbDist(cfd,ELEProbDist,10)

for cond in cfd.conditions():
    


        