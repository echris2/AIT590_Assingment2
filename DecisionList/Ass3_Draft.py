# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 11:11:57 2020

@author: eachr
"""
import sys
import os
import re
import math

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import ConditionalFreqDist, ConditionalProbDist, ELEProbDist

# =============================================================================
# training_data = sys.argv[1] #read training data
# testing_data = sys.argv[2] #read testing data
# final_decision_list = sys.argv[3] #output list for decision rules
# =============================================================================

training_data = "line-data\line-train.xml"
testing_data = "line-data\line-test.xml"
final_decision_list = "line-data\DL.txt"

decision_list = []
target ="line" # word sense target to test

with open(training_data, 'r') as tdata:
    parse_tdat = BeautifulSoup(tdata, 'html.parser') #parse training data
 
training_list = [] #init holding list for training
for instance in parse_tdat.find_all('instance'):
    sent = dict() #dictionary to hold elements of each instance
    sent ['id'] = instance['id'] # ket is id
    sent ['sense'] = instance.answer['senseid'] #value is id
    text = "" #init text string for sentence
    for s in instance.find_all('s'):
        text = text+ " "+ s.get_text()
        #processing sentences
        text = text.lower()
        text = re.sub(r'[Ll]ines?','line',text) #standardize plural lines
        text = re.sub(r'[\.\,\?\!\\\'\-\$\%\(\);&"]',' ',text) #remove special characters
        
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in stopwords.words('english')]# remove stop words      
    sent['text'] = tokens
    training_list.append(sent)
    
##Conditional Frequency of each word
phone_count = 0 #init count of each sense
product_count = 0

for item in training_list:
    ##count for majority class
    if item['sense'] == 'phone':
        phone_count +=1 
    elif item['sense'] == 'product':
        product_count +=1
    condition = item['sense']

#output majority class  
if phone_count > product_count:
    print('The majority class is phone. Default rule will be assign to this class')
    majority = "phone"
    
elif phone_count < product_count:
    print('The majority class is product. Default rule will be assign to this class')
    majority = "product"

         
def add_condition(cfd, data, n):
    '''
    

    Parameters
    ----------
    cfd : TYPE cfd Object
        DESCRIPTION. cfd to add condition to
    data : TYPE data to count
        DESCRIPTION.
    n : TYPE index relation to target word
        DESCRIPTION.

    Returns
    -------
    cfd : TYPE cfd objects
        DESCRIPTION. cfd model with conditional frequencies <index_position_word>

    '''
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
    '''
    

    Parameters
    ----------
    n : TYPE integer
        DESCRIPTION. index position in relation to target word
    corpus : TYPE list of tokens
        DESCRIPTION.

    Returns
    -------
    TYPE string
        DESCRIPTION. target word, or blank if not possible

    '''
    root_index = corpus.index(target) #position of line
    n_word_index = root_index + n #position of target word
    if len(corpus) > n_word_index and n_word_index >= 0:
        return corpus[n_word_index]
    else:
        return ""

cfd = ConditionalFreqDist()
nWindow = range(-2,1) #range for n
for i in nWindow:
    if i != 0:
        cfd = add_condition(cfd, training_list, i)    
        

cpd = ConditionalProbDist(cfd,ELEProbDist,10)
temp = []
for cond in cpd.conditions():
    
    prob_s1 = cpd[cond].prob("phone")
    prob_s2 = cpd[cond].prob("product")
    
    prob_frac = prob_s1/prob_s2
    if prob_frac == 0:
        likelihood = 0
    else:
        likelihood = math.log(prob_frac,2)
        
    sense = "phone"
    if likelihood < 0:
        sense = "product"
    
    decision_list.append([cond,abs(likelihood), sense])

def sortDL(declist):
    declist.sort(key = lambda x:x[1], reverse = True)
    return declist

decision_listFinal = sortDL(decision_list)

##read test data
testing_list = []

with open(testing_data, 'r') as data:
    parse_test = BeautifulSoup(data, 'html.parser')

test_list = []
for instance in parse_test.find_all('instance'):
    sent = dict()
    sent ['id'] = instance['id']
    text = ""
    for s in instance.find_all('s'):
        text = text+ " "+ s.get_text()
        #processing sentences
        text = text.lower()
        text = re.sub(r'[Ll]ines?','line',text) #standardize plural lines
        text = re.sub(r'[\.\,\?\!\\\'\-\$\%\(\);"&]',' ',text)
        
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in stopwords.words('english')]
    sent['text'] = tokens
    test_list.append(sent)
 
answers = []
for item in test_list:
    rule_no = 0
    for rule in decision_listFinal:
        rule_no += 1 #count rule
        #print("DL Reached Rule ", rule_no)
        position, loc, word = rule[0].split("_") #parse n, location, and specific word
        
        test_word = get_positional_n(int(position), item['text']) #get word as same position as rule is testing
        found = False #flag for matching rule
        if test_word == word:
            found = True #mark rule as found
            #print(test_word, word, rule_no)
            #print("DL Reached Rule ", rule_no) 
            sense = rule[2]
            temp = '<answer instance="%s" senseid="%s"/>'%(item['id'], sense)
            answers.append(temp)
            print('<answer instance="%s" senseid="%s"/>'%(item['id'], sense))
            print("Referencing rule number", rule_no, "on word", test_word,"\n")
            break
            
    if found == False:
        sense = majority
        temp = ['<answer instance="%s" senseid="%s"/>'%(item['id'], sense)]
        answers.append(temp)
        print('<answer instance="%s" senseid="%s"/>'%(item['id'], sense))
        print("Referencing default rule\n")
            
##make decisionlist text file
with open(final_decision_list,'w') as out:
    for rule in decision_listFinal:
        out.write('%s\n' %(rule))
        
##write answer file
with open("DL_answers.txt",'w') as out:
    for ans in answers:
        out.write('%s\n'%ans)