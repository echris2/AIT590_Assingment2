# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 14:42:25 2020

@author: eachr
"""

import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import pandas as pd
import re
import sys


#DLanswers_txt = sys.argv[1] #model answers
#answers_txt = sys.argv[2] #gold standard answers

answers_txt = "line-data/line-answers.txt"
DLanswers_txt = "DL_answers.txt"

#open the two files taken as input from command line and strip out '\n'
key_answers = {} #init dictionary to hold gold standard answers
with open(answers_txt, 'r') as key:
    for line in key:
        parse = re.search('<answer instance="(.*)" senseid="(.*)"/>', line) #parse id and sense
        key_answers[parse[1]] = parse[2] #add sense and id to dictionary
    
gen_answers = {} #init dictionary to hold predicted answers
with open(DLanswers_txt, 'r') as preds:
    for line in preds:
        parse = re.search('<answer instance="(.*)" senseid="(.*)"/>', line)
        gen_answers[parse[1]] = parse[2]


predDF = pd.DataFrame(gen_answers.items(), columns = ['id', 'prediction'])
actDF = pd.DataFrame(key_answers.items(),columns = ['id', 'actual'])

correct = 0
incorrect = 0
total = 0
for key in key_answers.keys():
    total +=1
    
    if (gen_answers[key] == key_answers[key]):
        correct +=1
    elif (gen_answers[key] != key_answers[key]):
        incorrect +=1

acc = round(correct/total*100,2)
print("\nThe accuracy of the Decision List on the given test data is", acc, "%\n")
df = pd.merge(predDF,actDF, on = 'id') 

confMat = pd.crosstab(df.prediction,df.actual)
print(confMat)

