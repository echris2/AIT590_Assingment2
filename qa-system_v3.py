# -*- coding: utf-8 -*-
"""
# ### Programming Assignment 4: Question Answering
#
# <br>
# **Team 3:**
# <br>
# Srikanth Vadlamani, Jason Drahos, Eric Christiansen, Stephen Andre
# Date 20 July, 2020
# Course: AIT-5990-B01
# 1. Introduction to the project: The assignment is to a simple question answering system,
                                    capabale of answering Who, What, When, Where questions
#
#2.  To begin the program the user enters the following command into the command line
#    py qa-system.py n 
#           Where n = blank text file in which to store the log
#3.   Example
#       Input:
#           py qa-system.py mylogfile.txt
#

            
#
#4. Basic functionality:
            
            Welcomes user
            Prompts user query
            Confirms question is Who, what, when, where
            Breaks query into quesiton word, and topic
            Search Wikipedia for topic
            Pull summary for top ranked page
            Search summary for question-word-specific pattern
            Extract pattern and return answer
            Loop with query prompt until user enters sentinel word
            
            
# Citation:
#       Sainikhithamadduri. (2019, June 6). Sainikhithamadduri/Wordsense-Disambiguation-. 
            Retrieved July 08, 2020, from https://github.com/sainikhithamadduri/Question-Answering-System-NLP
#       
#      Orozco, F. (2020, July 15). Suppress Print Output in Python. 
            Retrieved July 20, 2020, from https://codingdose.info/2018/03/22/supress-print-output-in-python/
            
        Zola, A. (n.d.). Simple Question Answering (QA) Systems That Use Text Similarity Detection in Python. 
            Retrieved July 20, 2020, from https://www.kdnuggets.com/2020/04/simple-question-answering-systems-text-similarity-python.html
"""

import sys
import os
import re
import nltk
import wikipedia as wiki
import string

from contextlib import contextmanager


from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import wordnet
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.chunk import tree2conlltags
from nltk.stem import PorterStemmer

import time
import logging

#log_file = sys.argv[1]

start_time = time.localtime()

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

log_format = '%(asctime)s %(filename)s: %(message)s'

log_file = sys.argv[1]
logging.basicConfig(filename=log_file, format=log_format,
                    datefmt='%Y-%m-%d %H:%M:%S')

# =============================================================================
# logging.basicConfig(filename="mylogfile.txt", format=log_format,
#                     datefmt='%Y-%m-%d %H:%M:%S')
# =============================================================================

logger.info("=====QA-System Start==========")


re_questionWords = r'([Ww]h(at|ere|en|o))' #regex to identify question word in query
re_verbs = r'(is|are|was|were|has|have|did|does)' #regex to identify verb following question word
re_topic = r'(.*)' #the rest of the question, ie the topic

punc = string.punctuation  

nouns = ['NN','NNP','NNPS'] #noun tags
verbs = ['VB','VBD','VBG','VBN','VBP','VBZ'] #verb tags

class NoStdStreams(object):
    '''
    Class to suppress console output, useful for wikipedia package which has 
    a pre-printed warning

    '''
    def __init__(self,stdout = None, stderr = None):
        self.devnull = open(os.devnull,'w')
        self._stdout = stdout or self.devnull or sys.stdout
        self._stderr = stderr or self.devnull or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush(); self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush(); self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.devnull.close()

def get_input():
    '''
    Returns
    -------
    user_query : TYPE string
        DESCRIPTION. User input

    '''
    user_query = input("[User] > ")
    return user_query


def searchWiki(topic, pageRank = 0):
    '''
    

    Parameters
    ----------
    topic : TYPE user input from get_input function
        DESCRIPTION.
    pageRank : TYPE, integer optional
        DESCRIPTION. The default is 0. search result ranking
    Returns
    -------
    TYPE
        DESCRIPTION. Returns user-specified ranked search result

    '''
    try:
        results = wiki.search(topic) #search topic
    except:
        results = False #if error, return false
    
    if results == []:
        results = False #if no results return false
    return results[pageRank]

def extractWikiContent(page):
    '''
    

    Parameters
    ----------
    page : TYPE 
        DESCRIPTION. output from searchWiki() function

    Returns
    -------
    pagetext : TYPE
        DESCRIPTION.

    '''
        
    try:
       with NoStdStreams():
           pagetext = wiki.page(page, auto_suggest = False).summary #take first page result summary
           pagetext = re.sub(r'\w\.(\w\.)+',' ', pagetext) #remove initials peridos for better sentence tokenizing
           pagetext = re.sub(r'\s+',' ', pagetext) #remove extra spacing
        
        #pagetext = re.sub('\([^)]*\)',"",pagetext)
    except wiki.DisambiguationError as e: #Error handling specifically take top ranked result
        with NoStdStreams():
            pagetext = wiki.page(e.options[0], auto_suggest = False).summary  
            pagetext = re.sub(r'\w\.(\w\.)+',' ', pagetext)
            pagetext = re.sub(r'\s+',' ', pagetext) #remove extra spacing
    
    return pagetext

def BirthDeathFlagger(topic):
    '''
    Flag whether question is asking for birthday or death date

    Parameters
    ----------
    topic : TYPE 
        DESCRIPTION. user_input topic

    Returns
    -------
    str
        DESCRIPTION. FLAG

    '''
    birth_pattern = r'(born|birth(day)?)'
    birth_flag = False
    
    death_pattern = r'(die(d)?|death|passed away|killed)'
    death_flag = False
    
    if re.search(birth_pattern, topic, re.IGNORECASE) is not None: #question pertains specifically to birth/death days
        return 'BIRTH'
    elif re.search(death_pattern, topic, re.IGNORECASE) is not None: #question pertains specifically to birth/death days
        return 'DEATH'
    else: 
        return ''
        


def answerQuestion(user_input):
    '''
    Parameters
    ----------
    user_input : TYPE user input string
        DESCRIPTION.

    Returns
    -------
    ans 
        DESCRIPTION. The extracted information to answer user query

    '''
    
    parseQ = re.search( " ".join([re_questionWords,re_verbs,re_topic]), user_input) #parse user_input
    
    if parseQ is None:
        print("This question is not a valid question type for this system")
        return 
    
    qWord = parseQ[1] #W-word question
    verb = parseQ[3] #follow up verb
    topic = parseQ[4].replace('?','') #rest of the question
    
    topic_tokens = word_tokenize(topic) #tokenize topic
    topic_tags = nltk.pos_tag(topic_tokens) #POS tag topic
    
    topic_nouns = " ".join([tag[0] for tag in topic_tags if tag[1] in nouns]) #string together nouns in input
    verb_tokens = " ".join([tag[0] for tag in topic_tags if tag[1] in verbs]) #string together verbs in input
    proper_tokens = [tag[0] for tag in topic_tags if tag[1] in ['NNP', 'NNPS']] #take only proper nouns
    topic_proper_nouns = " ".join(proper_tokens) #string together proper tokens
    
    if qWord.lower() == "who": #if question is who
        
        id_verbs = "(is|are|was|were)"
        
        
        prop_pattern = '(' + "(.* )".join(proper_tokens) + ')'#allow for middle names and nicknames between proper names
        match_pattern = prop_pattern + '( .* )*' + id_verbs + '(.*)\.' #Search for name, id verb, text pattern
        
        
        result = searchWiki(topic_proper_nouns) #search
        
        try:
            if result: #if results, take content
                pagetext = extractWikiContent(result)
                if pagetext: #if contnent, tokenize sentences
                    sent_toks = sent_tokenize(pagetext)
                    candidates = [] #initatlize sentence storage
                    for sent in sent_toks: #find sentences with matching pattern
                        matched = re.search(match_pattern, sent)
                        if matched is not None: #if match is found append name pattern, id verb and rest of of text
                            candidates.append([matched[1],matched[4],matched[5]])
                    
                    if candidates == []: #if no sentences relate
                        ans = "Sorry I was not able to find any information relating to this topic"
                    else:
                        ans = candidates[0] #take top sentence
                        ans =" ".join(ans) + '.' #join together pattern matches
                    
                    logging.info("WHO-Answer is: "+ ans)
                else: #if no page is found
                    ans = "Sorry your question was not specific enough, please provide more information"
            else: #if no resutls are found
                ans = "Sorry I was not able to find any information relating to this topic"
        except: #if error along the way
            ans = "Sorry, that question was not specific enough for me to answer"
    
        
    elif qWord.lower() == "where":
        
        match_patterns = '(.*) ((are|is)? located|(are|is) found|occurred at) (.*)' #find basic occurence patterns
        result = searchWiki(topic_nouns) #search
        if result:
            pagetext = extractWikiContent(result) #pull page
            if pagetext:
                sent_toks = sent_tokenize(pagetext) #tokenize sentences
                
                candidates = []
                for sent in sent_toks:
                    if re.search(match_patterns, sent, re.IGNORECASE) is not None:                
                       candidates.append(sent) 
                
                if candidates == []:
                    try:
                        first_sent = sent_toks[0].replace('.','')
                        sec_match = re.search(r'.* in (.*,.*)', first_sent)
                        ans = topic.capitalize() + " is in " + sec_match[1] + "."
                    except:
                        ans = "Sorry I was not able to find any information relating to this topic"
                else:
                    ans = candidates[0]
                    ans = re.sub(', and .*','.',ans) #remove extra information
                    match = re.search(match_patterns,ans)
                    
                    ans = topic + " " + match[2] + " " + match[5]
# =============================================================================
#                     matches = re.search(match_patterns, ans, re.IGNORECASE)
#                     loc_tags = nltk.pos_tag(matches [5])
#                     loc = " ".join([tag[0] for tag in loc_tags if tag[1] in ['NNP', 'NNPS']])
# =============================================================================
                    
                logging.info("WHERE-Answer is: "+ ans)
            else:
                ans = "Sorry your question was not specific enough, please provide more information"
        else:
            ans = "Sorry I was not able to find any information relating to this topic"
        
    
    elif qWord.lower() == "what":
        match_patterns = ""
        
        result = searchWiki(topic_nouns)
        if result:
            pagetext = extractWikiContent(result)
            if pagetext:
                sent_toks = sent_tokenize(pagetext)
                ans = re.sub('\([^)]*\)',"",sent_toks[0]) #remove any items in parentheses
                #ans = re.sub(', and .*','\.',ans)
                logging.info("WHAT-Answer is: "+ ans)
            else:
                ans = "Sorry your question was not specific enough, please provide more information"
        else:
            ans = "Sorry I was not able to find any information relating to this topic"
        
        
            
        
    elif qWord.lower() == "when":
        
        #regex for several different date patterns
        match_date_patterns = r'''(((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w* \d{1,2}(–\d{1,2})?(, \d{2,4})?)|(\d{1,2}\d{1,2}\d{2,4})|(\d{4} to \d{4}))'''
        match_text_patterns = r"(on|of|during)"
        
        birth_death_flag = BirthDeathFlagger(topic)
            
        result = searchWiki(topic_nouns)
        if result:
            pagetext = extractWikiContent(result)
            if pagetext:
                                         
                sent_toks = sent_tokenize(pagetext)
                candidates = []
                for sent in sent_toks:
                   if re.search(match_text_patterns,sent, re.IGNORECASE) is not None:
                       if re.search(match_date_patterns,sent, re.IGNORECASE) is not None:
                           candidates.append(sent) 
                 
                ans = candidates[0] 
                event_date = re.search(match_date_patterns,ans )  
                if len(event_date.group()) < 5:
                    ans = topic_nouns + " occurred in " + event_date.group()  + '.'        
                else:
                    ans = topic_nouns + " occurred on " + event_date.group()  + '.' 
                   
                if birth_death_flag != '':
                    
                   for candidate in candidates:
                       if birth_death_flag == 'BIRTH':
                           bdate = re.search('born', candidate)
                           if bdate is not None:
                               event_date = re.search(match_date_patterns,candidate)
                               if len(event_date.group()) < 5:
                                   ans = topic_proper_nouns + " was born in " + event_date.group() +'.'
                               else:
                                   ans = topic_proper_nouns + " was born on " + event_date.group() +'.'
                               break
                           else:
                               ans = "I could not find any information relating to the birth of " + topic_proper_nouns +'.'
                       if birth_death_flag == 'DEATH':
                           ddate = re.search(r'(die(d)?|passed away|killed)', candidate)
                           if ddate is not None:
                               event_date = re.search(match_date_patterns,candidate)
                               if len(event_date.group()) < 5:
                                   ans = topic_proper_nouns + " passed away in " + event_date.group()  + '.'
                               else:
                                   ans = topic_proper_nouns + " passed away on " + event_date.group()  + '.'
                               break
                           else:
                               ans = "I could not find any information relating to the death of " + topic_proper_nouns +'.'
                       
                
                
                logging.info("WHEN-Answer is: "+ ans)
            else:
                ans = "Sorry your question was not specific enough, please provide more information"
        else:
            ans = "Sorry I was not able to find any information relating to this topic"
        
    else:
        ans = "Please ask a Who/What/When/Where Question"
        
    return ans

def main():
    print("Hello! Welcome to our Question Answering system!\n")
    print("This system is designed to simple answer Who, What, When, and Where questions, so please ask accordingly.\n")
    print("Use 'quit' or 'exit' to leave the system")
    print("\nAre you ready to begin? Ask away!\n")
    
    sentinels = ['quit','exit']
    flag = False
    
    while flag == False:
        query = get_input()
        
        if query in sentinels:
            print("Thanks so much for your time")
            flag = True
        else:
            logging.info("Question was: "+ query)
            try:
                answer = answerQuestion(query)
                answer = re.sub("\s+", " ",answer)
                print(answer)
            except:
                print("Sorry I was not able to find any information relating to that query")

if __name__ == "__main__":
    main()