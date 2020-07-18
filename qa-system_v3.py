# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 11:36:07 2020
@author: eachr
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

#log_file = sys.argv[1]
# =============================================================================
# logging.basicConfig(filename=log_file, format=log_format,
#                     datefmt='%Y-%m-%d %H:%M:%S')
# =============================================================================

logging.basicConfig(filename="mylogfile.txt", format=log_format,
                    datefmt='%Y-%m-%d %H:%M:%S')

logger.info("=====QA-System Start==========")


re_questionWords = r'([Ww]h(at|ere|en|o))'
re_verbs = r'(is|are|was|were|has|have|did|does)'
re_topic = r'(.*)'

punc = string.punctuation 

nouns = ['NN','NNP','NNPS']
verbs = ['VB','VBD','VBG','VBN','VBP','VBZ']

class NoStdStreams(object):
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
    user_query = input("[User] > ")
    return user_query


def searchWiki(topic, pageRank = 0):
    try:
        results = wiki.search(topic)
    except:
        results = False
    
    if results == []:
        results = False
    return results[pageRank]

def extractWikiContent(page):
        
    try:
       with NoStdStreams():
           pagetext = wiki.page(page, auto_suggest = False).summary #take first page result
           pagetext = re.sub(r'\w\.(\w\.)+',' ', pagetext)
           pagetext = re.sub(r'\s+',' ', pagetext) #remove extra spacing
        
        #pagetext = re.sub('\([^)]*\)',"",pagetext)
    except wiki.DisambiguationError as e:
        with NoStdStreams():
            pagetext = wiki.page(e.options[0], auto_suggest = False).summary 
            pagetext = re.sub(r'\w\.(\w\.)+',' ', pagetext)
            pagetext = re.sub(r'\s+',' ', pagetext) #remove extra spacing
    
    return pagetext

def BirthDeathFlagger(topic):
        birth_pattern = r'(born|birth(day)?)'
        birth_flag = False
        
        death_pattern = r'(die(d)?|death|passed away)'
        death_flag = False
        
        if re.search(birth_pattern, topic, re.IGNORECASE) is not None: #question pertains specifically to birth/death days
            return 'BIRTH'
        elif re.search(death_pattern, topic, re.IGNORECASE) is not None: #question pertains specifically to birth/death days
            return 'DEATH'
        else: 
            return ''
        


def answerQuestion(user_input):
    
    parseQ = re.search( " ".join([re_questionWords,re_verbs,re_topic]), user_input)
    
    if parseQ is None:
        print("This question is not a valid question type for this system")
        return 
    
    qWord = parseQ[1] #W-word question
    verb = parseQ[3] #follow up verb
    topic = parseQ[4] #rest of the question
    
    topic_tokens = word_tokenize(topic) #tokenize topic
    topic_tags = nltk.pos_tag(topic_tokens) #POS tag topic
    
    topic_nouns = " ".join([tag[0] for tag in topic_tags if tag[1] in nouns])
    verb_tokens = " ".join([tag[0] for tag in topic_tags if tag[1] in verbs])
    
    proper_tokens = [tag[0] for tag in topic_tags if tag[1] in ['NNP', 'NNPS']]
    topic_proper_nouns = " ".join(proper_tokens)
    #v_syns = [lemma.name() for syn in wordnet.synsets(verb) for lemma in syn.lemmas()]
    
    if qWord.lower() == "who": #NER case
        
        id_verbs = "(is|are|was|were)"
        
        
        prop_pattern = '(' + "(.* )".join(proper_tokens) + ')'#allow for middle names and nicknames between proper names
        match_pattern = prop_pattern + '( .* )*' + id_verbs + '(.*)\.'
        
        
        result = searchWiki(topic_proper_nouns)
        
        try:
            if result:
                pagetext = extractWikiContent(result)
                if pagetext:
                    sent_toks = sent_tokenize(pagetext)
                    
                    candidates = []
                    for sent in sent_toks:
                        matched = re.search(match_pattern, sent)
                        if matched is not None:
                            candidates.append([matched[1],matched[4],matched[5]])
                    
                    if candidates == []:
                        ans = "Sorry I was not able to find any information relating to this topic"
                    else:
                        ans = candidates[0]
                    
                    ans =" ".join(ans) + '.'
                    
                    logging.info("WHO-Answer is: "+ ans)
                else:
                    ans = "Sorry your question was not specific enough, please provide more information"
            else:
                ans = "Sorry I was not able to find any information relating to this topic"
        except:
            ans = "Sorry, that question was not specific enough for me to answer"
    
        
    elif qWord.lower() == "where":
        
        match_patterns = '(.*) (locat(ed|ion)|(are|is) found|occurred at|is in) (.*)'
        result = searchWiki(topic_nouns)
        if result:
            pagetext = extractWikiContent(result)
            if pagetext:
                sent_toks = sent_tokenize(pagetext)
                
                candidates = []
                for sent in sent_toks:
                    if re.search(match_patterns, sent, re.IGNORECASE) is not None:                
                       candidates.append(sent) 
                
                if candidates == []:
                    ans = "Sorry I was not able to find any information relating to this topic"
                else:
                    ans = candidates[0]
                    
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
                ans = re.sub('\([^)]*\)',"",sent_toks[0])
                #ans = re.sub(', and .*','\.',ans)
                logging.info("WHAT-Answer is: "+ ans)
            else:
                ans = "Sorry your question was not specific enough, please provide more information"
        else:
            ans = "Sorry I was not able to find any information relating to this topic"
        
        
            
        
    elif qWord.lower() == "when":
        
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