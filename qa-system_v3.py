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


from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import wordnet
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.chunk import tree2conlltags
from nltk.stem import PorterStemmer

#log_file = sys.argv[1]
#lf = open(log_file)

re_questionWords = r'([Ww]h(at|ere|en|o))'
re_verbs = r'(is|are|was|were|has|have|did|does)'
re_topic = r'(.*)'

punc = string.punctuation 

nouns = ['NN','NNP','NNPS']
verbs = ['VB','VBD','VBG','VBN','VBP','VBZ']

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

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

def extractWikiContent(page,sent_no = 0):
        
    try:
       with HiddenPrints():
           pagetext = wiki.page(page, auto_suggest = False).content #take first page result
           pagetext = re.sub(r'\w\.(\w\.)+',' ', pagetext)
           pagetext = re.sub(r'\s+',' ', pagetext) #remove extra spacing
        
        #pagetext = re.sub('\([^)]*\)',"",pagetext)
    except:
        pagetext = False
      
    return pagetext

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
    
    #v_syns = [lemma.name() for syn in wordnet.synsets(verb) for lemma in syn.lemmas()]
    
    if qWord.lower() == "who": #NER case
        
        id_verbs = "(is|are|was|were)"
        proper_tokens = [tag[0] for tag in topic_tags if tag[1] in ['NNP', 'NNPS']]
        
        prop_pattern = '(' + "(.* )".join(proper_tokens) + ')'#allow for middle names and nicknames between proper names
        match_pattern = prop_pattern + '( .* )*' + id_verbs + '(.*).'
        
        topic_proper_nouns = " ".join(proper_tokens)
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
                    ans = candidates[0]
                    ans =" ".join(ans) + '\.'
                else:
                    ans = "Sorry your question was not specifc enough, please provide more information"
            else:
                ans = "Sorry I was not able to find any information relating to this topic"
        except:
            ans = "Sorry, that question was not specific enough for me to answer"
            
    elif qWord.lower() == "where":
        
        match_patterns = '(.*) (locat(ed|ion)|address|found|occurred at|in) (.*)'
        result = searchWiki(topic_nouns)
        if result:
            pagetext = extractWikiContent(result)
            if pagetext:
                sent_toks = sent_tokenize(pagetext)
                
                candidates = []
                for sent in sent_toks:
                    if re.search(match_patterns, sent, re.IGNORECASE) is not None:                
                       candidates.append(sent) 
                
                ans = candidates[0]    
            else:
                ans = "Sorry your question was not specifc enough, please provide more information"
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
            else:
                ans = "Sorry your question was not specifc enough, please provide more information"
        else:
            ans = "Sorry I was not able to find any information relating to this topic"
        
            
            
        
    elif qWord.lower() == "when":
        
        match_date_patterns = r'''(((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w* \d{1,2}(, \d{2,4})?)|
        (\d{1,2}/\d{1,2}/\d{2,4})|(\d{4} to \d{4}))'''
        match_text_patterns = r"(on|of|during)"
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
                
                
                event_date = re.search(match_date_patterns, candidates[0])     
                ans = topic_nouns.capitalize() + "occurred on " + event_date
                
            else:
                ans = "Sorry your question was not specifc enough, please provide more information"
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
            answer = answerQuestion(query)
            answer = re.sub("\s+", " ",answer)
            print(answer)

if __name__ == "__main__":
    main()

    
    