#!/usr/bin/env python
# coding: utf-8

import json
import io
import time
import codecs
from collections import defaultdict
import numpy as np
import spacy
from absl import logging
import tensorflow_hub as hub
import tensorflow as tf
import os
import pytextrank
import textacy
import re


stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]


# Organizing the contract into a data structure
"""
Returns the SHA in an array.
"""
def import_text(filename):
    with open(filename, 'r') as f:
        all_lines = f.readlines()
    all_lines = [item.strip() for item in all_lines if item.strip()]
    
    return all_lines


"""
Returns if the sent entered is a title
"""
def sent_is_title(sent, titles):
    sent = re.sub(r'[^a-zA-Z ]+', '', sent)
    for title in titles:
        title = re.sub(r'[^a-zA-Z ]+', '', title)
        if ((title.lower() in sent.lower()) or (sent.lower() in title.lower())):
            if (len(sent) / len(title) < 2) or (len(sent.split()) / len(title.split()) < 2):
                return True
    return False


"""
Returns if the sent entered is a definition
"""
def sent_is_definition(sent, definitions):
    for definition in definitions:
        if (definition.lower() in sent.lower()) and '“' in sent and '”' in sent:
            if 'shall have the meaning set forth' in sent.lower() or 'shall mean' in sent.lower() or 'means' in sent.lower() or 'shall have the meaning' in sent.lower():
                return True
    return False


def embed_message(s, embed):
    s_embeddings = embed(s)['outputs']
    return s_embeddings


def split_string(s, level='normal'):
    if level == 'normal':
        s = s.replace('. ', '@#$')
    else:
        s = s.replace('. ', '@#$').replace('(a) ', '@#$').replace('(b) ', '@#$').replace('(c) ',
                    '@#$').replace('(d) ', '@#$').replace('(A) ', '@#$').replace('(B) ', '@#$').replace('(C) ',
                    '@#$').replace('(D) ', '@#$').replace('\t', '@#$').replace('\xa0', ' ')
    
    return s.split('@#$')


"""
For each section, it returns its sentences split by
    . |(a) |(b) |(c) |(d) |(A) |(B) |(C) |(D)
Also returns their corresponding tags in a separate list.
"""
def return_split_sents(sents, level='normal'):
    split_sents = sents[:]
    for i, sent in enumerate(sents):
        split_sents[i] = split_string(sent, level)
    
    return split_sents


"""
Returns all the keyphrases of the document
"""
def get_keyphrases(sents, nlp):
    text = '\n'.join(sents)
    doc = nlp(text)
    
    return [p.text for p in doc._.phrases]


"""
Returns two lists: noun-phrases (or noun chunks), verb-phrases.
"""
def extract_parses(sents, nlp):
    sents1 = sents[:]
    sents2 = sents[:]
    for i, sent in enumerate(sents):
        doc = nlp(sent)
        verb_phrases = textacy.extract.matches(doc, "POS:VERB:? POS:ADV:* POS:VERB:+")
        sents1[i] = [str(chunk).strip() for chunk in doc.noun_chunks]
        sents2[i] = [str(item).strip() for item in verb_phrases]
    assert len(sents1) == len(sents2)
    assert len(sents) == len(sents2)
    
    return sents1, sents2



def get_phrases(phrases, sents):
    sents_phrases = []
    for line in sents:
        sents_phrases.append([p for p in phrases if p in line])
    assert len(sents_phrases) == len(sents)
    
    return sents_phrases


def embed_chunks(sents1, embed):
    start_time = time.time()
    sents1_embeddings = []
    for sent in sents1:
        e = embed(sent)['outputs']
        if np.shape(e)[0] == 0:
            sents1_embeddings.append(np.zeros((1, np.shape(e)[1])))
        else:
            sents1_embeddings.append(e)
    print('Embedded after: ', time.time() - start_time)
    return sents1_embeddings




class Query:
    def __init__(self, query, embed):
        
        last_time = time.time()
        nlp = spacy.load("en_core_web_sm")
        tr = pytextrank.TextRank()
        nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)
        print("Created query after")
        print("--- %.2f seconds ---" % (time.time() - last_time))
        
        doc = nlp(query)
        
        self.verb_phrases = [item for item in textacy.extract.matches(doc, "POS:VERB:? POS:ADV:* POS:VERB:+")]
        self.q_full = embed_message([query], embed)
        self.q_noun_chunks = embed_message([str(chunk).strip() for chunk in doc.noun_chunks], embed)
        self.q_verb_phrases = []
        if self.verb_phrases:
            self.q_verb_phrases = embed_message([str(item).strip() for item in self.verb_phrases], embed)
        else:
            self.q_verb_phrases = np.zeros((1, 512))
        self.q_words = embed_message([w.strip().lower() for w in query.split() if w.strip().lower() not in stop_words], embed)


    
def closest_paragraph_parsed(query, d, sents, titles, embed):
    a1, a4, a5, a7, b2, b4, b7, c5, c7, d2, d4 = [17, 2, 5, 6, 3, 7, 8, 6, 4, 10, 6]
    
    last_time = time.time()
    
    
    nlp = spacy.load("en_core_web_sm")
    tr = pytextrank.TextRank()
    nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)
    doc = nlp(query)
    
    verb_phrases = [item for item in textacy.extract.matches(doc, "POS:VERB:? POS:ADV:* POS:VERB:+")]
    q_full = embed_message([query], embed)
    q_noun_chunks = embed_message([str(chunk).strip() for chunk in doc.noun_chunks], embed)
    q_verb_phrases = []
    if verb_phrases:
        q_verb_phrases = embed_message([str(item).strip() for item in verb_phrases], embed)
    else:
        q_verb_phrases = np.zeros((1, 512))
    q_words = embed_message([w.strip().lower() for w in query.split() if w.strip().lower() not in stop_words], embed)
    
    
    print("Created query after")
    print("--- %.2f seconds ---" % (time.time() - last_time))
    
    
    last_time = time.time()
    
    res = []

    for i in range(len(sents)):
        s = d[i]
        
        s11 = a1 * np.inner(s.sent_e, q_full) ### 18 choose one only (for sure remove a3)
        s12 = 1 * np.max(np.inner(s.split_sent_e, q_full), initial=0) ###
        s13 = 1 * np.max(np.inner(s.ultra_split_sent_e, q_full), initial=0) ###
        s14 = a4 * np.max(np.inner(s.sent_np_e, q_full), initial=0) ### 3
        s15 = a5 * np.max(np.inner(s.sent_vp_e, q_full), initial=0) ### 6
        s16 = 1 * np.max(np.inner(s.sent_defs_e, q_full), initial=0)
        s17 = a7 * np.max(np.inner(s.sent_phrases_e, q_full), initial=0) ### 5
        
        s21 = 1 * np.mean(np.inner(s.sent_e, q_noun_chunks)) # doesn't help
        s22 = b2 * np.mean(np.max(np.inner(s.split_sent_e, q_noun_chunks), axis=0, initial=0)) ### 4 # could change to np.max on the outside
        s23 = 1 * np.mean(np.max(np.inner(s.ultra_split_sent_e, q_noun_chunks), axis=0, initial=0)) # could change to np.max on the outside
        s24 = b4 * np.mean(np.max(np.inner(s.sent_np_e, q_noun_chunks), axis=0, initial=0)) ### 8
        s25 = 1 * np.mean(np.max(np.inner(s.sent_vp_e, q_noun_chunks), axis=0, initial=0))
        s26 = 1 * np.max(np.inner(s.sent_defs_e, q_noun_chunks), initial=0)
        s27 = b7 * np.max(np.inner(s.sent_phrases_e, q_noun_chunks), initial=0) ### 7
        
        s31 = 1 * np.mean(np.inner(s.sent_e, q_verb_phrases)) # could change to np.max
        s32 = 1 * np.mean(np.max(np.inner(s.split_sent_e, q_verb_phrases), axis=0, initial=0))
        s33 = 1 * np.mean(np.max(np.inner(s.ultra_split_sent_e, q_verb_phrases), axis=0, initial=0))
        s34 = 1 * np.mean(np.max(np.inner(s.sent_np_e, q_verb_phrases), axis=0, initial=0))
        s35 = c5 * np.mean(np.max(np.inner(s.sent_vp_e, q_verb_phrases), axis=0, initial=0)) ### 7
        s36 = 1 * np.max(np.inner(s.sent_defs_e, q_verb_phrases), initial=0)
        s37 = c7 * np.max(np.inner(s.sent_phrases_e, q_verb_phrases), initial=0) ### 5
        
        s41 = 1 * np.mean(np.inner(s.sent_e, q_words))
        s42 = d2 * np.max(np.mean(np.inner(s.split_sent_e, q_words), axis=1), initial=0) ### 9
        s43 = 1 * np.max(np.mean(np.inner(s.ultra_split_sent_e, q_words), axis=1), initial=0)
        s44 = d4 * np.max(np.mean(np.inner(s.sent_np_e, q_words), axis=1), initial=0) ### 7
        s45 = 1 * np.max(np.mean(np.inner(s.sent_vp_e, q_words), axis=1), initial=0)
        s46 = 1 * np.max(np.mean(np.inner(s.sent_defs_e, q_words), axis=1), initial=0)
        s47 = 1 * np.max(np.mean(np.inner(s.sent_phrases_e, q_words), axis=1), initial=0)
        
        score = s11 + s12 + s13 + s14 + s15 + s16 + s17 + s21 + s22 + s23 + s24 + s25 + s26 + s27 + s31 + s32 + s33 + s34 + s35 + s36 + s37 + s41 + s42 + s43 + s44 + s45 + s46 + s47
        
        if not sent_is_title(sents[i], titles) and len(sents[i].split()) > 4:
            res.append((sents[i], score))


    print("Finished multiplying vectors after")
    print("--- %.2f seconds ---" % (time.time() - last_time))

    last_time = time.time()
    res = sorted(res, key=lambda x : x[1], reverse=True)
    print("Finished sorting results after")
    print("--- %.2f seconds ---" % (time.time() - last_time))

    return res


def final_results(query, d, sents, titles, embed, n=10):
    res = closest_paragraph_parsed(query, d, sents, titles, embed)
    return [item[0] for item in res[:n]]

