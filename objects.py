#!/usr/bin/env python
# coding: utf-8

import io
import os
import spacy
import pytextrank
import textacy
import tensorflow_hub as hub
import tensorflow as tf
import time
from absl import logging
from unsupervised_lookup import embed_message, return_split_sents, get_keyphrases, get_phrases, embed_chunks, extract_parses


def return_embedder():
    last_time = time.time()
    #try:
    #module_url = "/4/"
    #embed = hub.load(module_url)
    #except:
    module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/4"
    embed = hub.load(module_url)
    #embed = tf.saved_model.load(module_url)
    print("Loaded TFHUB module after")
    print("--- %.2f seconds ---" % (time.time() - last_time))
    last_time = time.time()
    # Reduce logging output.
    logging.set_verbosity(logging.ERROR)
    return embed


# basically a list of Line objects
class Document:
    def __init__(self, sents, titles, defs, embed):
        self.sents = sents
        self.defs = defs
        self.titles = titles
        
        self.embed = embed
        #self.embed = return_embedder()
        self.ultra_split_sents = return_split_sents(self.sents, 'ultra')
        self.split_sents = return_split_sents(self.sents)
        
        last_time = time.time()
        print("Embedding")
        self.sents_embeddings = self.embed(self.sents)['outputs']
        print("--- %.2f seconds ---" % (time.time() - last_time))
        
        self.nlp = spacy.load("en_core_web_sm")
        # add PyTextRank to the spaCy pipeline
        tr = pytextrank.TextRank()
        self.nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)
        
        self.phrases = get_keyphrases(self.sents, self.nlp)
        self.sents_phrases = get_phrases(self.phrases, self.sents)
        self.sents_defs = get_phrases(self.defs, self.sents)
        self.sents_phrases_embedding = embed_chunks(self.sents_phrases, self.embed)
        self.sents_defs_embedding = embed_chunks(self.sents_defs, self.embed)
        self.sents_np, self.sents_vp = extract_parses(self.sents, self.nlp)
        self.sents_np_embedding = embed_chunks(self.sents_np, self.embed)
        self.sents_vp_embedding = embed_chunks(self.sents_vp, self.embed)
        self.split_sents_embeddings = embed_chunks(self.split_sents, self.embed)
        self.ultra_split_sents_embeddings = embed_chunks(self.ultra_split_sents, self.embed)

        self.lines = [Line(sent_e=self.sents_embeddings[i],
                           split_sent_e=self.split_sents_embeddings[i],
                           ultra_split_sent_e=self.ultra_split_sents_embeddings[i],
                           sent_np_e=self.sents_np_embedding[i],
                           sent_vp_e=self.sents_vp_embedding[i],
                           sent_defs_e=self.sents_defs_embedding[i],
                           sent_phrases_e=self.sents_phrases_embedding[i],
                           idx=i) for i in range(len(sents))]


    def __len__(self):
        assert len(self.lines) == len(self.sents)
        return len(self.lines)

    def __getitem__(self, i):
        return self.lines[i]


class Line:
    def __init__(self, sent_e, split_sent_e, ultra_split_sent_e,
                 sent_np_e, sent_vp_e, sent_defs_e, sent_phrases_e, idx):
        self.sent_e = sent_e
        self.split_sent_e = split_sent_e
        self.ultra_split_sent_e = ultra_split_sent_e
        self.sent_np_e = sent_np_e
        self.sent_vp_e = sent_vp_e
        self.sent_defs_e = sent_defs_e
        self.sent_phrases_e = sent_phrases_e
        self.idx = idx











