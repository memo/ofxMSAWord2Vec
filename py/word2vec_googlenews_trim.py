# coding: utf-8

"""
@author: memo

load the original 3.6GB google news model by Mikolov et al (https://code.google.com/archive/p/word2vec/)
limit to the words in wiktionary's top 100K words
remove all duplicate entries, including different capitalization variations
preserve vector for capitalization  of most common entry (e.g. Paris)
save two models, 1.) preserving capitalization and 2.) all words forced to lowercase

see README (github.com/memo/ofxMSAWord2Vec) for more info

"""

import os
from word2vec_utils import *

do_load_intermediate_files = True
max_words_to_load=-1
max_words_to_save=-1

data_path = './data'  # root of data path relative to python working folder            
top_words_path = os.path.join(data_path, 'txt/wikitop-100k_clean.txt')
orig_vocab_path = os.path.join(data_path, 'txt/GoogleNews_vocab.txt')

orig_vocab_lc_path = os.path.splitext(orig_vocab_path)[0] + '_lc.txt'
top_words_unique_path = os.path.splitext(top_words_path)[0] + '_unique.txt'
#top_words_unique_lc_path = os.path.splitext(top_words_path)[0] + '_unique_lowercase.txt'
reduced_vocab_path = os.path.splitext(orig_vocab_path)[0] + '_reduced.txt'
reduced_vocab_lc_path = os.path.splitext(orig_vocab_path)[0] + '_reduced_lowercase.txt'


def find_missing_words(orig_vocab, words):
    # words in wiki top 100 which aren't in GoogleNews (mainly foreign words and junk)
    print 'Finding missing words...',
    missing_words = list(set(words) - set(orig_vocab))
    print len(missing_words), 'words found'

    print 'Finding indices'    
    missing_indices = [words.index(x) for x in missing_words]
    
    missing_word_pairs = zip(missing_indices, missing_words)

    print 'Sorting...',
    missing_word_pairs = sorted(missing_word_pairs)
    print 'done.'
    return missing_word_pairs




def test():
    # load original google vectors
    vecg = load_word_vectors_bin(os.path.expanduser('~/Downloads/text/GoogleNews/GoogleNews-vectors-negative300.bin'), max_words_to_load)
"""
    orig_vocab          : original full vocabulary (3M) of word vectors, unsorted, full of duplicates etc
    orig_vocab_lc       : lower case version of above
    top_words           : raw wiki top 100 words, sorted by commonness
    top_words_unique    : unique version of above (duplicates removed)
    reduced_vocab       : intersection of top_words_unique and orig_vocab, sorted by commonness
    reduced_vocab_lc    : lowercase version of above
    
"""        
        
    # if the intermediate files do exist and we want to load them
    if do_load_intermediate_files:
        # keys (vocabulary) of orig words
        with open(orig_vocab_path, 'r') as f:
            orig_vocab = f.read().split('\n')
        
        # lower case vocabulary of orig words
        with open(orig_vocab_lc_path, 'r') as f:
            orig_vocab_lc = f.read().split('\n')
    
        # wiki top 100 words with duplicates and capitalization variations
        with open(top_words_path, 'r') as f:
            top_words = f.read().splitlines()

        # unique wiki top words (maintaining capilization variations)
        with open(top_words_unique_path, 'r') as f:
            top_words_unique = f.read().split('\n')
        
        # unique wiki top words to lower case
#        with open(top_words_unique_lc_path, 'r') as f:
#            top_words_unique_lc = f.read().split('\n')
        
        # reduce vocabulary with intersection with top words
        with open(reduced_vocab_path, 'r') as f:
            reduced_vocab = f.read().split('\n')
    
        # same as above but forcing everything lower case
        with open(reduced_vocab_lc_path, 'r') as f:
            reduced_vocab_lc = f.read().split('\n')
    
    
    # otherwise create them from scratch and save them
    else:
        # keys (vocabulary) of orig words
        orig_vocab = vecg.keys()
        with open(orig_vocab_path, 'w') as f:
            f.write('\n'.join(orig_vocab))
            
        # lower case vocabulary of orig words
        orig_vocab_lc = map(str.lower, orig_vocab)
        with open(orig_vocab_lc_path, 'w') as f:
            f.write('\n'.join(orig_vocab_lc))
    
        # wiki top 100 words with duplicates and capitalization variations
        with open(top_words_path, 'r') as f:
            top_words = f.read().splitlines()
            
        # unique wiki top words (maintaining capilization variations)
        top_words_unique = make_word_list_unique(top_words, ignore_case=True)
        with open(top_words_unique_path, 'w') as f:
            f.write('\n'.join(top_words_unique))
        
        # unique wiki top words to lower case
#        top_words_unique_lc = [x.lower() for x in top_words_unique]
#        top_words_unique_lc = make_word_list_unique(top_words_unique_lc)
#        with open(top_words_unique_lc_path, 'w') as f:
#            f.write('\n'.join(top_words_unique_lc))
        
        # reduce vocabulary with intersection with top words
        reduced_vocab = keep_words(top_words_unique, orig_vocab, ignore_case=True)
        reduced_vocab = make_word_list_unique(reduced_vocab, ignore_case=True)
        with open(reduced_vocab_path, 'w') as f:
            f.write('\n'.join(reduced_vocab))
    
        # same as above but forcing everything lower case
        reduced_vocab_lc = map(str.lower, reduced_vocab)
        reduced_vocab_lc = make_word_list_unique(reduced_vocab_lc, ignore_case=True)
        with open(reduced_vocab_lc_path, 'w') as f:
            f.write('\n'.join(reduced_vocab_lc))
    
        # the version of the vocabulary as found in the word_vectors
        # construct dictionary mapping original word to index
#        orig_vocab_dict = dict()
#        for i,x in enumerate(orig_vocab_lc):
#            orig_vocab_dict[x] = i
#        reduced_vocab_wv = [find_correct_case(x, orig_vocab, orig_vocab_lc, orig_vocab_dict) for x in reduced_vocab]
    
    missing_words_vocab = find_missing_words(orig_vocab, top_words_unique)
#    missing_words_lc = find_missing_words(orig_vocab, top_words_unique_lc)
            
    
    vocab = reduced_vocab
    vocab_lc = reduced_vocab_lc
    # apply cap to number of words
    if max_words_to_save > 0:
        vocab = vocab[:max_words_to_save]
        vocab_lc = vocab_lc[:max_words_to_save]


    # dictionary mapping word to index
    orig_vocab_dict = dict()
    for i,x in enumerate(orig_vocab_lc):
        orig_vocab_dict[x] = i
    #reduced_vocab_wv = [find_correct_case(x, orig_vocab, orig_vocab_lc, orig_vocab_dict) for x in reduced_vocab]


    
    # trim and saves
#    vecg2 = trim_word_vectors(vecg, vocab, ignore_case=False)
    vecg2 = trim_word_vectors(vecg, vocab,
                              vecs_keys=orig_vocab,
                              vecs_keys_lc=orig_vocab_lc,
                              vecs_keys_dict=orig_vocab_dict,
                              ignore_case=True)
    save_word_vectors_bin(vecg2, 'GoogleNews-vectors-negative300_trimmed.bin')
    missing_words_vecs =  find_missing_words(vecg2.keys(), vocab)

    # trim and save lowercase version
#    vecg2_lc = trim_word_vectors(vecg, vocab_lc, ignore_case=False)
    vecg2_lc = trim_word_vectors(vecg, vocab_lc,
                              vecs_keys=orig_vocab,
                              vecs_keys_lc=orig_vocab_lc,
                              vecs_keys_dict=orig_vocab_dict,
                              ignore_case=True)

    save_word_vectors_bin(vecg2_lc, 'GoogleNews-vectors-negative300_trimmed_lowercase.bin')
    missing_words_vecs_lc =  find_missing_words(vecg2_lc.keys(), vocab_lc)

