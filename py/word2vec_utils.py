# coding: utf-8

"""
@author: memo

Util functions for playing with, trimming, saving, loading word vector embeddings

see README (github.com/memo/ofxMSAWord2Vec) for more info

TYPE STRUCTURE: 
word_vecs = dict { (string) word : (np.array) vector }

"""


from __future__ import print_function

# using nltk to strip punctuation from and tokenize phrases 
# could easily be done without nltk

import nltk  # only needed for phrase_shift()
# make sure nltk data for tokenizing is downloaded before using phrase_shfit()
# nltk.download()

import csv
import struct
import sys
import time
import random
import numpy as np
from difflib import get_close_matches


def inc_and_write_perc(i, maxi):
    """for displaying percentage progress"""
    mult = 100.0/maxi
    old_perc = round(i * mult)
    i += 1
    new_perc = round(i * mult)
    if new_perc != old_perc:
        sys.stdout.write("%d%%\r" % new_perc)
        sys.stdout.flush()
        #print("%d%%\r" % new_perc)
    return i
    

def load_word_vectors_bin(filepath, max_words=-1):
    """
    Binary format used by Tomas Mikolov et al
    https://code.google.com/archive/p/word2vec/
    
    Python loader code based on @ottokart's from  
    https://gist.github.com/ottokart/673d82402ad44e69df85
    
    max_words : if non-zero, stops loading after having reached max_words words
    """
    
    print("load_word_vectors_bin :", filepath, " ... ")
    start_time = time.time()
    word_vecs = dict()    
    with open(filepath, 'rb') as f:
        c = None
        
        # read the header
        header = ""
        while c != "\n":
            c = f.read(1)
            header += c
    
        total_num_words, num_dims = (int(x) for x in header.split())
        if max_words > 0:
            num_words = min(max_words, total_num_words)
        else:
            num_words = total_num_words
        
        print("num_words: %d/%d" % (num_words, total_num_words))
        print("num_dims: %d" % num_dims)

        i = 0        
        while len(word_vecs) < num_words:
            word = ""
            while True:
                c = f.read(1)
                if c == " ":
                    break
                word += c
    
            binary_vector = f.read(4 * num_dims)
            word_vecs[word] = [ struct.unpack_from('f', binary_vector, x)[0] 
                              for x in xrange(0, len(binary_vector), 4) ]
                                  
            # convert to numpy
            word_vecs[word] = np.array(word_vecs[word], dtype=np.float32)
            
            #  write new percentage if changed
            i = inc_and_write_perc(i, num_words)
            
    print("done in %s seconds." % (time.time() - start_time))
    print('-' * 60)
    return word_vecs



def save_word_vectors_bin(word_vecs, filepath):
    """
    Binary format used by Tomas Mikolov et al
    https://code.google.com/archive/p/word2vec/
    """

    print("save_word_vectors_bin :", filepath, " ... ")
    start_time = time.time()
    num_words = len(word_vecs)
    num_dims = len(word_vecs[word_vecs.keys()[0]])
    i = 0        
    with open(filepath, 'wb') as f:
        f.write(str(num_words) + ' ' + str(num_dims) + '\n')
    
        for word in word_vecs:
            # write word
            f.write(word + ' ')
            
            # write vector
            for v in word_vecs[word]:
                f.write(struct.pack('f', v))
            # f.write(struct.pack(vec_format, word_vecs[word]))
            
            #  write new percentage if changed
            i = inc_and_write_perc(i, num_words)
                
    print("done in %s seconds." % (time.time() - start_time))
    print('-' * 60)



def save_word_vectors_csv(word_vecs, filepath, delimiter='	'): # default delimeter is TAB
    print("save_word_vectors_csv :", filepath, " ... ")
    start_time = time.time()
    num_words = len(word_vecs)
    num_dims = len(word_vecs[word_vecs.keys()[0]])
    i = 0        
    with open(filepath, 'w') as f:
        writer = csv.writer(f, delimiter=delimiter, quotechar='"', quoting=csv.QUOTE_NONE)
        writer.writerow(['num_words', num_words, 'num_dims', num_dims])
        for key in word_vecs:
            # write to csv
            writer.writerow([key] + list(word_vecs[key]))
            
            #  write new percentage if changed
            i = inc_and_write_perc(i, num_words)

    print("done in %s seconds." % (time.time() - start_time))
    print('-' * 60)


        
            
def normalize_vector(v):
    """Normalize a numpy vector, checking if nonzero first"""
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v/norm


def normalize_word_vectors(word_vecs):
    """returns normalized word vectors"""

    print("normalize_word_vectors ... ", end='')
    start_time = time.time()
    word_vecs_norm = dict()
    for word in word_vecs:
        word_vecs_norm[word] = normalize_vector(word_vecs[word])
    print("done in %s seconds." % (time.time() - start_time))
    return word_vecs_norm


"""
# for testing capitalization stuff
l0 = ['piPe', 'cat', 'PIPE', 'cat', 'cAt', 'DOG', 'caT']
l1 = ['memo', 'jane', 'cat', 'CAT', 'cAt', 'dOg']

make_word_list_unique(l0, ignore_case=False)
make_word_list_unique(l0, ignore_case=True)

keep_words(l0, l1, False)
keep_words(l0, l1, True)

make_word_list_unique(keep_words(l0, l1, False), False)
make_word_list_unique(keep_words(l0, l1, False), True)
make_word_list_unique(keep_words(l0, l1, True), False)
make_word_list_unique(keep_words(l0, l1, True), True)

find_correct_case('CaT', l0)
"""
        

def find_correct_case(word, words, words_lc=None, words_dict=None):
    """
    Find correct capitalization first occurance of word in list ignoring case
    words_lc (optional) has to have the same order as words, and be lowercase versions
    word_dict (optional) maps lowercase word to its index
    """
    
    # first check words list as is
    if word in words:
        return word
    
    # create lower case words list if lowercase words list hasn't been passed
    if words_lc == None:
        words_lc = map(str.lower, words)
    
    word_lc = word.lower()
    if word_lc in words_lc:
        if words_dict != None:
            index = words_dict[word_lc]
        else:
            index = words_lc.index(word_lc)
        return words[index]
        
    # return nothing
    return None
    
    
# TODO check ignore_case
def make_word_list_unique(words, ignore_case):
    """Make list of words unique (remove duplicates) while preserving order"""
    seen = set()
    if ignore_case:
        ret = [x for x in words if not (x.lower() in seen or seen.add(x.lower()))]
    else:
        ret = [x for x in words if not (x in seen or seen.add(x))]
    return ret
    

# TODO check ignore_case
def trim_word_vectors(word_vecs, vocab, vecs_keys=None, vecs_keys_lc=None, vecs_keys_dict=None, ignore_case=False):
    """Trims word_vecs to the words in vocab"""
    
    print("trim_word_vectors")
    start_time = time.time()

    if ignore_case:
        keys_set = frozenset(map(str.lower, word_vecs.keys()))
#        word_vecs_new = dict( (k, word_vecs[k]) for k in vocab if k.lower() in keys_set)
        word_vecs_new_keys =( find_correct_case(k, vecs_keys, vecs_keys_lc, vecs_keys_dict) for k in vocab if k.lower() in keys_set)
        word_vecs_new = dict( (k, word_vecs[k]) for k in word_vecs_new_keys )
    else:
        keys_set = frozenset(word_vecs.keys())
        word_vecs_new = dict( (k, word_vecs[k]) for k in vocab if k in keys_set )

    print("done in %s seconds." % (time.time() - start_time))
    print('-' * 60)

    return word_vecs_new


# TODO check ignore_case
def keep_words(l1, l2, ignore_case):
    """Return intersection of two lists of words, maintaining order of first"""

    if ignore_case:
        l2_set = frozenset(map(str.lower, l2))
        r = [x for x in l1 if x.lower() in l2_set]
    else:
        l2_set = frozenset(l2)
        r = [x for x in l1 if x in l2_set]
    return r
    

    # TODO: make case insensitive
def remove_words_with_similarity(words, filter_words, cutoff=0.6):
    """remove filter_words (and its variants) from words list"""
    # if string passed instead of list, make it a list
    if type(filter_words)==str:
        filter_words=(filter_words,)
        
    l = list(words)
    for filter_word in filter_words:
        # find matches
        matches = get_close_matches(filter_word, l, n=5, cutoff=cutoff)
        
        # remove from list
        l = [x for x in l if x not in matches] 
    return l


    # TODO: make case insensitive
def remove_words_self(words, cutoff=0.6):
    """self filter word variants KEEPING ONLY WORD ITSELF (first appearance of word)"""
    # clone words array (to not destroy it)    
    l = list(words)
    for i, word in enumerate(l):
        
        # find matches for current word
        matches = get_close_matches(word, l, n=5, cutoff=cutoff)
        
        # iterate matches (excluding current word)
        for m in matches[1:]:
            # find index
            j = l.index(m)
            
            # replace if greater (i.e. below in list)
            if j > i:
                l[j] = word
    l = make_word_list_unique(l, ignore_case=False)
    return l



    # TODO: make case insensitive
def remove_words_from_word_dist_pairs(word_dists, filter_words, cutoff=0.6):
    if type(filter_words)==str:
        filter_words=[filter_words]
    
    
    for filter_word in filter_words:
        # apply word filters    
        words = zip(*word_dists)[0]
        
        # filter out similar words from words list
        words = remove_words_with_similarity(words, filter_word, cutoff=cutoff)
        
        # remove words from word_dists pair
        word_dists = [ (x[0], x[1]) for x in word_dists if x[0] in words ]
    return word_dists
    


def find_closest_words(word_vecs, o, top_k=10, dist_mode=0, cutoff=1):
    """
    Finds closest top_k words, either by cosine distance or L2 norm.
    Returns list of (word, distance) pairs
    Brute force search. Could be optimized, but pretty fast for 100K vocab.

    e.g.
    find_closest_words(word_vecs, 'king')
    
    word_vecs : dictionary of word vectors. maps word (string) to vector (numpy array)
        for cosine distance pass in normalized word_vecs,
        for L2 distance pass in unnormalized word_vecs
            
    o : the word (as string) or vector (python list or numpy array) to find closest words to
    top_k : how many results to return
    dist_mode : 0 for cosine distance (dot product of normalized vectors), 1 for L2 norm (euclidean distance)
    cutoff : if nonzero, threshold for similarity based on word stem (e.g. king, kings, kingly etc.)
                    e.g.
                    1.0: filter out only src word as is
                    0.8: filter out only src word and very similar words
                    0.1: filter out vaguely similar words, 
                    
    """
    
    vec0 = []

    # src word we're search for
    word0 = None
    
    # check type of o and act accordingly
    if type(o) == str:
        # find correct capitalization
        word0 = find_correct_case(o, word_vecs.keys())
        
        # see if it exists in embedding
        try:
            vec0 = word_vecs[word0]
        except KeyError, e:
            print('KeyError: "%s"' % str(e))
            return [(o, 1)] # otherwise return same word

    elif type(o) == np.ndarray or type(o) == list or type(o) == tuple:
        vec0 = o
        
    # if vector is python list, convert to numpy array    
    if type(vec0) != np.ndarray:
        vec0 = np.array(vec0, dtype=np.float32)
        
    # create a normalized version of the vector    
    vec0_norm = normalize_vector(vec0)    
    
    
    # will store all words and distances
    word_dists = []
    
    # iterate all words and save distances
    for word1 in word_vecs:
        vec1 = word_vecs[word1] # must be normalized if using cos distance
        dist = np.dot(vec0_norm, vec1) if dist_mode==0 else np.linalg.norm(vec0 - vec1)
        word_dists.append( (word1, dist) )


    # sort distances list by distance (cosine similarity or L2 norm)    
    word_dists = sorted(word_dists, key=lambda x:x[1], reverse=True)

    # apply filter to output if desired    
    if word0 != None and cutoff>0:
        # for speed purposes take top_k * arbitrary number (assuming this much redundancy max)
        word_dists = word_dists[:top_k * 2]
        word_dists = remove_words_from_word_dist_pairs(word_dists, word0, cutoff=cutoff)
    
    # return top_k results
    return word_dists[:top_k]


def do_word_maths(vecs, vecs_norm, word_op_pairs, top_k=10, dist_mode=0, cutoff=1):
    """
    input: list of (operator, word) pairs
    output: sorted list of words, unfiltered

    vecs : dictionary of (unnormalized) word vectors. maps word (string) to vector (numpy array)
    vecs_norm : dictionary of (normalized) word vectors. maps word (string) to vector (numpy array)
    words : list of strings
    top_k : how many results to return
    dist_mode : 0 for cosine distance (dot product of normalized vectors), 1 for L2 norm (euclidean distance)
    cutoff : if nonzero, threshold for similarity based on word stem (e.g. king, kings, kingly etc.)
                    e.g.
                    1.0: filter out only src word as is
                    0.8: filter out only src word and very similar words
                    0.1: filter out vaguely similar words, 

    """

    word_vecs_keys = vecs.keys()
    word_vecs_keys_lc = map(str.lower, word_vecs_keys)
    
    v_sum = None 
    for ow in word_op_pairs:
        o = ow[0]
        w = ow[1]
        try:
            v = vecs[find_correct_case(w, word_vecs_keys, word_vecs_keys_lc)]
        except KeyError, e:
            print('KeyError: "%s"' % str(e))
            return None
        
        
        if v_sum == None:
            v_sum = o * v
        else:
            v_sum += o * v
            
        
    word_dists = find_closest_words(vecs_norm, v_sum, top_k, dist_mode, cutoff=0)
    
    if cutoff > 0:
        # get src words
        filter_words = zip(*word_op_pairs)[1]

        word_dists = remove_words_from_word_dist_pairs(word_dists, filter_words, cutoff=cutoff)
        
    return word_dists[:top_k]


def word_analogy(vecs, vecs_norm, words, top_k=10, dist_mode=0, cutoff=1):
    """
    Try word analaogies.
    word[0] is to word[1] as _____ is to word[2]
    i.e. word[0] - word[1] + word[2]
    Returns list of (word, distance) pairs
            
    vecs : dictionary of (unnormalized) word vectors. maps word (string) to vector (numpy array)
    vecs_norm : dictionary of (normalized) word vectors. maps word (string) to vector (numpy array)
    words : list of strings
    top_k : how many results to return
    dist_mode : 0 for cosine distance (dot product of normalized vectors), 1 for L2 norm (euclidean distance)
    cutoff : if nonzero, threshold for similarity based on word stem (e.g. king, kings, kingly etc.)
                    e.g.
                    1.0: filter out only src word as is
                    0.8: filter out only src word and very similar words
                    0.1: filter out vaguely similar words, 
    """
    
    ops = [1, -1, +1]
    word_ops = zip(ops, words)
    return do_word_maths(vecs, vecs_norm, word_ops, top_k=top_k, dist_mode=dist_mode, cutoff=cutoff)


def phrase_shift(word_vecs,
                 phrase,
                 offset_rand_start = 0,
                 offset_rand_end = 5,
                 dist_mode = 0,
                 prune_variants = False,
                 rand_pow = 1):
    """
    Randomly shift all words in a phrase. 
    
    word_vecs : dictionary of word vectors. maps word (string) to vector (numpy array)
        for cosine distance pass in normalized word_vecs,
        for L2 distance pass in unnormalized word_vecs
    word1, word2, word3 : words (as strings)
    top_k : how many results to return
    dist_mode : 0 for cosine distance (dot product of normalized vectors), 1 for L2 norm (euclidean distance)
    """

    # tokenize words (split all words and punctuation, could be done without nltk)    
    words = nltk.word_tokenize(phrase)
    
    new_words = []
    
    # iterate all words
    for word in words:
        # get nearest words (this returns tuples iincluding words and their distances)
        alt_words = find_closest_words(word_vecs, word, offset_rand_end, dist_mode)
        
        # extract words (i.e. get rid of distances), and convert to lowercase        
        alt_words = [w[0].lower() for w in alt_words]
        
        #remove UNKs
        alt_words = [w for w in alt_words if  w != 'unk']

        # remove variants of the word (could do in single pass with above)
        if prune_variants:
            alt_words = [w for w in alt_words if (word not in w) and (w not in word) == True]

        index = -1
        if len(alt_words) > 0:        
            
            # pick a random index
            findex = random.uniform(0, 1)
            findex = pow(findex, rand_pow) # higher rand_pow tends towards zero
            index = int(np.interp(findex, [0., 1.], [offset_rand_start, offset_rand_end]))
            
            # pick the word
            new_word = alt_words[index % len(alt_words)]
        else:
            new_word = word # if no alternative words are found, use word as is

        # add to our list of new words
        new_words.append(new_word)
        print(word, '->', new_word, index, alt_words)
        
    
    # join all words with spaces
    new_phrase = ' '.join(new_words)

    # all of the punctuation also have spaces before them now, remove those spaces
    # not very smart way of doing it :/
    punctuation = ',./?;\!'
    for c in punctuation:
        ss = ' ' + c 
        new_phrase = new_phrase.replace(ss, c)
        
    new_phrase = new_phrase.replace('@ ', '@') # remove spaces after @
    new_phrase = new_phrase.replace('_', ' ') # replace underscores with space
    new_phrase = new_phrase.replace(" 's ", "'s ") # remove space before 's
        
    return new_phrase 



# TODO: test properly
from sklearn.manifold import TSNE

def plot_tsne(word_vecs, num_points=500):
    ii = random.sample(range(0, len(word_vecs)), num_points)
    words = word_vecs.keys();
    words = [ words[x] for x in ii]
    vecs = [word_vecs[x] for x in words]

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    embeddings = tsne.fit_transform(np.row_stack(vecs)) # final_embeddings[1:num_points+1, :])
    
    pylab.figure(figsize=(15,15))  # in inches
    for i, label in enumerate(words):
        x, y = embeddings[i,:]
        pylab.scatter(x, y)
        pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                   ha='right', va='bottom')
    pylab.show()

