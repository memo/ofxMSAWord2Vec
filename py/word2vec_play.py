# coding: utf-8

"""
@author: memo

"""
import os
from word2vec_utils import *

# handy vars to link directly to the different models to compare
data_path = './data'  # root of data path relative to python working folder
vec_path = os.path.join(data_path, 'vec')  # location of pretrained vector models

gorig_path = os.path.join(vec_path, 'GoogleNews-vectors-negative300') 
gtrim_path = os.path.join(vec_path, 'GoogleNews-vectors-negative300_trimmed_53K')
gtrim_lc_path = os.path.join(vec_path, 'GoogleNews-vectors-negative300_trimmed_53K_lowercase')
wiki_path = os.path.join(vec_path, 'memo_wiki9_word2vec_2016.06.10_04.42.00')


# instead of running the whole function, much easier to run each line one by one
# Very easy to do in spyder ( https://pythonhosted.org/spyder/ )
def test():
#    # load lowercase cutdown version of Mikolov et al's GoogleNews model
    vecs_gtrim_lc = load_word_vectors_bin(gtrim_lc_path + '.bin') 
#
#    # normalized version of above vectors
    vecs_gtrim_lc_n = normalize_word_vectors(vecs_gtrim_lc)
#    
    # load cutdown version of Mikolov et al's GoogleNews model
    vecs_gtrim = load_word_vectors_bin(gtrim_path + '.bin')

    # normalized version of above vectors
    vecs_gtrim_n = normalize_word_vectors(vecs_gtrim)
    
    # load 100K word vocabulary trained on 1B characters of wikipedia
    vecs_wiki9 = load_word_vectors_bin(wiki_path + '.bin')

    # normalized version of above vectors
    vecs_wiki9_n = normalize_word_vectors(vecs_wiki9)
    

    # choose a model to work with
    vecs, vecs_n = vecs_gtrim, vecs_gtrim_n # google news trim
    vecs, vecs_n = vecs_gtrim_lc, vecs_gtrim_lc_n # google news lowercase
    vecs, vecs_n = vecs_wiki9, vecs_wiki9_n # wikipedia
    
    
    # find closest words from different models
    w = 'king'
    w = 'pAris'
    w = 'bot'

    find_correct_case(w, vecs.keys())
    
    find_closest_words(vecs_n, w, cutoff=0)
    find_closest_words(vecs_n, w, cutoff=1)
    find_closest_words(vecs_n, w, cutoff=0.1)

    
    # perform word analogies (look at top result that excludes the three input words)
    ws = ('king', 'queen', 'woman')
    ws = ('king', 'man', 'woman')
    ws = ('paris', 'france', 'spain')
    word_analogy(vecs, vecs_n, ws, cutoff=0)
    word_analogy(vecs, vecs_n, ws, cutoff=1)
    word_analogy(vecs, vecs_n, ws, cutoff=0.8)
    word_analogy(vecs, vecs_n, ws, cutoff=0.6)
    word_analogy(vecs, vecs_n, ws, cutoff=0.5)
    word_analogy(vecs, vecs_n, ws, cutoff=0.2)
    
    wo = ( (+1, 'twitter'), (+1, 'bot'))
    do_word_maths(vecs, vecs_n, wo, cutoff=0)
    do_word_maths(vecs, vecs_n, wo, cutoff=1)
    do_word_maths(vecs, vecs_n, wo, cutoff=0.8)
    do_word_maths(vecs, vecs_n, wo, cutoff=0.6)


    phrase = "Friendship depends on trust and trust grows when we live our lives honestly and sincerely, cultivating respect and concern for others."
    phrase_shift(vecs_n, phrase)
    #phrase_shift(vecs_trim_n, "Friendship depends on trust and trust grows when we live our lives honestly and sincerely, cultivating respect and concern for others.")
    #phrase_shift(vecs_wiki9_n, "Friendship depends on trust and trust grows when we live our lives honestly and sincerely, cultivating respect and concern for others.")    
