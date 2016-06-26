//                                     __
//    ____ ___  ___  ____ ___  ____   / /__   __
//   / __ `__ \/ _ \/ __ `__ \/ __ \ / __/ | / /
//  / / / / / /  __/ / / / / / /_/ // /_ | |/ /
// /_/ /_/ /_/\___/_/ /_/ /_/\____(_)__/ |___/
//
//
// Created by Memo Akten, www.memo.tv
//
// load and play with word vector embeddings (mappings of words to high dimensional vectors),
//
// see README (github.com/memo/ofxMSAWord2Vec) for more info


#pragma once

#include "ofMain.h"

namespace msa {

class Word2Vec {
public:
    typedef map<string, vector<float> > WordVectorMap;

    // load word vector embeddings from BIN as used by Tomas Mikolov et al (and in word2vec_file_io.py)
    bool load_bin(string filename, int log_skip = 10000);

    // load word vector embeddings from CSV as exported from word2vec_file_io.py
    bool load_csv(string filename, string delimiter = "\t", int log_skip = 1000);

    // get raw and normalized vectors for word
    // case insensitive search (ignore_case==true) is slower and returns first instance if more than one capitalization variation is found (e.g. King, king, KING)
    vector<float> word_vector(string word, bool ignore_case = false) const;
    vector<float> word_vector_norm(string word, bool ignore_case = false) const;

    // standard getters
    int get_num_dims() const { return num_dims; }
    int get_num_words() const { return num_words; }


    // find closest words to given word or vector
    // returns sorted vector of closest top_k <word, distance> pairs
    // dist_mode==0: use cosine distance (recommmended), dist_mode==1: use L2 norm
    // case insensitive search (ignore_case==true) is slower and returns first instance if more than one capitalization variation is found (e.g. King, king, KING)
    vector < pair<string, float> > find_closest_words(string word, int top_k=16, int dist_mode=0, bool ignore_case=false) const;
    vector < pair<string, float> > find_closest_words(const vector<float>& vec0, int top_k=16, int dist_mode=0) const;


protected:
    WordVectorMap word_vectors; // maps word(string) to a vector of floats
    WordVectorMap word_vectors_norm; // maps word(string) to a normalized vector of floats
    long long num_dims;
    long long num_words;

    void clear();
    void update_norm_vectors();

    static vector<float> search_word_vectors(const WordVectorMap& word_vectors, string word, bool ignore_case = false);
};

}
