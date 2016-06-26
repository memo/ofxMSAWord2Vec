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


#include "ofxMSAWord2Vec.h"
#include "ofxMSAVectorUtils.h"

namespace msa {

//--------------------------------------------------------------
bool Word2Vec::load_bin(string filename, int log_skip) {
    ofLogNotice() << "Word2Vec::load_csv(" << filename << ")";
    ofLogNotice() << "This may take a while, please be patient!";
    //        ofBuffer buf(filename);

    float start_time = ofGetElapsedTimef();

    FILE *f = fopen(ofToDataPath(filename).c_str(), "rb");
    if (f == NULL) {
        ofLogError() << "File not found";
        return false;
    }

    clear();

    fscanf(f, "%lld", &num_words);
    fscanf(f, "%lld", &num_dims);
    ofLogVerbose() << "num_words: " << num_words << ", num_dims: " << num_dims;

    for(int word_index = 0; word_index < num_words; word_index++) {
        // read characters into word until ' ' is encountered
        string word;
        while (1) {
            char c = fgetc(f);
            if (feof(f) || (c == ' ')) break;
            word += c;
        }

        if((log_skip > 0) && (word_index % log_skip == 0)) ofLogVerbose() << "Reading row " << word_index << " " << word_index * 100/ num_words << "% : " << word;


        word_vectors[word] = vector<float>(num_dims);
        vector<float> &vec = word_vectors[word];

        // read all floats
        for(int a = 0; a < num_dims; a++) fread(&vec[a], sizeof(float), 1, f);
    }
    fclose(f);

    update_norm_vectors();

    ofLogVerbose() << "Done in " << (ofGetElapsedTimef() - start_time);
    return true;
}



//--------------------------------------------------------------
bool Word2Vec::load_csv(string filename, string delimiter, int log_skip) {
    ofLogNotice() << "Word2Vec::load_csv(" << filename << ")";
    ofLogNotice() << "This may take a while, please be patient!";

    float start_time = ofGetElapsedTimef();

    ofFile file(filename);
    if(!file.exists()){
        ofLogError("The file " + filename + " is missing");
        return false;
    }

    clear();

    ofBuffer buf(file);

    int row_index = 0;
    // read header
    auto&& line = buf.getFirstLine();
    auto cells = ofSplitString(line, delimiter);
    num_words = ofToInt(cells[1]);
    num_dims = ofToInt(cells[3]);
    ofLogVerbose() << "num_words: " << num_words << ", num_dims: " << num_dims;

    while(!buf.isLastLine()) {
        auto&& line = buf.getNextLine();

        // split row into cells (first column contains word, rest contains vector data)
        auto cells = ofSplitString(line, delimiter);

        // get the word
        string word = cells[0];
        if((log_skip > 0) && (row_index % log_skip == 0)) ofLogVerbose() << "Reading row " << row_index << " : " << word;

        // get and check number of dimensions
        int nd = cells.size() - 1; // minus key (word)

        // ensure number of dimensions is same on all rows
        if(num_dims == nd) {
            word_vectors[word] = vector<float>(num_dims);
            for(int i=0; i<num_dims; i++) word_vectors[word][i] = ofToFloat(cells[i+1]);
        } else {
            ofLogWarning() << "Embedding sizes don't match. Skipping row " << row_index << ", word: " << word << " (" << num_dims << "!=" << nd << ")";
            //                return false;
        }
        row_index++;
    }

    update_norm_vectors();

    ofLogVerbose() << "Done in " << (ofGetElapsedTimef() - start_time);
    return true;
}




//--------------------------------------------------------------
bool case_insensitive_eq(const char& lhs, const char& rhs) {
    return std::toupper(lhs) == std::toupper(rhs);
}

bool case_insensitive_string_eq(const std::string& lhs, const std::string& rhs) {
  return std::equal(lhs.begin(),
                    lhs.end(),
                    rhs.begin(),
                    case_insensitive_eq);
}

struct case_insensitive_key_eq {
  case_insensitive_key_eq(const std::string& key) : key_(key) {}

  bool operator()(Word2Vec::WordVectorMap::value_type item) const {
    return case_insensitive_string_eq(item.first, key_);
  }
  std::string key_;
};


//--------------------------------------------------------------
vector<float> Word2Vec::search_word_vectors(const Word2Vec::WordVectorMap& word_vectors, string word, bool ignore_case) {
    auto &&it = word_vectors.find(word);

    // return vector if word is found as is
    if(it != word_vectors.end()) return it->second;

    // do case insensitive search if ignore_case is true
    if(ignore_case) {
        it = find_if(word_vectors.begin(), word_vectors.end(), case_insensitive_key_eq(word));
        if(it != word_vectors.end()) return it->second;
    }
    return vector<float>();
}

//--------------------------------------------------------------
vector<float> Word2Vec::word_vector(string word, bool ignore_case) const {
    return search_word_vectors(word_vectors, word, ignore_case);
}



//--------------------------------------------------------------
vector<float> Word2Vec::word_vector_norm(string word, bool ignore_case) const {
    return search_word_vectors(word_vectors_norm, word, ignore_case);
}



//--------------------------------------------------------------
vector < pair<string, float> > Word2Vec::find_closest_words(string word0, int top_k, int dist_mode, bool ignore_case) const {
    const auto& vec = word_vector(word0, ignore_case);
    if(vec.empty()) {
        ofLogWarning() << "Word '" << word0 << "' not found in dictionary.";
        return vector<pair<string, float> >();
    }

    return find_closest_words(vec, top_k, dist_mode);
}



//--------------------------------------------------------------
vector < pair<string, float> > Word2Vec::find_closest_words(const vector<float>& vec0, int top_k, int dist_mode) const {
    ofLogVerbose() << "find_closest_words - this may take a while, please be patient!";

    // create a normalized version of the vector
    vector<float>vec0_norm = vector_utils::normalized(vec0);

    // will store all distances
    vector < pair<string, float> > dists;//(num_words);

    // iterate all words and save distances
    for(auto&& word1 : word_vectors) {
        const vector<float>& vec1 = word1.second;
        vector<float> vec1_norm = word_vectors_norm.at(word1.first);
        float dist = (dist_mode == 0) ? vector_utils::dot(vec0_norm, vec1_norm) : -vector_utils::l2_norm(vector_utils::subtract(vec0, vec1));
        dists.push_back(make_pair(word1.first, dist));
    }

    // sort distances list by distance
    std::sort(dists.begin(), dists.end(), [](auto &left, auto &right) { return left.second > right.second; });

    // return top_k results
    vector < pair<string, float> > top_dists(dists.begin(), dists.begin() + top_k);
    return top_dists;
}


//--------------------------------------------------------------
void Word2Vec::clear() {
    num_words = 0;
    num_dims = 0;
    word_vectors.clear();
    word_vectors_norm.clear();
}


//--------------------------------------------------------------
void Word2Vec::update_norm_vectors() {
    word_vectors_norm.clear();

    for(auto&& word : word_vectors) {
        word_vectors_norm[word.first] = vector_utils::normalized(word.second);
    }
}

}
