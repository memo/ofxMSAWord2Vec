//                                     __
//    ____ ___  ___  ____ ___  ____   / /__   __
//   / __ `__ \/ _ \/ __ `__ \/ __ \ / __/ | / /
//  / / / / / /  __/ / / / / / /_/ // /_ | |/ /
// /_/ /_/ /_/\___/_/ /_/ /_/\____(_)__/ |___/
//
//
// Created by Memo Akten, www.memo.tv
//


#include "ofMain.h"

#include "ofxMSAWord2Vec.h"
#include "ofxMSAVectorUtils.h"

class ofApp : public ofBaseApp {
public:

    msa::Word2Vec word2vec;

    string str_input = "";      // string entered by user
    stringstream str_results;   // string containing results to display on screen

    // paths to various different model files
    vector<ofFile> model_files;

    int model_index = 0;        // which model to use (from array above)
    int dist_mode = 0;          // how to measure distance  between vectors. 0: cosine similarity (norm dot product), 1: L2 norm (euclidean distance)
    bool auto_search = true;    // automatically search with every key press (could be slow on slow computers)
    bool ignore_case = false;   // do case insensitive search (can be slower)

    //--------------------------------------------------------------
    void setup(){
        ofBackground(0);
        ofSetColor(255);
        ofSetVerticalSync(true);
        ofSetFrameRate(60);
        ofSetLogLevel(OF_LOG_VERBOSE);

        ofDirectory dir;
        dir.listDir("vec");
        model_files = dir.getFiles();

        load_model(model_files[0]);
    }

    //--------------------------------------------------------------
    void load_model(ofFile file) {
        if(file.getExtension() == "csv") word2vec.load_csv(file.getAbsolutePath());
        else word2vec.load_bin(file.getAbsolutePath());
    }

    //--------------------------------------------------------------
    void process_string(const string& str_input) {
        // input can be a single word, or a phrase (sequence of words), or simple arithmetic operation. Split into tokens.
        auto tokens = ofSplitString(str_input, " ", true, true);


        vector<float> vec;
        int op = 0; // arithmetic operator to perform

        // iterate all tokens (words and maths operators)
        for(auto&& t : tokens) {

            // if maths operator, save sign and write to display string
            if(t == "+") { op = 1; str_results << t << endl; }
            else if(t == "-") { op = -1; str_results << t << endl; }
            else {
                // otherwise get vector for word
                auto v = word2vec.word_vector(t, ignore_case);

                // if word found...
                if(!v.empty()) {
                    // if we have a maths operator on stack, apply it, otherwise save vector as is
                    if(op) vec = msa::vector_utils::weighted_sum(vec, 1.0f, v, 1.0f * op);
                    else vec = v;

                    // write to display string
                    str_results << t << ": " << msa::vector_utils::to_string(vec) << endl;
                } else {
                    str_results << t << ": " << "not found" << endl;
                }
                op = 0; // clear maths operator stack
            }
        }

        if(!vec.empty()) {
            if(tokens.size() > 1) str_results << "=" << endl << msa::vector_utils::to_string(vec) << endl;
            str_results << endl;

            // find nearest words to result vector
            auto results = word2vec.find_closest_words(vec, 20, dist_mode);

            // write to display string
            for(auto&&l : results) str_results << l.second << " " << l.first << endl;
        }
    }

    //--------------------------------------------------------------
    void draw(){

        stringstream s;
        s << ofGetFrameRate() << endl;
        s << endl;
        s << "model (toggle with '1') : " << model_files[model_index].getFileName() << " | num_words: " << word2vec.get_num_words() << " | num_dims: " << word2vec.get_num_dims() << endl;
        s << "dist_mode (toggle with '2') : " << (dist_mode==0 ? "cosine distance" : "L2 norm") << endl;
        s << "auto_search (toggle with '3':) : " << (auto_search ? "TRUE" : "FALSE") << endl;
        s << "ignore_case (toggle with '4':) : " << (ignore_case ? "TRUE" : "FALSE") << endl;
        s << endl;
        s << "ENTER to search" << endl;
        s << "BKSPC to delete last character" << endl;
        s << "TAB to clear entry" << endl;
        s << "+ and - operators (with spaces before and after) to do vector math" << endl;
        s << endl;
        s << endl;
        s << "Type in a word, phrase or maths operation to see closest words:" << endl;
        s << "e.g.: king" << endl;
        s << "e.g.: king - queen + woman (should return 'man')" << endl;
        s << "e.g.: Paris - France + Spain (should return 'Madrid')" << endl;
        s << endl;
        s << str_input << ((ofGetFrameNum()/30) % 2 == 0 ? "_" : "") << endl;
        s << endl;
        s << endl;
        s << str_results.str() << endl; // display string

        ofDrawBitmapString(s.str(), 50, 50);
    }

    //--------------------------------------------------------------
    void keyPressed(int key){
        str_results.str("");

        bool do_search = auto_search;

        switch(key) {
        case OF_KEY_TAB:
            str_input = "";
            break;

        case OF_KEY_BACKSPACE:
            str_input = str_input.substr(0, str_input.size() - 1);
            break;

        case OF_KEY_RETURN:
            do_search = true;
            break;

        case '1':
            model_index = (model_index + 1) % model_files.size();
            load_model(model_files[model_index]);
            break;

        case '2':
            dist_mode = 1 - dist_mode;
            break;

        case '3':
            auto_search ^= true;
            break;

        case '4':
            ignore_case ^= true;
            break;

        case ' ':
        case '+':
        case '-':
            str_input += key;
            break;

        default:
            if(isalnum(key)) str_input += key;

        }

        if(do_search) process_string(str_input);
    }
};


//========================================================================
int main( ){
    ofSetupOpenGL(1024,768,OF_WINDOW);			// <-------- setup the GL context
    ofRunApp(new ofApp());
}
