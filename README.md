
## Introduction
A C++ openFrameworks addon to load and play with word vector embeddings (mappings of words to high dimensional vectors), and supplementary python files to train / save / load / convert / test word vector embeddings. 

The C++ openFrameworks code does not do word vector *training*, it only loads pretrained models in a binary or csv format. There is python code for training in the ./py folder. 

The binary file format is the one used by Mikolov et al at https://code.google.com/archive/p/word2vec/ and the models are based on "Efficient Estimation of Word Representations in Vector Space" (http://arxiv.org/pdf/1301.3781.pdf)

See bottom of README for more info and tutorials on word2vec.

## Usage
Download the pretrained models from the [releases tab](https://github.com/memo/ofxMSAWord2Vec/releases). More info on these below (For easiest / best results I recommend GoogleNews_xxxxxx_lowercase.bin).

###openFrameworks
	msa::Word2Vec word2vec;
	// load binary file (as opposed to csv, which is much slower to load)
	word2vec.load_bin('path/to/model.bin');

	// return sorted list of 10 closest words to 'kitten' (and their similarities)
	ret = word2vec.find_closest_words('kitten', 10);
	// returns: [[1 kitten], [0.78 puppy], [0.77 kittens], [0.75 cat], [0.74 pup], [0.72 puppies], [0.67 dog], [0.66 tabby], [0.65 chihuahua], [0.65 cats]]


	// perform vector arithmetic on words

	// get vectors for words...
	vking = word2vec.word_vector('king');
	vman = word2vec.word_vector('man');
	vwoman = word2vec.word_vector('woman');

	// get resulting vector for 'king' - 'man' + 'woman'
	v = msa::vector_utils::add(msa::vector_utils::subtract(vking, vman), vwoman);

	// find closest words to resulting vector. top result is 'queen'
	ret = word2vec.find_closest_words(v);

More 3-way analogies can be found in the paper [Linguistic Regularities in Continuous Space Word Representations. Mikolov et al. 2013](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/rvecs.pdf)
and nice explanation / tutorial at http://deeplearning4j.org/word2vec

There are also interesting two-way operations, e.g. (I selected interesting results from top 5)
twitter + bot = memes (seriously!)
human - god = animal
nature - god = dynamics
science + god = mankind, metaphysics,
human - love = animal
sex - love = intercourse, prostitution, masturbation, rape
love - sex = adore
love - passion = ugh, what'd

I made a twitter bot that is randomly exploring this space [Word of Math](https://twitter.com/wordofmath) 

###Python
Similar to above, but a lot more functionality, e.g. to find correct capitalization of word in model, trim words, perform intersections between two lists etc. I didn't port these to C++/openFrameworks because: 1.) it's way easier in python. 2.) the idea is you trim and prepare the model in python, then load the prepped model in C++. See ./py for examples.

###Tips
When finding closest words in embedded space cosine similarity is generally used as opposed to euclidean distance (L2 norm). Cosine similarity of two vectors is simply the *dot product of the normalized vectors*. For performance reasons I store both the normalized vectors and unnormalized vectors. This way for cosine similarity I can just dot product the normalized vectors straight away without having to normalize on the fly. 

When doing vector arithmetic on words, the *unnormalized* vectors should be used not the normalized vectors. Also, the results often include the original source words, so you may need to prune those. 

CSV files are human readable, but much larger and slower to load than BIN files. Once the file is loaded memory usage and performance is identical.

You can load the original 3.6 GB GoogleNews model by Mikolov et al but it will take quite a while to load, and rather slow to search. See **Models** section below.  




## Pre-trained word vector models
###memo_wiki9_word2vec_2016.06.10_04.42.00
A model I trained on 1 billion characters (~143 million words) of wikipedia. It's a skip-gram model with 256 dimensions and 100,000 word vocabulary, all lowercase. Surprisingly, despite the relatively small train training set (143 million words, as opposed to say 100 billion words, see below) the analogy math (king - man + woman -> queen etc) works quite well. 

The python source for this can be found in ./py/word2vec_train.py and the source data can be downloaded from http://mattmahoney.net/dc/textdata. The enwik9.zip (and enwik8.zip) files contain 1,000 million (and 100 million) characters respectively, but that includes all xml tags, links etc. The perl script in Appendix A (on src site) strips all tags and converts the text to lowercase characters, suitable for word2vec training. 

###GoogleNews-vectors-negative300_trimmed_53K_xxxx
This is based on *and is a very simplified version of* Mikolov et al's GoogleNews model. 

Mikolov et al trained it on roughly 100 billion words of Google News, with the parameters (note it's continous bag of words, not skip-gram):

	-cbow 1 -size 300 -window 5 -negative 3 -hs 0 -sample 1e-5 -threads 12 -binary 1 -min-count 10

The final model has a vocabulary of 3 million words (containing loads of weird symbols and special strings) and is 3.6GB, making it rather unmanageable and overkill for most applications (it takes forever to load and a simple search is far from realtime on normal hardware). So I drastically reduced the vocabulary (down to 53K words) by:
- limiting it to the top 100,000 english words as listed by wiktionary (https://gist.github.com/h3xx/1976236)
- removing all duplicate entries
- removing all variations of capitalization, keeping the vector for the *most common* entry (as listed in the wiktionary top 100K words). e.g. 'Paris', 'PARIS', 'paris' all gets replaced with single (most common, e.g. 'Paris') entry.

The final vocabulary size dropped to 53K (there were lots of entries in the top wiki 100K list which were also non-sensical and not in the Google News model). I saved two versions of the embeddings: 1.) preserving capitializations and 2.) forcing all entries to lowercase (which makes searching much quicker and easier). So these should suffice for most day to day needs. However if changes are required (e.g. to keep different capitalizations), the src for this transformation from 3.6GB/3M word vocabulary to 64MB/53K word vocabulary is in ./py/word2vec_googlenews_trim.py and the original 3.6GB data file can be downloaded from https://code.google.com/archive/p/word2vec/.

##Dependencies
###openFrameworks
The openFrameworks addon (and example) require my little [ofxMSAVectorUtils](https://github.com/memo/ofxMSAVectorUtils) addon and nothing else. It only loads a binary (or CSV) file and performs simple vector math and lookups. Should work on any platform on any recent openFrameworks version (v0.9+).


###Python
The python code for *training* requires Tensorflow 0.9 (possibly earlier would work, but that's what I'm using). The python code for loading / saving / playing with files and embeddings (*excluding training*) requires numpy and nltk. nltk isn't actually essential, it's only used in the python function 'phrase_shift' which shifts a whole sentence or phrase in embedded space where I use nltk for tokenizing the phrase. This could easily be done without nltk, but why bother when it's already there and [works out of the box so nicely](https://xkcd.com/353/).




##Acknowledgements
This is work-in-progress, made as part of a residency at Google's [Artists and Machine Intelligence](https://ami.withgoogle.com/) program. 
Currently two twitter bots are using this as a testbed: [Word of Math](https://twitter.com/wordofmath) and [Almost Inspire](https://twitter.com/almost_inspire). More releases in the pipeline. 

##References
**tutorials and background info on word2vec:**
https://code.google.com/archive/p/word2vec/
http://www.offconvex.org/2015/12/12/word-embeddings-1/
http://www.offconvex.org/2016/02/14/word-embeddings-2/
http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
http://rare-technologies.com/word2vec-tutorial/
https://www.tensorflow.org/versions/master/tutorials/word2vec/index.html

**papers:**
[Efficient Estimation of Word Representations in Vector Space. Mikolov et al. 2013](http://arxiv.org/pdf/1301.3781.pdf)
[Distributed Representations of Words and Phrases and their Compositionality. Mikolov et al. 2013](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
[Linguistic Regularities in Continuous Space Word Representations. Mikolov et al. 2013](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/rvecs.pdf)

**related / more recent:**
[GloVe: Global Vectors for Word Representation. Pennington et al. 2014](http://www-nlp.stanford.edu/pubs/glove.pdf)
[Distributed Representations of Sentences and Documents. Le at al. 2014](http://www.jmlr.org/proceedings/papers/v32/le14.pdf)
[Skip Thought Vectors. Kiros et al. 2015](http://papers.nips.cc/paper/5950-skip-thought-vectors.pdf)
[Deep Visual-Semantic Alignments for Generating Image Descriptions. Karpathy et al. 2015](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Karpathy_Deep_Visual-Semantic_Alignments_2015_CVPR_paper.pdf)

