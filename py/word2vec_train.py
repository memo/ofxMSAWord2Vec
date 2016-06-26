# coding: utf-8
"""
@author: memo

Trains word2vec vector embeddings using skip-gram model as described in 
"Efficient Estimation of Word Representations in Vector Space" by Mikolov et al
http://arxiv.org/pdf/1301.3781.pdf

Based on the Tensorflow Udacity example at
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/5_word2vec.ipynb

also see
https://www.tensorflow.org/versions/r0.9/tutorials/word2vec/index.html


After training, exports data as binary
See file_io.py to convert between formats (CSV vs BIN)

edit 'data_path' and 'opts' dict below
"""

from __future__ import print_function
import collections
import math
import numpy as np
import random
import tensorflow as tf
import zipfile
#from matplotlib import pylab
from six.moves import range
import pickle
import time
import pprint
import os

from word2vec_utils import *

#%%

# So we can time the whole procedure
start_time = time.time()

# used to suffix exported files with timestamp
time_str = time.strftime("%Y.%m.%d_%H.%M.%S")

print('Starting at time:', time_str)



data_path   = 'data/'  # root of data path (see below)
txt_path    = data_path + 'txt/'  # where input text corpus are loaded from
vec_path    = data_path + 'vec/'  # where out vector file is exported to
model_path  = data_path + 'tmp/' + time_str + '/' # temp files (checkpoints)

# create output dir if doesn't exist
try: 
    os.makedirs(model_path)
except OSError:
    if not os.path.isdir(model_path):
        raise

#%%
opts = dict(
        batch_size = 128,  # mini-batch size
        num_dims = 256, # Dimension of the embedding vector.
        skip_window = 5, # How many words to consider left and right.
        num_skips = 4, # How many times to reuse an input to generate a label.
        # We pick a random validation set to sample nearest neighbors. here we limit the
        # validation samples to the words that have a low numeric ID, which by
        # construction are also the most frequent. 
        valid_size = 16, # Random set of words to evaluate similarity on.
        valid_window = 100, # Only pick dev samples in the head of the distribution.
        num_sampled = 128, # Number of negative examples to sample.
        
        num_words = 100000, # how many words to include in vocabulary
        
        num_steps = int(5e4+1), # number of iterations
        
        #input_filename = txt_path + 'text8.zip', # input corpus (must be zipped) 
        input_filename = txt_path + 'text9.zip', # input corpus (must be zipped) 
        )
        
# generate validation examples        
opts['valid_examples'] = np.array(random.sample(range(opts['valid_window']), opts['valid_size']))
pprint.pprint(opts)

# save options to file
print('Saving options file')
pickle.dump(opts, open(vec_path + 'word2vec_opts_' + time_str + '.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)



#%% Read the data into a string.
def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words"""
    print("Loading text file", filename)
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

words = read_data(opts['input_filename'])
opts['num_words'] = len(words)
print('Number of words %d' % opts['num_words'])



#%% Build the dictionary and replace rare words with UNK token.
def build_dataset(words):
    print("Building dataset")    
    word_count = [['UNK', -1]]
    word_count.extend(collections.Counter(words).most_common(opts['num_words'] - 1))
    dictionary = dict()
    for word, _ in word_count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count = unk_count + 1
        data.append(index)
    word_count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) 
    return data, word_count, dictionary, reverse_dictionary

data, word_count, dictionary, reverse_dictionary = build_dataset(words)
print('Most common words (+UNK)', word_count[:5])
print('Sample data', data[:10])
del words  # Hint to reduce memory.



#%% Function to generate a training batch for the skip-gram model.
data_index = 0
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1 # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window    # target label at the center of the buffer
        targets_to_avoid = [ skip_window ]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels

print('data:', [reverse_dictionary[di] for di in data[:8]])

for num_skips, skip_window in [(2, 1), (4, 2)]:
    data_index = 0
    batch, labels = generate_batch(batch_size=8, num_skips=num_skips, skip_window=skip_window)
    print('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
    print('    batch:', [reverse_dictionary[bi] for bi in batch])
    print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])


#%% Train a skip-gram model.

#graph = tf.Graph()
#with graph.as_default():

  # Input data.
train_dataset = tf.placeholder(tf.int32, shape=[opts['batch_size']])
train_labels = tf.placeholder(tf.int32, shape=[opts['batch_size'], 1])
valid_dataset = tf.constant(opts['valid_examples'], dtype=tf.int32)
  
  # Variables.
embeddings = tf.Variable(
    tf.random_uniform([opts['num_words'], opts['num_dims']], -1.0, 1.0))
    
softmax_weights = tf.Variable(
    tf.truncated_normal([opts['num_words'], opts['num_dims']],
                         stddev=1.0 / math.sqrt(opts['num_dims'])))
                         
softmax_biases = tf.Variable(tf.zeros([opts['num_words']]))
  
  # Model.
  # Look up embeddings for inputs.
embed = tf.nn.embedding_lookup(embeddings, train_dataset)

  # Compute the softmax loss, using a sample of the negative labels each time.
loss = tf.reduce_mean(
    tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases, embed,
                               train_labels, opts['num_sampled'], opts['num_words']))

  # Optimizer.
  # Note: The optimizer will optimize the softmax_weights AND the embeddings.
  # This is because the embeddings are defined as a variable quantity and the
  # optimizer's `minimize` method will by default modify all variable quantities 
  # that contribute to the tensor it is passed.
  # See docs on `tf.train.Optimizer.minimize()` for more details.
optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
  
  # Compute the similarity between minibatch examples and all embeddings.
  # We use the cosine distance:
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))

normalized_embeddings = embeddings / norm

valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)

similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))


#%%

# init saver
saver = tf.train.Saver()

# will store losses (so I can plot later)
losses = []

# doing session like this so I can play around in console
session = tf.Session()
session.run(tf.initialize_all_variables())

print('Initialized')
average_loss = 0
for step in range(opts['num_steps']):
    batch_data, batch_labels = generate_batch(
        opts['batch_size'], opts['num_skips'], opts['skip_window'])
    feed_dict = {train_dataset : batch_data, train_labels : batch_labels}
    _, l = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += l
    if step % 2000 == 0:
        if step > 0:
            average_loss = average_loss / 2000
        # The average loss is an estimate of the loss over the last 2000 batches.
        print('Average loss at step %d: %f' % (step, average_loss))
        losses.append((step, average_loss)) 
        average_loss = 0
    # note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:  
        sim = session.run(similarity)
        for i in range(opts['valid_size']):
            valid_word = reverse_dictionary[opts['valid_examples'][i]]
            top_k = 8 # number of nearest neighbors
            nearest = (-sim[i, :]).argsort()[1:top_k+1]
            log = 'Nearest to %s:' % valid_word
            for k in range(top_k):
                close_word = reverse_dictionary[nearest[k]]
                log = '%s %s,' % (log, close_word)
            print(log)
    if step % 50000 == 0 and step > 0:
        print("Saving checkpoint...")
        saver.save(session,
            os.path.join(model_path, "model.ckpt"),
            global_step=step)
        print("...done.")
        
        
        
#%%
        
final_embeddings = session.run(embeddings)
final_normalized_embeddings = session.run(normalized_embeddings)

end_time = time.time()
opts['train_duration'] = end_time - start_time
print("Train duration =", opts['train_duration'], 'seconds')

word_vecs = dict()
for i in reverse_dictionary:
    word_vecs[reverse_dictionary[i]] = final_embeddings[i]
    
    
print('Saving vectos')
save_word_vectors_bin(vec_path + 'word2vec_' + time_str + '.bin')

