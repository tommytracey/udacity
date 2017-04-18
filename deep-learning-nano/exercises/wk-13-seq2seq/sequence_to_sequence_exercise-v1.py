
# coding: utf-8

# *Exercise completed by Thomas Tracey on Apr 14, 2017*

# # Character Sequence to Sequence 
# In this notebook, we'll build a model that takes in a sequence of letters, and outputs a sorted version of that sequence. We'll do that using what we've learned so far about Sequence to Sequence models.
# 
# <img src="images/sequence-to-sequence.jpg"/>
# 
# 
# ## Dataset 
# 
# The dataset lives in the /data/ folder. At the moment, it is made up of the following files:
#  * **letters_source.txt**: The list of input letter sequences. Each sequence is its own line. 
#  * **letters_target.txt**: The list of target sequences we'll use in the training process. Each sequence here is a response to the input sequence in letters_source.txt with the same line number.

# In[5]:

import helper

# Load the data
source_path = 'data/letters_source.txt'
target_path = 'data/letters_target.txt'

source_sentences = helper.load_data(source_path)
target_sentences = helper.load_data(target_path)


# Let's start by examining the current state of the dataset. `source_sentences` contains the entire input sequence file as text delimited by newline symbols.

# In[6]:

source_sentences[:50].split('\n')


# `target_sentences` contains the entire output sequence file as text delimited by newline symbols.  Each line corresponds to the line from `source_sentences`.  `target_sentences` contains a sorted list characters of the line.

# In[7]:

target_sentences[:50].split('\n')


# ## Preprocess
# To do anything useful with it, we'll need to turn the characters into a list of integers: 

# In[8]:

set_words_1 = list(set(char for line in source_sentences.split('\n') for char in line))


# In[9]:

set_words_1[:10]


# In[10]:

def char_to_vec(data):
    # Define character sets
    special_chars = ['<pad>', '<unk>', '<s>',  '<\s>']
    set_chars = list(set(char for line in data.split('\n') for char in line))
    
    # Map chars to IDs
    int_to_char = {i: char for i, char in enumerate(special_chars + set_chars)}
    char_to_int = {char: i for i, char in int_to_char.items()}

    return int_to_char, char_to_int


# Build int_to_char and char_to_int dicts for source and target data
source_int_to_char, source_char_to_int = char_to_vec(source_sentences)
target_int_to_char, target_char_to_int = char_to_vec(target_sentences)

# Convert characters to ids
source_char_ids = [[source_char_to_int.get(letter, source_char_to_int['<unk>']) for letter in line] for line in source_sentences.split('\n')]
target_char_ids = [[target_char_to_int.get(letter, target_char_to_int['<unk>']) for letter in line] for line in target_sentences.split('\n')]

print("Example source sequence")
print(source_char_ids[:3])
print("\n")
print("Example target sequence")
print(target_char_ids[:3])


# The last step in the preprocessing stage is to determine the the longest sequence size in the dataset we'll be using, then pad all the sequences to that length.

# In[11]:

def pad_id_sequences(source_ids, source_letter_to_int, target_ids, target_letter_to_int, seq_length):
    new_source_ids = [sentence + [source_char_to_int['<pad>']] * (seq_length - len(sentence))                       for sentence in source_ids]
    new_target_ids = [sentence + [target_char_to_int['<pad>']] * (seq_length - len(sentence))                       for sentence in target_ids]
    
    return new_source_ids, new_target_ids


# Use the longest sequence as sequence length
seq_length = max([len(sentence) for sentence in source_char_ids] +                       [len(sentence) for sentence in target_char_ids])

# Pad all sequences up to sequence length
source_vecs, target_vecs = pad_id_sequences(source_char_ids, source_char_to_int,                                             target_char_ids, target_char_to_int, seq_length)


print("Sequence Length")
print(seq_length)
print("\n")
print("Input sequence example")
print(source_vecs[:3])
print("\n")
print("Target sequence example")
print(target_vecs[:3])


# This is the final shape we need them to be in. We can now proceed to building the model.

# ## Model
# #### Check the Version of TensorFlow
# This will check to make sure you have the correct version of TensorFlow

# In[1]:

from distutils.version import LooseVersion
import tensorflow as tf

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))


# ### Hyperparameters

# In[16]:

# Number of Epochs
epochs = 10
# Batch Size
batch_size = 128
# RNN Size
rnn_size = 100
# Number of Layers
num_layers = 2
# Embedding Size
encode_embed_size = 13
decode_embed_size = 13
# Learning Rate
learning_rate = 0.003


# ### Input

# In[17]:

# set placeholders

inputs_ = tf.placeholder(tf.int32, [batch_size, seq_length], name='inputs')
targets_ = tf.placeholder(tf.int32, [batch_size, seq_length], name='targets')
lr = tf.placeholder(tf.float32)


# ### Sequence to Sequence
# The decoder is probably the most complex part of this model. We need to declare a decoder for the training phase, and a decoder for the inference/prediction phase. These two decoders will share their parameters (so that all the weights and biases that are set during the training phase can be used when we deploy the model).
# 
# 
# First, we'll need to define the type of cell we'll be using for our decoder RNNs. We opted for LSTM.
# 
# Then, we'll need to hookup a fully connected layer to the output of decoder. The output of this layer tells us which word the RNN is choosing to output at each time step.
# 
# Let's first look at the inference/prediction decoder. It is the one we'll use when we deploy our chatbot to the wild (even though it comes second in the actual code).
# 
# <img src="images/sequence-to-sequence-inference-decoder.png"/>
# 
# We'll hand our encoder hidden state to the inference decoder and have it process its output. TensorFlow handles most of the logic for us. We just have to use [`tf.contrib.seq2seq.simple_decoder_fn_inference`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/simple_decoder_fn_inference) and [`tf.contrib.seq2seq.dynamic_rnn_decoder`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/dynamic_rnn_decoder) and supply them with the appropriate inputs.
# 
# Notice that the inference decoder feeds the output of each time step as an input to the next.
# 
# As for the training decoder, we can think of it as looking like this:
# <img src="images/sequence-to-sequence-training-decoder.png"/>
# 
# The training decoder **does not** feed the output of each time step to the next. Rather, the inputs to the decoder time steps are the target sequence from the training dataset (the orange letters).

# ### Encoding
# - Embed the input data using [`tf.contrib.layers.embed_sequence`](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/embed_sequence)
# - Pass the embedded input into a stack of RNNs.  Save the RNN state and ignore the output.

# In[39]:

# vocab size
source_vocab_size = len(source_char_to_int)

# Encoder embedding
enc_embed_input = tf.contrib.layers.embed_sequence(inputs_, source_vocab_size, encode_embed_size)
    
# Encoder
enc_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(rnn_size)] * num_layers)
_, enc_state = tf.nn.dynamic_rnn(enc_cell, enc_embed_input, scope='reuse', dtype=tf.float32)


# # STOPPING WORK HERE UNTIL BETTER MATERIALS ARE AVAILABLE

# ### Process Decoding Input

# In[40]:

import numpy as np

### I DON'T UNDERSTAND THIS ####

# Process the input we'll feed to the decoder
ending = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1, 1])
dec_input = tf.concat([tf.fill([batch_size, 1], target_char_to_int['<s>']), ending], 1)

demonstration_outputs = np.reshape(range(batch_size * seq_length), (batch_size, seq_length))

# print Targets and Processed Decoding Input
sess = tf.InteractiveSession()
print("Targets")
print(demonstration_outputs[:2])
print("\n")
print("Processed Decoding Input")
print(sess.run(dec_input, {targets: demonstration_outputs})[:2])


# ### Decoding
# - Embed the decoding input
# - Build the decoding RNNs
# - Build the output layer in the decoding scope, so the weight and bias can be shared between the training and inference decoders.

# In[42]:

target_vocab_size = len(target_char_to_int)

### I DON'T UNDERSTAND THIS ####

# Decoder Embedding
dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

# Decoder RNNs
dec_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(rnn_size)] * num_layers)

with tf.variable_scope("decoding") as decoding_scope:
    # Output Layer
    output_fn = lambda x: tf.contrib.layers.fully_connected(x, target_vocab_size, None, scope=decoding_scope)


# #### Decoder During Training
# - Build the training decoder using [`tf.contrib.seq2seq.simple_decoder_fn_train`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/simple_decoder_fn_train) and [`tf.contrib.seq2seq.dynamic_rnn_decoder`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/dynamic_rnn_decoder).
# - Apply the output layer to the output of the training decoder

# In[12]:


    # Training Decoder

    
    # Apply output function
    
    


# #### Decoder During Inference
# - Reuse the weights the biases from the training decoder using [`tf.variable_scope("decoding", reuse=True)`](https://www.tensorflow.org/api_docs/python/tf/variable_scope)
# - Build the inference decoder using [`tf.contrib.seq2seq.simple_decoder_fn_inference`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/simple_decoder_fn_inference) and [`tf.contrib.seq2seq.dynamic_rnn_decoder`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/dynamic_rnn_decoder).
#  - The output function is applied to the output in this step 

# In[13]:


    # Inference Decoder

    


# ### Optimization
# Our loss function is [`tf.contrib.seq2seq.sequence_loss`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/sequence_loss) provided by the tensor flow seq2seq module. It calculates a weighted cross-entropy loss for the output logits.

# In[14]:

# Loss function



# Optimizer



# Gradient Clipping



# ## Train
# We're now ready to train our model. If you run into OOM (out of memory) issues during training, try to decrease the batch_size.

# In[ ]:





# ## Prediction

# In[16]:

input_sentence = 'hello'



print('Input')
print('  Word Ids:      {}'.format([i for i in input_sentence]))
print('  Input Words: {}'.format([source_int_to_letter[i] for i in input_sentence]))

print('\nPrediction')
print('  Word Ids:      {}'.format([i for i in np.argmax(chatbot_logits, 1)]))
print('  Chatbot Answer Words: {}'.format([target_int_to_letter[i] for i in np.argmax(chatbot_logits, 1)]))

