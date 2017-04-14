
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

# In[1]:




# Let's start by examining the current state of the dataset. `source_sentences` contains the entire input sequence file as text delimited by newline symbols.

# In[2]:




# `source_sentences` contains the entire output sequence file as text delimited by newline symbols.  Each line corresponds to the line from `source_sentences`.  `source_sentences` contains a sorted characters of the line.

# In[3]:




# ## Preprocess
# To do anything useful with it, we'll need to turn the characters into a list of integers: 

# In[4]:

def extract_character_vocab(data):


# Build int2letter and letter2int dicts


# Convert characters to ids


print("Example source sequence")
print(source_letter_ids[:3])
print("\n")
print("Example target sequence")
print(target_letter_ids[:3])


# The last step in the preprocessing stage is to determine the the longest sequence size in the dataset we'll be using, then pad all the sequences to that length.

# In[5]:

def pad_id_sequences(source_ids, source_letter_to_int, target_ids, target_letter_to_int, sequence_length):


# Use the longest sequence as sequence length


# Pad all sequences up to sequence length


print("Sequence Length")
print(sequence_length)
print("\n")
print("Input sequence example")
print(source_ids[:3])
print("\n")
print("Target sequence example")
print(target_ids[:3])


# This is the final shape we need them to be in. We can now proceed to building the model.

# ## Model
# #### Check the Version of TensorFlow
# This will check to make sure you have the correct version of TensorFlow

# In[6]:

from distutils.version import LooseVersion
import tensorflow as tf

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))


# ### Hyperparameters

# In[7]:

# Number of Epochs
epochs = 
# Batch Size
batch_size = 
# RNN Size
rnn_size = 
# Number of Layers
num_layers = 
# Embedding Size
encoding_embedding_size = 
decoding_embedding_size = 
# Learning Rate
learning_rate = 


# ### Input

# In[8]:

# set placeholders



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

# In[9]:

# vocab size


# Encoder embedding


# Encoder



# ### Process Decoding Input

# In[10]:

import numpy as np

# Process the input we'll feed to the decoder


# print Targets and Processed Decoding Input



# ### Decoding
# - Embed the decoding input
# - Build the decoding RNNs
# - Build the output layer in the decoding scope, so the weight and bias can be shared between the training and inference decoders.

# In[11]:

target_vocab_size = len(target_letter_to_int)

# Decoder Embedding


# Decoder RNNs


    # Output Layer



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


# test change 6