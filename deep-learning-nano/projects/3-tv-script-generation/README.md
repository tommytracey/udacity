
# TV Script Generation
In this project, you'll generate your own [Simpsons](https://en.wikipedia.org/wiki/The_Simpsons) TV scripts using RNNs.  You'll be using part of the [Simpsons dataset](https://www.kaggle.com/wcukierski/the-simpsons-by-the-data) of scripts from 27 seasons.  The Neural Network you'll build will generate a new TV script for a scene at [Moe's Tavern](https://simpsonswiki.com/wiki/Moe's_Tavern).
## Get the Data
The data is already provided for you.  You'll be using a subset of the original dataset.  It consists of only the scenes in Moe's Tavern.  This doesn't include other versions of the tavern, like "Moe's Cavern", "Flaming Moe's", "Uncle Moe's Family Feed-Bag", etc..


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper

data_dir = './data/simpsons/moes_tavern_lines.txt'
text = helper.load_data(data_dir)
# Ignore notice, since we don't use it for analysing the data
text = text[81:]
```

## Explore the Data
Play around with `view_sentence_range` to view different parts of the data.


```python
view_sentence_range = (0, 10)

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import numpy as np

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))
scenes = text.split('\n\n')
print('Number of scenes: {}'.format(len(scenes)))
sentence_count_scene = [scene.count('\n') for scene in scenes]
print('Average number of sentences in each scene: {}'.format(np.average(sentence_count_scene)))

sentences = [sentence for scene in scenes for sentence in scene.split('\n')]
print('Number of lines: {}'.format(len(sentences)))
word_count_sentence = [len(sentence.split()) for sentence in sentences]
print('Average number of words in each line: {}'.format(np.average(word_count_sentence)))

print()
print('The sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))
```

    Dataset Stats
    Roughly the number of unique words: 11492
    Number of scenes: 262
    Average number of sentences in each scene: 15.248091603053435
    Number of lines: 4257
    Average number of words in each line: 11.50434578341555
    
    The sentences 0 to 10:
    Moe_Szyslak: (INTO PHONE) Moe's Tavern. Where the elite meet to drink.
    Bart_Simpson: Eh, yeah, hello, is Mike there? Last name, Rotch.
    Moe_Szyslak: (INTO PHONE) Hold on, I'll check. (TO BARFLIES) Mike Rotch. Mike Rotch. Hey, has anybody seen Mike Rotch, lately?
    Moe_Szyslak: (INTO PHONE) Listen you little puke. One of these days I'm gonna catch you, and I'm gonna carve my name on your back with an ice pick.
    Moe_Szyslak: What's the matter Homer? You're not your normal effervescent self.
    Homer_Simpson: I got my problems, Moe. Give me another one.
    Moe_Szyslak: Homer, hey, you should not drink to forget your problems.
    Barney_Gumble: Yeah, you should only drink to enhance your social skills.
    
    


## Implement Preprocessing Functions
The first thing to do to any dataset is preprocessing.  Implement the following preprocessing functions below:
- Lookup Table
- Tokenize Punctuation

### Lookup Table
To create a word embedding, you first need to transform the words to ids.  In this function, create two dictionaries:
- Dictionary to go from the words to an id, we'll call `vocab_to_int`
- Dictionary to go from the id to word, we'll call `int_to_vocab`

Return these dictionaries in the following tuple `(vocab_to_int, int_to_vocab)`


```python
import numpy as np
import problem_unittests as tests
from collections import Counter

def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    word_counts = Counter(text)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}
    
    return vocab_to_int, int_to_vocab


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_create_lookup_tables(create_lookup_tables)
```

    Tests Passed


### Tokenize Punctuation
We'll be splitting the script into a word array using spaces as delimiters.  However, punctuations like periods and exclamation marks make it hard for the neural network to distinguish between the word "bye" and "bye!".

Implement the function `token_lookup` to return a dict that will be used to tokenize symbols like "!" into "||Exclamation_Mark||".  Create a dictionary for the following symbols where the symbol is the key and value is the token:
- Period ( . )
- Comma ( , )
- Quotation Mark ( " )
- Semicolon ( ; )
- Exclamation mark ( ! )
- Question mark ( ? )
- Left Parentheses ( ( )
- Right Parentheses ( ) )
- Dash ( -- )
- Return ( \n )

This dictionary will be used to token the symbols and add the delimiter (space) around it.  This separates the symbols as it's own word, making it easier for the neural network to predict on the next word. Make sure you don't use a token that could be confused as a word. Instead of using the token "dash", try using something like "||dash||".


```python
def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenize dictionary where the key is the punctuation and the value is the token
    """
    token_dict = {'.':'||PERIOD||',
   ',':'||COMMA||',
   '"':'||QUOTATION_MARK||',
   ';':'||SEMICOLON||',
   '!':'||EXCLAMATION_MARK||',
   '?':'||QUESTION_MARK||',
   '(':'||LEFT_PAREN||',
   ')':'||RIGHT_PAREN||',
   '--':'||HYPHENS||',
   '?':'||QUESTION_MARK||',
   '\n':'||NEW_LINE||'}

    return token_dict

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_tokenize(token_lookup)
```

    Tests Passed


## Preprocess all the data and save it
Running the code cell below will preprocess all the data and save it to file.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# Preprocess Training, Validation, and Testing Data
helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)
```

# Check Point
This is your first checkpoint. If you ever decide to come back to this notebook or have to restart the notebook, you can start from here. The preprocessed data has been saved to disk.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper
import numpy as np
import problem_unittests as tests

int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
```

## Build the Neural Network
You'll build the components necessary to build a RNN by implementing the following functions below:
- get_inputs
- get_init_cell
- get_embed
- build_rnn
- build_nn
- get_batches

### Check the Version of TensorFlow and Access to GPU


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from distutils.version import LooseVersion
import warnings
import tensorflow as tf

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
```

    TensorFlow Version: 1.0.0
    Default GPU Device: /gpu:0


### Input
Implement the `get_inputs()` function to create TF Placeholders for the Neural Network.  It should create the following placeholders:
- Input text placeholder named "input" using the [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder) `name` parameter.
- Targets placeholder
- Learning Rate placeholder

Return the placeholders in the following the tuple `(Input, Targets, LearingRate)`


```python
def get_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate)
    """
    Input = tf.placeholder(tf.int32, [None, None], name='input')
    Targets = tf.placeholder(tf.int32, [None, None], name='targets')
    LearningRate = tf.placeholder(tf.float32, name='LearningRate')

    return (Input, Targets, LearningRate)

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_inputs(get_inputs)
```

    Tests Passed


### Build RNN Cell and Initialize
Stack one or more [`BasicLSTMCells`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/BasicLSTMCell) in a [`MultiRNNCell`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/MultiRNNCell).
- The Rnn size should be set using `rnn_size`
- Initalize Cell State using the MultiRNNCell's [`zero_state()`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/MultiRNNCell#zero_state) function
    - Apply the name "initial_state" to the initial state using [`tf.identity()`](https://www.tensorflow.org/api_docs/python/tf/identity)

Return the cell and initial state in the following tuple `(Cell, InitialState)`


```python
def get_init_cell(batch_size, rnn_size):
    """
    Create an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """
    # Stack RNN cells
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=0.6)
    Cell = tf.contrib.rnn.MultiRNNCell([drop] * 2)  # num_layers = 2

    # Initialize cell state
    InitialState = tf.identity(Cell.zero_state(batch_size, tf.float32), name='initial_state')

    return (Cell, InitialState)

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_init_cell(get_init_cell)
```

    Tests Passed


### Word Embedding
Apply embedding to `input_data` using TensorFlow.  Return the embedded sequence.


```python
def get_embed(input_data, vocab_size, embed_dim):
    """
    Create embedding for <input_data>.
    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions
    :return: Embedded input.
    """
    embedding = tf.Variable(tf.random_uniform((vocab_size, embed_dim), -1, 1))
    
    return tf.nn.embedding_lookup(embedding, input_data)

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_embed(get_embed)
```

    Tests Passed


### Build RNN
You created a RNN Cell in the `get_init_cell()` function.  Time to use the cell to create a RNN.
- Build the RNN using the [`tf.nn.dynamic_rnn()`](https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn)
 - Apply the name "final_state" to the final state using [`tf.identity()`](https://www.tensorflow.org/api_docs/python/tf/identity)

Return the outputs and final_state state in the following tuple `(Outputs, FinalState)` 


```python
def build_rnn(cell, inputs):
    """
    Create a RNN using a RNN Cell
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """
    Outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
    FinalState = tf.identity(state, name='final_state')
    
    return (Outputs, FinalState)

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_build_rnn(build_rnn)
```

    Tests Passed


### Build the Neural Network
Apply the functions you implemented above to:
- Apply embedding to `input_data` using your `get_embed(input_data, vocab_size, embed_dim)` function.
- Build RNN using `cell` and your `build_rnn(cell, inputs)` function.
- Apply a fully connected layer with a linear activation and `vocab_size` as the number of outputs.

Return the logits and final state in the following tuple (Logits, FinalState) 


```python
def build_nn(cell, rnn_size, input_data, vocab_size):
    """
    Build part of the neural network
    :param cell: RNN cell
    :param rnn_size: Size of rnns
    :param input_data: Input data
    :param vocab_size: Vocabulary size
    :return: Tuple (Logits, FinalState)
    """
    # Apply embedding and build RNN
    embed = get_embed(input_data, vocab_size, embed_dim=rnn_size)
    Outputs, FinalState = build_rnn(cell, embed)

    # Apply fully connected layer with linear activation
    Logits = tf.contrib.layers.fully_connected(Outputs, vocab_size, activation_fn=None)
    
    return (Logits, FinalState)

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_build_nn(build_nn)
```

    Tests Passed


### Batches
Implement `get_batches` to create batches of input and targets using `int_text`.  The batches should be a Numpy array with the shape `(number of batches, 2, batch size, sequence length)`. Each batch contains two elements:
- The first element is a single batch of **input** with the shape `[batch size, sequence length]`
- The second element is a single batch of **targets** with the shape `[batch size, sequence length]`

If you can't fill the last batch with enough data, drop the last batch.

For exmple, `get_batches([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 2, 3)` would return a Numpy array of the following:
```
[
  # First Batch
  [
    # Batch of Input
    [[ 1  2  3], [ 7  8  9]],
    # Batch of targets
    [[ 2  3  4], [ 8  9 10]]
  ],
 
  # Second Batch
  [
    # Batch of Input
    [[ 4  5  6], [10 11 12]],
    # Batch of targets
    [[ 5  6  7], [11 12 13]]
  ]
]
```


```python
def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    # Calculate number of batches
    n_batches = int(len(int_text) / (batch_size * seq_length))

    # Drop the last few words to make only full batches
    xdata = np.array(int_text[: n_batches * batch_size * seq_length])
    ydata = np.array(int_text[1: n_batches * batch_size * seq_length + 1])
    
    # Split data into batches
    x_batches = np.split(xdata.reshape(batch_size, -1), n_batches, 1)
    y_batches = np.split(ydata.reshape(batch_size, -1), n_batches, 1)

    return np.array(list(zip(x_batches, y_batches)))

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_batches(get_batches)
```

    Tests Passed


## Neural Network Training
### Hyperparameters
Tune the following parameters:

- Set `num_epochs` to the number of epochs.
- Set `batch_size` to the batch size.
- Set `rnn_size` to the size of the RNNs.
- Set `seq_length` to the length of sequence.
- Set `learning_rate` to the learning rate.
- Set `show_every_n_batches` to the number of batches the neural network should print progress.


```python
# Number of Epochs
num_epochs = 200
# Batch Size
batch_size = 128
# RNN Size
rnn_size = 1024
# Sequence Length
seq_length = 11
# Learning Rate
learning_rate = 0.001
# Show stats for every n number of batches
show_every_n_batches = 98

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
save_dir = './save'
```

### Vocab set utilization


```python
word_count = len(int_text)
print("Words in dataset: ", word_count)

batch_count = len(int_text) // (batch_size * seq_length)
print("Batch count: ", batch_count)

words_used = batch_count * batch_size * seq_length
print("Words used in model: ", words_used)

unused_words = word_count - words_used
print("Unused words: ", unused_words)

utilization = words_used / word_count
print("Utilization: ", '{:.3f}'.format(utilization))
```

    Words in dataset:  69100
    Batch count:  49
    Words used in model:  68992
    Unused words:  108
    Utilization:  0.998


### Build the Graph
Build the graph using the neural network you implemented.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from tensorflow.contrib import seq2seq

train_graph = tf.Graph()
with train_graph.as_default():
    vocab_size = len(int_to_vocab)
    input_text, targets, lr = get_inputs()
    input_data_shape = tf.shape(input_text)
    cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
    logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size)

    # Probabilities for generating words
    probs = tf.nn.softmax(logits, name='probs')

    # Loss function
    cost = seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_data_shape[0], input_data_shape[1]]))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(lr)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
    train_op = optimizer.apply_gradients(capped_gradients)
```

## Train
Train the neural network on the preprocessed data.  If you have a hard time getting a good loss, check the [forms](https://discussions.udacity.com/) to see if anyone is having the same problem.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
batches = get_batches(int_text, batch_size, seq_length)

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})

        for batch_i, (x, y) in enumerate(batches):
            feed = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate}
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

            # Show every <show_every_n_batches> batches
            if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    len(batches),
                    train_loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_dir)
    print('Model Trained and Saved')
```

    Epoch   0 Batch    0/49   train_loss = 8.822
    Epoch   2 Batch    0/49   train_loss = 5.003
    Epoch   4 Batch    0/49   train_loss = 4.578
    Epoch   6 Batch    0/49   train_loss = 4.308
    Epoch   8 Batch    0/49   train_loss = 3.972
    Epoch  10 Batch    0/49   train_loss = 3.672
    Epoch  12 Batch    0/49   train_loss = 3.348
    Epoch  14 Batch    0/49   train_loss = 3.001
    Epoch  16 Batch    0/49   train_loss = 2.673
    Epoch  18 Batch    0/49   train_loss = 2.343
    Epoch  20 Batch    0/49   train_loss = 2.082
    Epoch  22 Batch    0/49   train_loss = 1.801
    Epoch  24 Batch    0/49   train_loss = 1.572
    Epoch  26 Batch    0/49   train_loss = 1.320
    Epoch  28 Batch    0/49   train_loss = 1.143
    Epoch  30 Batch    0/49   train_loss = 1.039
    Epoch  32 Batch    0/49   train_loss = 0.910
    Epoch  34 Batch    0/49   train_loss = 0.820
    Epoch  36 Batch    0/49   train_loss = 0.728
    Epoch  38 Batch    0/49   train_loss = 0.652
    Epoch  40 Batch    0/49   train_loss = 0.590
    Epoch  42 Batch    0/49   train_loss = 0.555
    Epoch  44 Batch    0/49   train_loss = 0.530
    Epoch  46 Batch    0/49   train_loss = 0.500
    Epoch  48 Batch    0/49   train_loss = 0.487
    Epoch  50 Batch    0/49   train_loss = 0.468
    Epoch  52 Batch    0/49   train_loss = 0.466
    Epoch  54 Batch    0/49   train_loss = 0.479
    Epoch  56 Batch    0/49   train_loss = 0.437
    Epoch  58 Batch    0/49   train_loss = 0.426
    Epoch  60 Batch    0/49   train_loss = 0.433
    Epoch  62 Batch    0/49   train_loss = 0.426
    Epoch  64 Batch    0/49   train_loss = 0.431
    Epoch  66 Batch    0/49   train_loss = 0.408
    Epoch  68 Batch    0/49   train_loss = 0.400
    Epoch  70 Batch    0/49   train_loss = 0.402
    Epoch  72 Batch    0/49   train_loss = 0.402
    Epoch  74 Batch    0/49   train_loss = 0.415
    Epoch  76 Batch    0/49   train_loss = 0.422
    Epoch  78 Batch    0/49   train_loss = 0.398
    Epoch  80 Batch    0/49   train_loss = 0.406
    Epoch  82 Batch    0/49   train_loss = 0.410
    Epoch  84 Batch    0/49   train_loss = 0.392
    Epoch  86 Batch    0/49   train_loss = 0.382
    Epoch  88 Batch    0/49   train_loss = 0.389
    Epoch  90 Batch    0/49   train_loss = 0.393
    Epoch  92 Batch    0/49   train_loss = 0.415
    Epoch  94 Batch    0/49   train_loss = 0.405
    Epoch  96 Batch    0/49   train_loss = 0.386
    Epoch  98 Batch    0/49   train_loss = 0.396
    Epoch 100 Batch    0/49   train_loss = 0.395
    Epoch 102 Batch    0/49   train_loss = 0.390
    Epoch 104 Batch    0/49   train_loss = 0.388
    Epoch 106 Batch    0/49   train_loss = 0.399
    Epoch 108 Batch    0/49   train_loss = 0.385
    Epoch 110 Batch    0/49   train_loss = 0.394
    Epoch 112 Batch    0/49   train_loss = 0.383
    Epoch 114 Batch    0/49   train_loss = 0.389
    Epoch 116 Batch    0/49   train_loss = 0.387
    Epoch 118 Batch    0/49   train_loss = 0.386
    Epoch 120 Batch    0/49   train_loss = 0.374
    Epoch 122 Batch    0/49   train_loss = 0.378
    Epoch 124 Batch    0/49   train_loss = 0.376
    Epoch 126 Batch    0/49   train_loss = 0.388
    Epoch 128 Batch    0/49   train_loss = 0.379
    Epoch 130 Batch    0/49   train_loss = 0.373
    Epoch 132 Batch    0/49   train_loss = 0.388
    Epoch 134 Batch    0/49   train_loss = 0.374
    Epoch 136 Batch    0/49   train_loss = 0.376
    Epoch 138 Batch    0/49   train_loss = 0.383
    Epoch 140 Batch    0/49   train_loss = 0.381
    Epoch 142 Batch    0/49   train_loss = 0.375
    Epoch 144 Batch    0/49   train_loss = 0.391
    Epoch 146 Batch    0/49   train_loss = 0.386
    Epoch 148 Batch    0/49   train_loss = 0.371
    Epoch 150 Batch    0/49   train_loss = 0.374
    Epoch 152 Batch    0/49   train_loss = 0.373
    Epoch 154 Batch    0/49   train_loss = 0.368
    Epoch 156 Batch    0/49   train_loss = 0.359
    Epoch 158 Batch    0/49   train_loss = 0.372
    Epoch 160 Batch    0/49   train_loss = 0.386
    Epoch 162 Batch    0/49   train_loss = 0.373
    Epoch 164 Batch    0/49   train_loss = 0.365
    Epoch 166 Batch    0/49   train_loss = 0.386
    Epoch 168 Batch    0/49   train_loss = 0.356
    Epoch 170 Batch    0/49   train_loss = 0.372
    Epoch 172 Batch    0/49   train_loss = 0.375
    Epoch 174 Batch    0/49   train_loss = 0.366
    Epoch 176 Batch    0/49   train_loss = 0.371
    Epoch 178 Batch    0/49   train_loss = 0.371
    Epoch 180 Batch    0/49   train_loss = 0.376
    Epoch 182 Batch    0/49   train_loss = 0.368
    Epoch 184 Batch    0/49   train_loss = 0.367
    Epoch 186 Batch    0/49   train_loss = 0.363
    Epoch 188 Batch    0/49   train_loss = 0.376
    Epoch 190 Batch    0/49   train_loss = 0.372
    Epoch 192 Batch    0/49   train_loss = 0.367
    Epoch 194 Batch    0/49   train_loss = 0.364
    Epoch 196 Batch    0/49   train_loss = 0.360
    Epoch 198 Batch    0/49   train_loss = 0.362
    Model Trained and Saved


## Save Parameters
Save `seq_length` and `save_dir` for generating a new TV script.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# Save parameters for checkpoint
helper.save_params((seq_length, save_dir))
```

# Checkpoint


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import tensorflow as tf
import numpy as np
import helper
import problem_unittests as tests

_, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
seq_length, load_dir = helper.load_params()
```

## Implement Generate Functions
### Get Tensors
Get tensors from `loaded_graph` using the function [`get_tensor_by_name()`](https://www.tensorflow.org/api_docs/python/tf/Graph#get_tensor_by_name).  Get the tensors using the following names:
- "input:0"
- "initial_state:0"
- "final_state:0"
- "probs:0"

Return the tensors in the following tuple `(InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)` 


```python
def get_tensors(loaded_graph):
    """
    Get input, initial state, final state, and probabilities tensor from <loaded_graph>
    :param loaded_graph: TensorFlow graph loaded from file
    :return: Tuple (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)
    """
    InputTensor = tf.Graph.get_tensor_by_name(loaded_graph, 'input:0')
    InitialStateTensor = tf.Graph.get_tensor_by_name(loaded_graph, 'initial_state:0')
    FinalStateTensor = tf.Graph.get_tensor_by_name(loaded_graph, 'final_state:0')
    ProbsTensor = tf.Graph.get_tensor_by_name(loaded_graph, 'probs:0')

    return (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_tensors(get_tensors)
```

    Tests Passed


### Choose Word
Implement the `pick_word()` function to select the next word using `probabilities`.


```python
def pick_word(probabilities, int_to_vocab):
    """
    Pick the next word in the generated text
    :param probabilities: Probabilites of the next word
    :param int_to_vocab: Dictionary of word ids as the keys and words as the values
    :return: String of the predicted word
    """
    top_n=10
    prob = np.squeeze(probabilities)
    prob[np.argsort(prob)[:-top_n]] = 0
    prob = prob / np.sum(prob)
    c = np.random.choice(len(int_to_vocab), 1, p=prob)[0]
    word = int_to_vocab[c]
    return word

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_pick_word(pick_word)
```

    Tests Passed


## Generate TV Script
This will generate the TV script for you.  Set `gen_length` to the length of TV script you want to generate.


```python
gen_length = 200
# homer_simpson, moe_szyslak, or Barney_Gumble
prime_word = 'moe_szyslak'

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(load_dir + '.meta')
    loader.restore(sess, load_dir)

    # Get Tensors from loaded model
    input_text, initial_state, final_state, probs = get_tensors(loaded_graph)

    # Sentences generation setup
    gen_sentences = [prime_word + ':']
    prev_state = sess.run(initial_state, {input_text: np.array([[1]])})

    # Generate sentences
    for n in range(gen_length):
        # Dynamic Input
        dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
        dyn_seq_length = len(dyn_input[0])

        # Get Prediction
        probabilities, prev_state = sess.run(
            [probs, final_state],
            {input_text: dyn_input, initial_state: prev_state})
        
        pred_word = pick_word(probabilities[dyn_seq_length-1], int_to_vocab)

        gen_sentences.append(pred_word)
    
    # Remove tokens
    tv_script = ' '.join(gen_sentences)
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        tv_script = tv_script.replace(' ' + token.lower(), key)
    tv_script = tv_script.replace('\n ', '\n')
    tv_script = tv_script.replace('( ', '(')
        
    print(tv_script)
```

    moe_szyslak: oh, what? i suppose you've seen a bigger star enjoy the good around once.
    moe_szyslak: ah, then those where's her magic stools?
    lisa_simpson: i learned something much to get outta outta with no god, i couldn't help those with my fortune back is come and arm-pittish her now,.
    apu_nahasapeemapetilon:(disgusted) what?! you know what are men with you tonight, eh?
    chief_wiggum:(chief_wiggum:) oh, there's one of that new change to hell.
    mayor_joe_quimby:...(to self) she's so hard. yesterday, you'll get me with the way to me you still dump?
    barney_gumble: huh?(hopeful) i know, and cheer, gentlemen. i got! we're going to pull with their restaurant dive.
    artie_ziff:(excited) what? they larry: shut up that she did.
    moe_szyslak:(big sobs) i'd be back.
    homer_simpson: nah, she's time for turn some while where they made a bars.
    lenny_leonard: yeah, i feel. we're gold with he reminds


# The TV Script is Nonsensical
It's ok if the TV script doesn't make any sense.  We trained on less than a megabyte of text.  In order to get good results, you'll have to use a smaller vocabulary or get more data.  Luckly there's more data!  As we mentioned in the begging of this project, this is a subset of [another dataset](https://www.kaggle.com/wcukierski/the-simpsons-by-the-data).  We didn't have you train on all the data, because that would take too long.  However, you are free to train your neural network on all the data.  After you complete the project, of course.
# Submitting This Project
When submitting this project, make sure to run all the cells before saving the notebook. Save the notebook file as "dlnd_tv_script_generation.ipynb" and save it as a HTML file under "File" -> "Download as". Include the "helper.py" and "problem_unittests.py" files in your submission.
