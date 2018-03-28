### Deep Learning Foundations Nanodegree
# Project: Language Translation

---
# Results
The sections below outline the work I completed as part of this project. The Jupyter Notebook document containing the source code is located [here](https://github.com/tommytracey/udacity/blob/master/deep-learning-nano/projects/4-language-translation/dlnd_language_translation-v2.ipynb).

## Overview
In this project, we take a peek into the realm of neural network machine translation. We’ll be training a sequence to sequence model on a dataset of English and French sentences that can translate new sentences from English to French.

## Get the Data
Since translating the whole language of English to French will take lots of time to train, Udacity has provided us with a small portion of the English corpus.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper
import problem_unittests as tests

source_path = 'data/small_vocab_en'
target_path = 'data/small_vocab_fr'
source_text = helper.load_data(source_path)
target_text = helper.load_data(target_path)
```

## Explore the Data
Play around with view_sentence_range to view different parts of the data.


```python
view_sentence_range = (11, 25)

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import numpy as np

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in source_text.split()})))

sentences = source_text.split('\n')
word_counts = [len(sentence.split()) for sentence in sentences]
print('Number of sentences: {}'.format(len(sentences)))
print('Average number of words in a sentence: {}'.format(np.average(word_counts)))

print()
print('English sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(source_text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))
print()
print('French sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(target_text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))
```

    Dataset Stats
    Roughly the number of unique words: 227
    Number of sentences: 137861
    Average number of words in a sentence: 13.225277634719028

    English sentences 11 to 25:
    he saw a old yellow truck .
    india is rainy during june , and it is sometimes warm in november .
    that cat was my most loved animal .
    he dislikes grapefruit , limes , and lemons .
    her least liked fruit is the lemon , but his least liked is the grapefruit .
    california is never cold during february , but it is sometimes freezing in june .
    china is usually pleasant during autumn , and it is usually quiet in october .
    paris is never freezing during november , but it is wonderful in october .
    the united states is never rainy during january , but it is sometimes mild in october .
    china is usually pleasant during november , and it is never quiet in october .
    the united states is never nice during february , but it is sometimes pleasant in april .
    india is never busy during autumn , and it is mild in spring .
    paris is mild during summer , but it is usually busy in april .
    france is never cold during september , and it is snowy in october .

    French sentences 11 to 25:
    il a vu un vieux camion jaune .
    inde est pluvieux en juin , et il est parfois chaud en novembre .
    ce chat était mon animal le plus aimé .
    il n'aime pamplemousse , citrons verts et les citrons .
    son fruit est moins aimé le citron , mais son moins aimé est le pamplemousse .
    californie ne fait jamais froid en février , mais il est parfois le gel en juin .
    chine est généralement agréable en automne , et il est généralement calme en octobre .
    paris est jamais le gel en novembre , mais il est merveilleux en octobre .
    les états-unis est jamais pluvieux en janvier , mais il est parfois doux en octobre .
    chine est généralement agréable en novembre , et il est jamais tranquille en octobre .
    les états-unis est jamais agréable en février , mais il est parfois agréable en avril .
    l' inde est jamais occupé à l'automne , et il est doux au printemps .
    paris est doux pendant l' été , mais il est généralement occupé en avril .
    france ne fait jamais froid en septembre , et il est neigeux en octobre .



```python
sentences[:10]
```




    ['new jersey is sometimes quiet during autumn , and it is snowy in april .',
     'the united states is usually chilly during july , and it is usually freezing in november .',
     'california is usually quiet during march , and it is usually hot in june .',
     'the united states is sometimes mild during june , and it is cold in september .',
     'your least liked fruit is the grape , but my least liked is the apple .',
     'his favorite fruit is the orange , but my favorite is the grape .',
     'paris is relaxing during december , but it is usually chilly in july .',
     'new jersey is busy during spring , and it is never hot in march .',
     'our least liked fruit is the lemon , but my least liked is the grape .',
     'the united states is sometimes busy during january , and it is sometimes warm in november .']



## Implement Preprocessing Function
### Text to Word Ids
As you did with other RNNs, you must turn the text into a number so the computer can understand it. In the function `text_to_ids()`, you'll turn `source_text` and `target_text` from words to ids.  However, you need to add the `<EOS>` word id at the end of each sentence from `target_text`.  This will help the neural network predict when the sentence should end.

You can get the `<EOS>` word id by doing:
```python
target_vocab_to_int['<EOS>']
```
You can get other word ids using `source_vocab_to_int` and `target_vocab_to_int`.


```python
def text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int):
    """
    Convert source and target text to proper word ids
    :param source_text: String that contains all the source text.
    :param target_text: String that contains all the target text.
    :param source_vocab_to_int: Dictionary to go from the source words to an id
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: A tuple of lists (source_id_text, target_id_text)
    """

    x = [[source_vocab_to_int.get(word, 0) for word in sentence.split()] for sentence in source_text.split('\n')]
    y = [[target_vocab_to_int.get(word, 0) for word in sentence.split()] for sentence in target_text.split('\n')]

    source_id_text = []
    target_id_text = []

    for i in range(len(x)):
        source_id_text.append(x[i])
        target_id_text.append(y[i] + [target_vocab_to_int['<EOS>']])

    return (source_id_text, target_id_text)

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_text_to_ids(text_to_ids)
```

    Tests Passed


### Preprocess all the data and save it
Running the code cell below will preprocess all the data and save it to file.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
helper.preprocess_and_save_data(source_path, target_path, text_to_ids)
```

# Check Point
This is your first checkpoint. If you ever decide to come back to this notebook or have to restart the notebook, you can start from here. The preprocessed data has been saved to disk.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import numpy as np
import helper

(source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = helper.load_preprocess()
```

### Check the Version of TensorFlow and Access to GPU
This will check to make sure you have the correct version of TensorFlow and access to a GPU


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


## Build the Neural Network
You'll build the components necessary to build a Sequence-to-Sequence model by implementing the following functions below:
- `model_inputs`
- `process_decoding_input`
- `encoding_layer`
- `decoding_layer_train`
- `decoding_layer_infer`
- `decoding_layer`
- `seq2seq_model`

### Input
Implement the `model_inputs()` function to create TF Placeholders for the Neural Network. It should create the following placeholders:

- Input text placeholder named "input" using the TF Placeholder name parameter with rank 2.
- Targets placeholder with rank 2.
- Learning rate placeholder with rank 0.
- Keep probability placeholder named "keep_prob" using the TF Placeholder name parameter with rank 0.

Return the placeholders in the following the tuple (Input, Targets, Learing Rate, Keep Probability)


```python
def model_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate, keep probability)
    """

    input_ = tf.placeholder(tf.int32, [None, None], name="input")
    targets = tf.placeholder(tf.int32, [None, None], name="targets")
    learning_rate = tf.placeholder(tf.float32, name="learning_rate")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    return (input_, targets, learning_rate, keep_prob)

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_model_inputs(model_inputs)
```

    Tests Passed


### Process Decoding Input
Implement `process_decoding_input` using TensorFlow to remove the last word id from each batch in `target_data` and concat the GO ID to the begining of each batch.


```python
def process_decoding_input(target_data, target_vocab_to_int, batch_size):
    """
    Preprocess target data for decoding
    :param target_data: Target Placehoder
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :param batch_size: Batch Size
    :return: Preprocessed target data
    """

    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    dec_input = tf.concat([tf.fill([batch_size, 1], target_vocab_to_int['<GO>']), ending], 1)

    return dec_input

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_process_decoding_input(process_decoding_input)
```

    Tests Passed


### Encoding
Implement `encoding_layer()` to create a Encoder RNN layer using [`tf.nn.dynamic_rnn()`](https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn).


```python
def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob):
    """
    Create encoding layer
    :param rnn_inputs: Inputs for the RNN
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param keep_prob: Dropout keep probability
    :return: RNN state
    """

    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    dropout = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell([dropout] * num_layers)
    outputs, encoder_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=None, dtype=tf.float32)

    return encoder_state

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_encoding_layer(encoding_layer)
```

    Tests Passed


### Decoding - Training
Create training logits using [`tf.contrib.seq2seq.simple_decoder_fn_train()`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/simple_decoder_fn_train) and [`tf.contrib.seq2seq.dynamic_rnn_decoder()`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/dynamic_rnn_decoder).  Apply the `output_fn` to the [`tf.contrib.seq2seq.dynamic_rnn_decoder()`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/dynamic_rnn_decoder) outputs.


```python
def decoding_layer_train(encoder_state, dec_cell, dec_embed_input, sequence_length, decoding_scope,
                         output_fn, keep_prob):
    """
    Create a decoding layer for training
    :param encoder_state: Encoder State
    :param dec_cell: Decoder RNN Cell
    :param dec_embed_input: Decoder embedded input
    :param sequence_length: Sequence Length
    :param decoding_scope: TenorFlow Variable Scope for decoding
    :param output_fn: Function to apply the output layer
    :param keep_prob: Dropout keep probability
    :return: Train Logits
    """

    # Training decoder
    train_dec_fn = tf.contrib.seq2seq.simple_decoder_fn_train(encoder_state)
    dropout = tf.contrib.rnn.DropoutWrapper(dec_cell, output_keep_prob=keep_prob)
    train_pred, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(dropout, train_dec_fn, dec_embed_input, \
                                                             sequence_length, scope=decoding_scope)

    # Apply output function
    train_logits = output_fn(train_pred)

    return train_logits


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_decoding_layer_train(decoding_layer_train)
```

    Tests Passed


### Decoding - Inference
Create inference logits using [`tf.contrib.seq2seq.simple_decoder_fn_inference()`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/simple_decoder_fn_inference) and [`tf.contrib.seq2seq.dynamic_rnn_decoder()`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/dynamic_rnn_decoder).


```python
def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, end_of_sequence_id,
                         maximum_length, vocab_size, decoding_scope, output_fn, keep_prob):
    """
    Create a decoding layer for inference
    :param encoder_state: Encoder state
    :param dec_cell: Decoder RNN Cell
    :param dec_embeddings: Decoder embeddings
    :param start_of_sequence_id: GO ID
    :param end_of_sequence_id: EOS Id
    :param maximum_length: Maximum length of
    :param vocab_size: Size of vocabulary
    :param decoding_scope: TensorFlow Variable Scope for decoding
    :param output_fn: Function to apply the output layer
    :param keep_prob: Dropout keep probability
    :return: Inference Logits
    """

    infer_dec_fn = tf.contrib.seq2seq.simple_decoder_fn_inference(output_fn, encoder_state, dec_embeddings, \
                                                                  start_of_sequence_id, end_of_sequence_id, \
                                                                  maximum_length, vocab_size)
    infer_logits, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell, infer_dec_fn, scope=decoding_scope)

    return infer_logits


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_decoding_layer_infer(decoding_layer_infer)
```

    Tests Passed


### Build the Decoding Layer
Implement `decoding_layer()` to create a Decoder RNN layer.

- Create RNN cell for decoding using `rnn_size` and `num_layers`.
- Create the output fuction using [`lambda`](https://docs.python.org/3/tutorial/controlflow.html#lambda-expressions) to transform it's input, logits, to class logits.
- Use the your `decoding_layer_train(encoder_state, dec_cell, dec_embed_input, sequence_length, decoding_scope, output_fn, keep_prob)` function to get the training logits.
- Use your `decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, end_of_sequence_id, maximum_length, vocab_size, decoding_scope, output_fn, keep_prob)` function to get the inference logits.

Note: You'll need to use [tf.variable_scope](https://www.tensorflow.org/api_docs/python/tf/variable_scope) to share variables between training and inference.


```python
def decoding_layer(dec_embed_input, dec_embeddings, encoder_state, vocab_size, sequence_length, rnn_size,
                   num_layers, target_vocab_to_int, keep_prob):
    """
    Create decoding layer
    :param dec_embed_input: Decoder embedded input
    :param dec_embeddings: Decoder embeddings
    :param encoder_state: The encoded state
    :param vocab_size: Size of vocabulary
    :param sequence_length: Sequence Length
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :param keep_prob: Dropout keep probability
    :return: Tuple of (Training Logits, Inference Logits)
    """

    # Sequence variables
    start_of_sequence_id = target_vocab_to_int['<GO>']
    end_of_sequence_id = target_vocab_to_int['<EOS>']

    # RNN cell
    dec_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(rnn_size)] * num_layers)

    with tf.variable_scope("decoding") as decoding_scope:
        # Output function
        output_fn = lambda x: tf.contrib.layers.fully_connected(x, vocab_size, None, scope=decoding_scope)

    with tf.variable_scope("decoding") as decoding_scope:
        # Use decoding_layer_train() to get training logits
        train_logits = decoding_layer_train(encoder_state, dec_cell, dec_embed_input, sequence_length, \
                                            decoding_scope, output_fn, keep_prob)
    with tf.variable_scope("decoding", reuse=True) as decoding_scope:
        # Use decoding_layer_infer() to get inference logits
        infer_logits = decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, \
                                            end_of_sequence_id, sequence_length, vocab_size, decoding_scope, \
                                            output_fn, keep_prob)

    return (train_logits, infer_logits)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_decoding_layer(decoding_layer)
```

    Tests Passed


### Build the Neural Network
Apply the functions you implemented above to:

- Apply embedding to the input data for the encoder.
- Encode the input using your `encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob)`.
- Process target data using your `process_decoding_input(target_data, target_vocab_to_int, batch_size)` function.
- Apply embedding to the target data for the decoder.
- Decode the encoded input using your `decoding_layer(dec_embed_input, dec_embeddings, encoder_state, vocab_size, sequence_length, rnn_size, num_layers, target_vocab_to_int, keep_prob)`.


```python
def seq2seq_model(input_data, target_data, keep_prob, batch_size, sequence_length, source_vocab_size, target_vocab_size,
                  enc_embedding_size, dec_embedding_size, rnn_size, num_layers, target_vocab_to_int):
    """
    Build the Sequence-to-Sequence part of the neural network
    :param input_data: Input placeholder
    :param target_data: Target placeholder
    :param keep_prob: Dropout keep probability placeholder
    :param batch_size: Batch Size
    :param sequence_length: Sequence Length
    :param source_vocab_size: Source vocabulary size
    :param target_vocab_size: Target vocabulary size
    :param enc_embedding_size: Decoder embedding size
    :param dec_embedding_size: Encoder embedding size
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: Tuple of (Training Logits, Inference Logits)
    """

    # Apply embedding to input data for encoder
    enc_embed_input = tf.contrib.layers.embed_sequence(input_data, source_vocab_size, enc_embedding_size)

    # Encode input using encoding_layer()
    encoder_state = encoding_layer(enc_embed_input, rnn_size, num_layers, keep_prob)

    # Process targets using process_decoding_input()
    dec_input = process_decoding_input(target_data, target_vocab_to_int, batch_size)

    # Apply embedding to the target data for decoder
    dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, dec_embedding_size]))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

    # Decode the encoded input using decoding_layer()
    logits = decoding_layer(dec_embed_input, dec_embeddings, encoder_state, target_vocab_size, \
                               sequence_length, rnn_size, num_layers, target_vocab_to_int, keep_prob)

    return logits


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_seq2seq_model(seq2seq_model)
```

    Tests Passed


## Neural Network Training
### Hyperparameters
Tune the following parameters:

- Set `epochs` to the number of epochs.
- Set `batch_size` to the batch size.
- Set `rnn_size` to the size of the RNNs.
- Set `num_layers` to the number of layers.
- Set `encoding_embedding_size` to the size of the embedding for the encoder.
- Set `decoding_embedding_size` to the size of the embedding for the decoder.
- Set `learning_rate` to the learning rate.
- Set `keep_probability` to the Dropout keep probability


```python
# Number of Epochs
epochs = 8
# Batch Size
batch_size = 128
# RNN Size
rnn_size = 256
# Number of Layers
num_layers = 2
# Embedding Size
encoding_embedding_size = 120
decoding_embedding_size = 120
# Learning Rate
learning_rate = 0.001
# Dropout Keep Probability
keep_probability = 0.65
```

### Build the Graph
Build the graph using the neural network you implemented.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
save_path = 'checkpoints/dev'
(source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = helper.load_preprocess()
max_target_sentence_length = max([len(sentence) for sentence in source_int_text])

train_graph = tf.Graph()
with train_graph.as_default():
    input_data, targets, lr, keep_prob = model_inputs()
    sequence_length = tf.placeholder_with_default(max_target_sentence_length, None, name='sequence_length')
    input_shape = tf.shape(input_data)

    train_logits, inference_logits = seq2seq_model(
        tf.reverse(input_data, [-1]), targets, keep_prob, batch_size, sequence_length, len(source_vocab_to_int), len(target_vocab_to_int),
        encoding_embedding_size, decoding_embedding_size, rnn_size, num_layers, target_vocab_to_int)

    tf.identity(inference_logits, 'logits')
    with tf.name_scope("optimization"):
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(
            train_logits,
            targets,
            tf.ones([input_shape[0], sequence_length]))

        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)
```

### Train
Train the neural network on the preprocessed data. If you have a hard time getting a good loss, check the forms to see if anyone is having the same problem.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import time

def get_accuracy(target, logits):
    """
    Calculate accuracy
    """
    max_seq = max(target.shape[1], logits.shape[1])
    if max_seq - target.shape[1]:
        target = np.pad(
            target_batch,
            [(0,0),(0,max_seq - target_batch.shape[1]), (0,0)],
            'constant')
    if max_seq - batch_train_logits.shape[1]:
        logits = np.pad(
            logits,
            [(0,0),(0,max_seq - logits.shape[1]), (0,0)],
            'constant')

    return np.mean(np.equal(target, np.argmax(logits, 2)))

train_source = source_int_text[batch_size:]
train_target = target_int_text[batch_size:]

valid_source = helper.pad_sentence_batch(source_int_text[:batch_size])
valid_target = helper.pad_sentence_batch(target_int_text[:batch_size])

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(epochs):
        for batch_i, (source_batch, target_batch) in enumerate(
                helper.batch_data(train_source, train_target, batch_size)):
            start_time = time.time()

            _, loss = sess.run(
                [train_op, cost],
                {input_data: source_batch,
                 targets: target_batch,
                 lr: learning_rate,
                 sequence_length: target_batch.shape[1],
                 keep_prob: keep_probability})

            batch_train_logits = sess.run(
                inference_logits,
                {input_data: source_batch, keep_prob: 1.0})
            batch_valid_logits = sess.run(
                inference_logits,
                {input_data: valid_source, keep_prob: 1.0})

            train_acc = get_accuracy(target_batch, batch_train_logits)
            valid_acc = get_accuracy(np.array(valid_target), batch_valid_logits)
            end_time = time.time()
            if batch_i % 50 == 0:
                print('Epoch {:>3} Batch {:>4}/{} - Train Accuracy: {:>6.3f}, Validation Accuracy: {:>6.3f}, Loss: {:>6.3f}'
                      .format(epoch_i, batch_i, len(source_int_text) // batch_size, train_acc, valid_acc, loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_path)
    print('Model Trained and Saved')
```

    Epoch   0 Batch    0/1077 - Train Accuracy:  0.295, Validation Accuracy:  0.305, Loss:  5.898
    Epoch   0 Batch   50/1077 - Train Accuracy:  0.399, Validation Accuracy:  0.464, Loss:  2.796
    Epoch   0 Batch  100/1077 - Train Accuracy:  0.450, Validation Accuracy:  0.504, Loss:  2.220
    Epoch   0 Batch  150/1077 - Train Accuracy:  0.470, Validation Accuracy:  0.501, Loss:  1.744
    Epoch   0 Batch  200/1077 - Train Accuracy:  0.455, Validation Accuracy:  0.518, Loss:  1.493
    Epoch   0 Batch  250/1077 - Train Accuracy:  0.494, Validation Accuracy:  0.517, Loss:  1.157
    Epoch   0 Batch  300/1077 - Train Accuracy:  0.476, Validation Accuracy:  0.554, Loss:  1.106
    Epoch   0 Batch  350/1077 - Train Accuracy:  0.521, Validation Accuracy:  0.563, Loss:  0.991
    Epoch   0 Batch  400/1077 - Train Accuracy:  0.544, Validation Accuracy:  0.572, Loss:  0.856
    Epoch   0 Batch  450/1077 - Train Accuracy:  0.537, Validation Accuracy:  0.573, Loss:  0.789
    Epoch   0 Batch  500/1077 - Train Accuracy:  0.570, Validation Accuracy:  0.599, Loss:  0.737
    Epoch   0 Batch  550/1077 - Train Accuracy:  0.568, Validation Accuracy:  0.615, Loss:  0.753
    Epoch   0 Batch  600/1077 - Train Accuracy:  0.606, Validation Accuracy:  0.604, Loss:  0.639
    Epoch   0 Batch  650/1077 - Train Accuracy:  0.566, Validation Accuracy:  0.614, Loss:  0.663
    Epoch   0 Batch  700/1077 - Train Accuracy:  0.584, Validation Accuracy:  0.619, Loss:  0.608
    Epoch   0 Batch  750/1077 - Train Accuracy:  0.619, Validation Accuracy:  0.627, Loss:  0.603
    Epoch   0 Batch  800/1077 - Train Accuracy:  0.612, Validation Accuracy:  0.643, Loss:  0.580
    Epoch   0 Batch  850/1077 - Train Accuracy:  0.617, Validation Accuracy:  0.630, Loss:  0.593
    Epoch   0 Batch  900/1077 - Train Accuracy:  0.651, Validation Accuracy:  0.635, Loss:  0.537
    Epoch   0 Batch  950/1077 - Train Accuracy:  0.647, Validation Accuracy:  0.632, Loss:  0.466
    Epoch   0 Batch 1000/1077 - Train Accuracy:  0.695, Validation Accuracy:  0.660, Loss:  0.439
    Epoch   0 Batch 1050/1077 - Train Accuracy:  0.590, Validation Accuracy:  0.640, Loss:  0.461
    Epoch   1 Batch    0/1077 - Train Accuracy:  0.666, Validation Accuracy:  0.638, Loss:  0.407
    Epoch   1 Batch   50/1077 - Train Accuracy:  0.647, Validation Accuracy:  0.656, Loss:  0.425
    Epoch   1 Batch  100/1077 - Train Accuracy:  0.700, Validation Accuracy:  0.691, Loss:  0.411
    Epoch   1 Batch  150/1077 - Train Accuracy:  0.737, Validation Accuracy:  0.678, Loss:  0.349
    Epoch   1 Batch  200/1077 - Train Accuracy:  0.705, Validation Accuracy:  0.711, Loss:  0.369
    Epoch   1 Batch  250/1077 - Train Accuracy:  0.711, Validation Accuracy:  0.716, Loss:  0.314
    Epoch   1 Batch  300/1077 - Train Accuracy:  0.769, Validation Accuracy:  0.744, Loss:  0.318
    Epoch   1 Batch  350/1077 - Train Accuracy:  0.759, Validation Accuracy:  0.732, Loss:  0.308
    Epoch   1 Batch  400/1077 - Train Accuracy:  0.789, Validation Accuracy:  0.748, Loss:  0.296
    Epoch   1 Batch  450/1077 - Train Accuracy:  0.801, Validation Accuracy:  0.727, Loss:  0.258
    Epoch   1 Batch  500/1077 - Train Accuracy:  0.813, Validation Accuracy:  0.757, Loss:  0.249
    Epoch   1 Batch  550/1077 - Train Accuracy:  0.755, Validation Accuracy:  0.775, Loss:  0.263
    Epoch   1 Batch  600/1077 - Train Accuracy:  0.822, Validation Accuracy:  0.770, Loss:  0.214
    Epoch   1 Batch  650/1077 - Train Accuracy:  0.793, Validation Accuracy:  0.793, Loss:  0.219
    Epoch   1 Batch  700/1077 - Train Accuracy:  0.786, Validation Accuracy:  0.786, Loss:  0.196
    Epoch   1 Batch  750/1077 - Train Accuracy:  0.828, Validation Accuracy:  0.822, Loss:  0.202
    Epoch   1 Batch  800/1077 - Train Accuracy:  0.813, Validation Accuracy:  0.800, Loss:  0.190
    Epoch   1 Batch  850/1077 - Train Accuracy:  0.776, Validation Accuracy:  0.810, Loss:  0.213
    Epoch   1 Batch  900/1077 - Train Accuracy:  0.838, Validation Accuracy:  0.825, Loss:  0.187
    Epoch   1 Batch  950/1077 - Train Accuracy:  0.857, Validation Accuracy:  0.819, Loss:  0.149
    Epoch   1 Batch 1000/1077 - Train Accuracy:  0.869, Validation Accuracy:  0.849, Loss:  0.149
    Epoch   1 Batch 1050/1077 - Train Accuracy:  0.839, Validation Accuracy:  0.850, Loss:  0.150
    Epoch   2 Batch    0/1077 - Train Accuracy:  0.856, Validation Accuracy:  0.835, Loss:  0.132
    Epoch   2 Batch   50/1077 - Train Accuracy:  0.880, Validation Accuracy:  0.864, Loss:  0.127
    Epoch   2 Batch  100/1077 - Train Accuracy:  0.877, Validation Accuracy:  0.863, Loss:  0.125
    Epoch   2 Batch  150/1077 - Train Accuracy:  0.911, Validation Accuracy:  0.882, Loss:  0.130
    Epoch   2 Batch  200/1077 - Train Accuracy:  0.869, Validation Accuracy:  0.882, Loss:  0.130
    Epoch   2 Batch  250/1077 - Train Accuracy:  0.881, Validation Accuracy:  0.883, Loss:  0.113
    Epoch   2 Batch  300/1077 - Train Accuracy:  0.958, Validation Accuracy:  0.876, Loss:  0.096
    Epoch   2 Batch  350/1077 - Train Accuracy:  0.918, Validation Accuracy:  0.877, Loss:  0.099
    Epoch   2 Batch  400/1077 - Train Accuracy:  0.912, Validation Accuracy:  0.904, Loss:  0.106
    Epoch   2 Batch  450/1077 - Train Accuracy:  0.934, Validation Accuracy:  0.890, Loss:  0.082
    Epoch   2 Batch  500/1077 - Train Accuracy:  0.898, Validation Accuracy:  0.894, Loss:  0.070
    Epoch   2 Batch  550/1077 - Train Accuracy:  0.868, Validation Accuracy:  0.920, Loss:  0.081
    Epoch   2 Batch  600/1077 - Train Accuracy:  0.945, Validation Accuracy:  0.910, Loss:  0.070
    Epoch   2 Batch  650/1077 - Train Accuracy:  0.925, Validation Accuracy:  0.893, Loss:  0.072
    Epoch   2 Batch  700/1077 - Train Accuracy:  0.952, Validation Accuracy:  0.903, Loss:  0.056
    Epoch   2 Batch  750/1077 - Train Accuracy:  0.926, Validation Accuracy:  0.926, Loss:  0.066
    Epoch   2 Batch  800/1077 - Train Accuracy:  0.906, Validation Accuracy:  0.922, Loss:  0.072
    Epoch   2 Batch  850/1077 - Train Accuracy:  0.909, Validation Accuracy:  0.921, Loss:  0.095
    Epoch   2 Batch  900/1077 - Train Accuracy:  0.924, Validation Accuracy:  0.925, Loss:  0.068
    Epoch   2 Batch  950/1077 - Train Accuracy:  0.932, Validation Accuracy:  0.913, Loss:  0.057
    Epoch   2 Batch 1000/1077 - Train Accuracy:  0.916, Validation Accuracy:  0.932, Loss:  0.061
    Epoch   2 Batch 1050/1077 - Train Accuracy:  0.945, Validation Accuracy:  0.929, Loss:  0.051
    Epoch   3 Batch    0/1077 - Train Accuracy:  0.946, Validation Accuracy:  0.929, Loss:  0.045
    Epoch   3 Batch   50/1077 - Train Accuracy:  0.945, Validation Accuracy:  0.927, Loss:  0.052
    Epoch   3 Batch  100/1077 - Train Accuracy:  0.938, Validation Accuracy:  0.951, Loss:  0.053
    Epoch   3 Batch  150/1077 - Train Accuracy:  0.923, Validation Accuracy:  0.928, Loss:  0.054
    Epoch   3 Batch  200/1077 - Train Accuracy:  0.923, Validation Accuracy:  0.907, Loss:  0.054
    Epoch   3 Batch  250/1077 - Train Accuracy:  0.937, Validation Accuracy:  0.924, Loss:  0.045
    Epoch   3 Batch  300/1077 - Train Accuracy:  0.949, Validation Accuracy:  0.932, Loss:  0.040
    Epoch   3 Batch  350/1077 - Train Accuracy:  0.943, Validation Accuracy:  0.923, Loss:  0.051
    Epoch   3 Batch  400/1077 - Train Accuracy:  0.945, Validation Accuracy:  0.940, Loss:  0.055
    Epoch   3 Batch  450/1077 - Train Accuracy:  0.942, Validation Accuracy:  0.903, Loss:  0.049
    Epoch   3 Batch  500/1077 - Train Accuracy:  0.939, Validation Accuracy:  0.931, Loss:  0.038
    Epoch   3 Batch  550/1077 - Train Accuracy:  0.914, Validation Accuracy:  0.930, Loss:  0.039
    Epoch   3 Batch  600/1077 - Train Accuracy:  0.946, Validation Accuracy:  0.930, Loss:  0.039
    Epoch   3 Batch  650/1077 - Train Accuracy:  0.957, Validation Accuracy:  0.942, Loss:  0.044
    Epoch   3 Batch  700/1077 - Train Accuracy:  0.950, Validation Accuracy:  0.925, Loss:  0.033
    Epoch   3 Batch  750/1077 - Train Accuracy:  0.936, Validation Accuracy:  0.950, Loss:  0.042
    Epoch   3 Batch  800/1077 - Train Accuracy:  0.927, Validation Accuracy:  0.935, Loss:  0.041
    Epoch   3 Batch  850/1077 - Train Accuracy:  0.934, Validation Accuracy:  0.945, Loss:  0.058
    Epoch   3 Batch  900/1077 - Train Accuracy:  0.948, Validation Accuracy:  0.937, Loss:  0.048
    Epoch   3 Batch  950/1077 - Train Accuracy:  0.949, Validation Accuracy:  0.916, Loss:  0.039
    Epoch   3 Batch 1000/1077 - Train Accuracy:  0.952, Validation Accuracy:  0.924, Loss:  0.039
    Epoch   3 Batch 1050/1077 - Train Accuracy:  0.960, Validation Accuracy:  0.945, Loss:  0.030
    Epoch   4 Batch    0/1077 - Train Accuracy:  0.948, Validation Accuracy:  0.929, Loss:  0.032
    Epoch   4 Batch   50/1077 - Train Accuracy:  0.957, Validation Accuracy:  0.930, Loss:  0.036
    Epoch   4 Batch  100/1077 - Train Accuracy:  0.953, Validation Accuracy:  0.945, Loss:  0.031
    Epoch   4 Batch  150/1077 - Train Accuracy:  0.940, Validation Accuracy:  0.952, Loss:  0.039
    Epoch   4 Batch  200/1077 - Train Accuracy:  0.946, Validation Accuracy:  0.932, Loss:  0.034
    Epoch   4 Batch  250/1077 - Train Accuracy:  0.929, Validation Accuracy:  0.942, Loss:  0.034
    Epoch   4 Batch  300/1077 - Train Accuracy:  0.963, Validation Accuracy:  0.922, Loss:  0.032
    Epoch   4 Batch  350/1077 - Train Accuracy:  0.957, Validation Accuracy:  0.924, Loss:  0.031
    Epoch   4 Batch  400/1077 - Train Accuracy:  0.946, Validation Accuracy:  0.948, Loss:  0.049
    Epoch   4 Batch  450/1077 - Train Accuracy:  0.950, Validation Accuracy:  0.936, Loss:  0.037
    Epoch   4 Batch  500/1077 - Train Accuracy:  0.958, Validation Accuracy:  0.950, Loss:  0.028
    Epoch   4 Batch  550/1077 - Train Accuracy:  0.943, Validation Accuracy:  0.934, Loss:  0.030
    Epoch   4 Batch  600/1077 - Train Accuracy:  0.957, Validation Accuracy:  0.958, Loss:  0.036
    Epoch   4 Batch  650/1077 - Train Accuracy:  0.963, Validation Accuracy:  0.944, Loss:  0.035
    Epoch   4 Batch  700/1077 - Train Accuracy:  0.963, Validation Accuracy:  0.935, Loss:  0.021
    Epoch   4 Batch  750/1077 - Train Accuracy:  0.952, Validation Accuracy:  0.946, Loss:  0.028
    Epoch   4 Batch  800/1077 - Train Accuracy:  0.956, Validation Accuracy:  0.945, Loss:  0.030
    Epoch   4 Batch  850/1077 - Train Accuracy:  0.955, Validation Accuracy:  0.950, Loss:  0.052
    Epoch   4 Batch  900/1077 - Train Accuracy:  0.954, Validation Accuracy:  0.946, Loss:  0.033
    Epoch   4 Batch  950/1077 - Train Accuracy:  0.964, Validation Accuracy:  0.936, Loss:  0.026
    Epoch   4 Batch 1000/1077 - Train Accuracy:  0.955, Validation Accuracy:  0.946, Loss:  0.039
    Epoch   4 Batch 1050/1077 - Train Accuracy:  0.967, Validation Accuracy:  0.930, Loss:  0.023
    Epoch   5 Batch    0/1077 - Train Accuracy:  0.963, Validation Accuracy:  0.950, Loss:  0.022
    Epoch   5 Batch   50/1077 - Train Accuracy:  0.972, Validation Accuracy:  0.945, Loss:  0.026
    Epoch   5 Batch  100/1077 - Train Accuracy:  0.979, Validation Accuracy:  0.951, Loss:  0.023
    Epoch   5 Batch  150/1077 - Train Accuracy:  0.965, Validation Accuracy:  0.943, Loss:  0.032
    Epoch   5 Batch  200/1077 - Train Accuracy:  0.961, Validation Accuracy:  0.941, Loss:  0.023
    Epoch   5 Batch  250/1077 - Train Accuracy:  0.952, Validation Accuracy:  0.954, Loss:  0.029
    Epoch   5 Batch  300/1077 - Train Accuracy:  0.977, Validation Accuracy:  0.944, Loss:  0.025
    Epoch   5 Batch  350/1077 - Train Accuracy:  0.965, Validation Accuracy:  0.927, Loss:  0.022
    Epoch   5 Batch  400/1077 - Train Accuracy:  0.949, Validation Accuracy:  0.952, Loss:  0.029
    Epoch   5 Batch  450/1077 - Train Accuracy:  0.954, Validation Accuracy:  0.940, Loss:  0.026
    Epoch   5 Batch  500/1077 - Train Accuracy:  0.975, Validation Accuracy:  0.943, Loss:  0.016
    Epoch   5 Batch  550/1077 - Train Accuracy:  0.970, Validation Accuracy:  0.943, Loss:  0.023
    Epoch   5 Batch  600/1077 - Train Accuracy:  0.983, Validation Accuracy:  0.939, Loss:  0.023
    Epoch   5 Batch  650/1077 - Train Accuracy:  0.976, Validation Accuracy:  0.958, Loss:  0.026
    Epoch   5 Batch  700/1077 - Train Accuracy:  0.961, Validation Accuracy:  0.936, Loss:  0.020
    Epoch   5 Batch  750/1077 - Train Accuracy:  0.964, Validation Accuracy:  0.946, Loss:  0.023
    Epoch   5 Batch  800/1077 - Train Accuracy:  0.964, Validation Accuracy:  0.954, Loss:  0.020
    Epoch   5 Batch  850/1077 - Train Accuracy:  0.965, Validation Accuracy:  0.959, Loss:  0.049
    Epoch   5 Batch  900/1077 - Train Accuracy:  0.971, Validation Accuracy:  0.946, Loss:  0.025
    Epoch   5 Batch  950/1077 - Train Accuracy:  0.966, Validation Accuracy:  0.961, Loss:  0.018
    Epoch   5 Batch 1000/1077 - Train Accuracy:  0.955, Validation Accuracy:  0.946, Loss:  0.020
    Epoch   5 Batch 1050/1077 - Train Accuracy:  0.985, Validation Accuracy:  0.937, Loss:  0.013
    Epoch   6 Batch    0/1077 - Train Accuracy:  0.960, Validation Accuracy:  0.958, Loss:  0.019
    Epoch   6 Batch   50/1077 - Train Accuracy:  0.968, Validation Accuracy:  0.925, Loss:  0.022
    Epoch   6 Batch  100/1077 - Train Accuracy:  0.986, Validation Accuracy:  0.945, Loss:  0.017
    Epoch   6 Batch  150/1077 - Train Accuracy:  0.969, Validation Accuracy:  0.944, Loss:  0.022
    Epoch   6 Batch  200/1077 - Train Accuracy:  0.957, Validation Accuracy:  0.958, Loss:  0.024
    Epoch   6 Batch  250/1077 - Train Accuracy:  0.967, Validation Accuracy:  0.954, Loss:  0.021
    Epoch   6 Batch  300/1077 - Train Accuracy:  0.972, Validation Accuracy:  0.945, Loss:  0.022
    Epoch   6 Batch  350/1077 - Train Accuracy:  0.966, Validation Accuracy:  0.932, Loss:  0.019
    Epoch   6 Batch  400/1077 - Train Accuracy:  0.971, Validation Accuracy:  0.943, Loss:  0.023
    Epoch   6 Batch  450/1077 - Train Accuracy:  0.968, Validation Accuracy:  0.948, Loss:  0.024
    Epoch   6 Batch  500/1077 - Train Accuracy:  0.973, Validation Accuracy:  0.961, Loss:  0.017
    Epoch   6 Batch  550/1077 - Train Accuracy:  0.966, Validation Accuracy:  0.934, Loss:  0.016
    Epoch   6 Batch  600/1077 - Train Accuracy:  0.970, Validation Accuracy:  0.952, Loss:  0.020
    Epoch   6 Batch  650/1077 - Train Accuracy:  0.972, Validation Accuracy:  0.969, Loss:  0.020
    Epoch   6 Batch  700/1077 - Train Accuracy:  0.972, Validation Accuracy:  0.974, Loss:  0.017
    Epoch   6 Batch  750/1077 - Train Accuracy:  0.960, Validation Accuracy:  0.958, Loss:  0.019
    Epoch   6 Batch  800/1077 - Train Accuracy:  0.979, Validation Accuracy:  0.960, Loss:  0.020
    Epoch   6 Batch  850/1077 - Train Accuracy:  0.967, Validation Accuracy:  0.944, Loss:  0.039
    Epoch   6 Batch  900/1077 - Train Accuracy:  0.968, Validation Accuracy:  0.960, Loss:  0.026
    Epoch   6 Batch  950/1077 - Train Accuracy:  0.974, Validation Accuracy:  0.961, Loss:  0.018
    Epoch   6 Batch 1000/1077 - Train Accuracy:  0.975, Validation Accuracy:  0.963, Loss:  0.020
    Epoch   6 Batch 1050/1077 - Train Accuracy:  0.974, Validation Accuracy:  0.939, Loss:  0.012
    Epoch   7 Batch    0/1077 - Train Accuracy:  0.967, Validation Accuracy:  0.962, Loss:  0.021
    Epoch   7 Batch   50/1077 - Train Accuracy:  0.967, Validation Accuracy:  0.939, Loss:  0.020
    Epoch   7 Batch  100/1077 - Train Accuracy:  0.980, Validation Accuracy:  0.961, Loss:  0.012
    Epoch   7 Batch  150/1077 - Train Accuracy:  0.964, Validation Accuracy:  0.949, Loss:  0.020
    Epoch   7 Batch  200/1077 - Train Accuracy:  0.977, Validation Accuracy:  0.951, Loss:  0.012
    Epoch   7 Batch  250/1077 - Train Accuracy:  0.969, Validation Accuracy:  0.971, Loss:  0.019
    Epoch   7 Batch  300/1077 - Train Accuracy:  0.969, Validation Accuracy:  0.951, Loss:  0.023
    Epoch   7 Batch  350/1077 - Train Accuracy:  0.977, Validation Accuracy:  0.950, Loss:  0.021
    Epoch   7 Batch  400/1077 - Train Accuracy:  0.975, Validation Accuracy:  0.961, Loss:  0.019
    Epoch   7 Batch  450/1077 - Train Accuracy:  0.977, Validation Accuracy:  0.944, Loss:  0.020
    Epoch   7 Batch  500/1077 - Train Accuracy:  0.982, Validation Accuracy:  0.958, Loss:  0.012
    Epoch   7 Batch  550/1077 - Train Accuracy:  0.971, Validation Accuracy:  0.952, Loss:  0.012
    Epoch   7 Batch  600/1077 - Train Accuracy:  0.980, Validation Accuracy:  0.946, Loss:  0.016
    Epoch   7 Batch  650/1077 - Train Accuracy:  0.977, Validation Accuracy:  0.961, Loss:  0.018
    Epoch   7 Batch  700/1077 - Train Accuracy:  0.979, Validation Accuracy:  0.972, Loss:  0.015
    Epoch   7 Batch  750/1077 - Train Accuracy:  0.965, Validation Accuracy:  0.963, Loss:  0.020
    Epoch   7 Batch  800/1077 - Train Accuracy:  0.975, Validation Accuracy:  0.972, Loss:  0.014
    Epoch   7 Batch  850/1077 - Train Accuracy:  0.979, Validation Accuracy:  0.968, Loss:  0.025
    Epoch   7 Batch  900/1077 - Train Accuracy:  0.979, Validation Accuracy:  0.962, Loss:  0.014
    Epoch   7 Batch  950/1077 - Train Accuracy:  0.974, Validation Accuracy:  0.979, Loss:  0.013
    Epoch   7 Batch 1000/1077 - Train Accuracy:  0.975, Validation Accuracy:  0.947, Loss:  0.018
    Epoch   7 Batch 1050/1077 - Train Accuracy:  0.983, Validation Accuracy:  0.935, Loss:  0.011
    Model Trained and Saved


### Save Parameters
Save the `batch_size` and `save_path` parameters for inference.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# Save parameters for checkpoint
helper.save_params(save_path)
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

_, (source_vocab_to_int, target_vocab_to_int), (source_int_to_vocab, target_int_to_vocab) = helper.load_preprocess()
load_path = helper.load_params()
```

## Sentence to Sequence
To feed a sentence into the model for translation, you first need to preprocess it.  Implement the function `sentence_to_seq()` to preprocess new sentences.

- Convert the sentence to lowercase
- Convert words into ids using `vocab_to_int`
 - Convert words not in the vocabulary, to the `<UNK>` word id.


```python
def sentence_to_seq(sentence, vocab_to_int):
    """
    Convert a sentence to a sequence of ids
    :param sentence: String
    :param vocab_to_int: Dictionary to go from the words to an id
    :return: List of word ids
    """

    # Convert sentences to lowercase
    sent_lower = sentence.lower().split()

    # Convert words to ids
    word_ids = []
    for word in sent_lower:
        if word in vocab_to_int:
            word_ids.append(vocab_to_int[word])
        # Convert unknown words
        else:
            word_ids.append(vocab_to_int['<UNK>'])

    return word_ids

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_sentence_to_seq(sentence_to_seq)
```

    Tests Passed


## Translate
This will translate `translate_sentence` from English to French.


```python
translate_sentence = 'he saw a old yellow truck .'


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
translate_sentence = sentence_to_seq(translate_sentence, source_vocab_to_int)

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(load_path + '.meta')
    loader.restore(sess, load_path)

    input_data = loaded_graph.get_tensor_by_name('input:0')
    logits = loaded_graph.get_tensor_by_name('logits:0')
    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

    translate_logits = sess.run(logits, {input_data: [translate_sentence], keep_prob: 1.0})[0]

print('Input')
print('  Word Ids:      {}'.format([i for i in translate_sentence]))
print('  English Words: {}'.format([source_int_to_vocab[i] for i in translate_sentence]))

print('\nPrediction')
print('  Word Ids:      {}'.format([i for i in np.argmax(translate_logits, 1)]))
print('  French Words: {}'.format([target_int_to_vocab[i] for i in np.argmax(translate_logits, 1)]))
```

    Input
      Word Ids:      [200, 105, 110, 141, 58, 65, 36]
      English Words: ['he', 'saw', 'a', 'old', 'yellow', 'truck', '.']

    Prediction
      Word Ids:      [208, 131, 113, 219, 277, 231, 147, 186, 1]
      French Words: ['il', 'a', 'vu', 'un', 'vieux', 'camion', 'jaune', '.', '<EOS>']


## Imperfect Translation
You might notice that some sentences translate better than others.  Since the dataset you're using only has a vocabulary of 227 English words of the thousands that you use, you're only going to see good results using these words.  For this project, you don't need a perfect translation. However, if you want to create a better translation model, you'll need better data.

You can train on the [WMT10 French-English corpus](http://www.statmt.org/wmt10/training-giga-fren.tar).  This dataset has more vocabulary and richer in topics discussed.  However, this will take you days to train, so make sure you've a GPU and the neural network is performing well on dataset we provided.  Just make sure you play with the WMT10 corpus after you've submitted this project.
## Submitting This Project
When submitting this project, make sure to run all the cells before saving the notebook. Save the notebook file as "dlnd_language_translation.ipynb" and save it as a HTML file under "File" -> "Download as". Include the "helper.py" and "problem_unittests.py" files in your submission.
