import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
import unicodedata
import re
import numpy as np
import os
import io

path_to_zip = keras.utils.get_file('spa-eng.zip', origin='https://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip', extract=True)

path_to_file = os.path.join(os.path.dirname(path_to_zip), 'spa-eng', 'spa.txt')

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def preprocess_senetence(w):
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.~,])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.~,]+", " ", w)
    w = w.rstrip().strip()
    w = '<strat> ' + w + ' <end>'
    return w

def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[preprocess_senetence(w) for w in l.split('\t')] for l in lines[:num_examples]]
    return zip(*word_pairs)

def max_length(tensor):
    return max(len(t) for t in tensor)

def tokenize(lang):
    lang_tokenizer = keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, lang_tokenizer

def load_dataset(path, num_examples=None):
    targ_lang, inp_lang = create_dataset(path, num_examples)
    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, target_lang_tokenizer = tokenize(targ_lang)
    return input_tensor, target_tensor, inp_lang_tokenizer, target_lang_tokenizer

input_tensor, target_tensor, input_lang_tokenizer, target_lang_tokenizer = load_dataset(path_to_file, 20000)

max_length_targ, max_length_inp = max_length(target_tensor), max_length(input_tensor)

X_train, X_test, y_train, y_test = train_test_split(input_tensor, target_tensor, test_size=0.2)

BUFFER_SIZE = len(X_train)
BATCH_SIZE = 64
steps_per_epoch = len(X_train) // BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = len(input_lang_tokenizer.word_index) + 1
vocab_targ_size = len(target_lang_tokenizer.word_index) + 1

dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

class Encoder(keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units
        self.embedding = keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = keras.layers.GRU(self.enc_units, return_sequences=True, return_state=True)
    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state
    def initilize_hidden_state(self):
        return tf.zeros((self.batch_size, self.enc_units))
    
encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

simple_hidden = encoder.initilize_hidden_state()

example_input_batch, example_target_batch = next(iter(dataset))
simple_output, simple_states = encoder(example_input_batch, simple_hidden)

class Attention(keras.layers.Layer):
    def __init__ (self, units):
        super(Attention, self).__init__()
        self.W1 = keras.layers.Dense(units)
        self.W2 = keras.layers.Dense(units)
        self.V = keras.layers.Dense(1)
    def call (self, query, values):
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
        atteion_weights = tf.nn.softmax(score, axis=1)
        context_vector = atteion_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, atteion_weights

attention_layer = Attention(10)
attention_result, attention_weights = attention_layer(simple_hidden, simple_output)

class Decoder(keras.Model):
    def __init__ (self, vocab_size, embedding_dim, dec_units, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.dec_units = dec_units
        self.embedding = keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = keras.layers.GRU(self.dec_units, return_sequences=True, return_state=True)
        self.fc = keras.layers.Dense(vocab_size)
        self.attention = Attention(self.dec_units)
    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights

decoder = Decoder(vocab_targ_size, embedding_dim, units, BATCH_SIZE)

optimizer = keras.optimizers.Adam()
loss_object = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

checkpoint_dir = 'chckpnts'
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

def train_step(inp, targ, enc_hidden):
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([target_lang_tokenizer.word_index['<strat>']] * BATCH_SIZE, 1)
        for t in range(1, targ.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_function(targ[:, t], predictions)
            dec_input = tf.expand_dims(targ[:, t], 1)
    batch_loss = (loss / int(targ.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss

EPOCHS = 20
for epoch in range(EPOCHS):
    enc_hidden = encoder.initilize_hidden_state()
    total_loss = 0
    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss
        print('Epoch: ', epoch)
        print('Loss: ', batch_loss.numpy())
    checkpoint.save(file_prefix=os.path.join(checkpoint_dir, 'ckpt'))

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# Save the encoder weights
encoder.save_weights('encoder_weights.h5')

# Save the decoder weights
decoder.save_weights('decoder_weights.h5')

# Save the tokenizers
import pickle

with open('input_lang_tokenizer.pkl', 'wb') as f:
    pickle.dump(input_lang_tokenizer, f)

with open('target_lang_tokenizer.pkl', 'wb') as f:
    pickle.dump(target_lang_tokenizer, f)

# Save important parameters
params = {
    'max_length_inp': max_length_inp,
    'max_length_targ': max_length_targ,
    'units': units,
    'vocab_inp_size': vocab_inp_size,
    'vocab_targ_size': vocab_targ_size
}

with open('params.pkl', 'wb') as f:
    pickle.dump(params, f)