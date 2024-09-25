import tensorflow as tf
import tensorflow.keras as keras
import pickle
import numpy as np
import unicodedata
import re

# Load the saved tokenizers
with open('input_lang_tokenizer.pkl', 'rb') as f:
    input_lang_tokenizer = pickle.load(f)

with open('target_lang_tokenizer.pkl', 'rb') as f:
    target_lang_tokenizer = pickle.load(f)

# Load the parameters
with open('params.pkl', 'rb') as f:
    params = pickle.load(f)
    max_length_inp = params['max_length_inp']
    max_length_targ = params['max_length_targ']
    units = params['units']
    vocab_inp_size = params['vocab_inp_size']
    vocab_targ_size = params['vocab_targ_size']

# Define the preprocess_sentence function as in the original code
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

# Redefine the models (Encoder and Decoder) and the Attention layer
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

class Attention(keras.layers.Layer):
    def __init__ (self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
    def call (self, query, values):
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

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

# Initialize the models with the loaded parameters
embedding_dim = 256
encoder = Encoder(vocab_inp_size, embedding_dim, units, 1)
decoder = Decoder(vocab_targ_size, embedding_dim, units, 1)

# Dummy call to build the models (this is crucial to initialize the variables)
dummy_input = tf.constant([[0]])
dummy_hidden = encoder.initilize_hidden_state()
encoder(dummy_input, dummy_hidden)
decoder(dummy_input, dummy_hidden, tf.zeros((1, 1, units)))

# Load the weights
encoder.load_weights('encoder_weights.h5')
decoder.load_weights('decoder_weights.h5')

# Define the evaluate function as in the original code
def evaluate(sentence):
    sentence = preprocess_senetence(sentence)
    inputs = [input_lang_tokenizer.word_index[i] for i in sentence.split(' ')]
    inputs = keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    result = ''
    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([target_lang_tokenizer.word_index['<strat>']], 0)
    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
        attention_weights = tf.reshape(attention_weights, (-1, ))
        predicted_id = tf.argmax(predictions[0]).numpy()
        result += target_lang_tokenizer.index_word[predicted_id] + ' '
        if target_lang_tokenizer.index_word[predicted_id] == '<end>':
            return result, sentence
        dec_input = tf.expand_dims([predicted_id], 0)
    return result, sentence

# Run the evaluate function
print(evaluate('quiero un poco de cafe'))
print(evaluate('Me puse enfermo'))
print(evaluate('Estoy hambriento'))