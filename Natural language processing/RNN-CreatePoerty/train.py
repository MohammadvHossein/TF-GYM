import tensorflow as tf
import numpy as np
from collections import Counter
import os

# Load the text
with open('khayyam.txt', 'rb') as file:
    text = file.read().decode(encoding='utf-8')

# Print the first 10 characters
print(text[:10])

# Create character dictionary
vocabolaries = sorted(set(text))
print(vocabolaries)
print(len(vocabolaries))

char2index = {u: i for i, u in enumerate(vocabolaries)}
index2char = np.array(vocabolaries)

# Convert text to integers
text_as_integer = np.array([char2index[c] for c in text])

# Create character dataset
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_integer)

# Create sequences
sequences = char_dataset.batch(101, drop_remainder=True)

# Function to create input and target
def sit(batch):
    input_text = batch[:-1]
    target_text = batch[1:]
    return input_text, target_text

# Create final dataset
dataset = sequences.map(sit)
dataset = dataset.shuffle(10000).batch(64, drop_remainder=True)  # Shuffle dataset for better training

# Model parameters
vocabolary_size = len(vocabolaries)
embedding_dim = 256
rnn_units = 1024

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocabolary_size, embedding_dim),
    tf.keras.layers.GRU(rnn_units, return_sequences=True, recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocabolary_size)
])

# Loss function
def loss_f(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

# Compile the model
model.compile(optimizer='adam', loss=loss_f)

# Ensure the directory exists
checkpoint_dir = 'khayyammolana'
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, 'checkpoints.weights.{epoch:02d}.h5'), save_weights_only=True)

# Train the model
history = model.fit(dataset, epochs=700, callbacks=[checkpoint])

# Load model weights
model_2 = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocabolary_size, embedding_dim),
    tf.keras.layers.GRU(rnn_units, return_sequences=True, recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocabolary_size)
])

latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
if latest_checkpoint is not None:
    model_2.load_weights(latest_checkpoint)
    model_2.build(tf.TensorShape([1, None]))
else:
    print('No checkpoint found.')

# Generate text
num_generate = 1000
first_string = 'به نام خداوند جان و خرد'
input_eval = [char2index[s] for s in first_string]
input_eval = tf.expand_dims(input_eval, 0)

model_2.reset_states()

text_generated = [first_string]
for i in range(num_generate):
    predictions = model_2.predict(input_eval)
    predictions = tf.squeeze(predictions, 0)
    predicted_ids = np.array(predictions.numpy()).argmax(axis=1)[-1]
    input_eval = tf.expand_dims(np.append(input_eval[0].numpy(), predicted_ids)[1:], 0)
    text_generated.append(index2char[predicted_ids])

# Save the generated text to a file
with open('generated_text.txt', 'w', encoding='utf-8') as f:
    f.write(''.join(text_generated))

# Print the generated text
for line in ''.join(text_generated).split('\n'):
    print(line)