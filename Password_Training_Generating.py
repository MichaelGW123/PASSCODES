# Michael Williamson
# Masters Research
# Revised Version
# Recurrent Neural Network - Text Generation

############################## Part 0 Imports ################################################

# Importing the libraries
import os
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import time


garget_example_batch = 0
example_batch_predictions = 0
checkpoint_callback = 0
target_example_batch = 0

g_importing_time = 0
g_vectorizing_time = 0
g_training_time = 0
g_generating_time = 0

length = 0
starting_words = []


def preprocess_data(file_name, number_of_lines):
    ############################## Part 1 - Data Preprocessing ########################
    start = time.time()

    print(file_name)
    # Open the file to grab the first 'number_of_lines' you decided to populate the model with when generating
    file_grab_start_words = open(file_name, 'r')

    for _ in range(number_of_lines):
        line = file_grab_start_words.readline()
        starting_words.append(line.strip())
    # count the rest of the lines in the file
    nonempty_lines = [line.strip("\n")
                      for line in file_grab_start_words if line != "\n"]
    line_count = len(nonempty_lines)+number_of_lines
    file_grab_start_words.close()
    print(f'Number of lines: {line_count} lines')

    text = open(file_name, 'rb').read().decode(encoding='utf-8')
    length = len(text)
    print(f'Length of text: {length} characters')
    end = time.time()
    g_importing_time = end - start
    print(g_importing_time)

    # The unique characters in the file
    vocab = sorted(set(text))
    print(f'{len(vocab)} unique characters')
    start = time.time()
    # Vectorize the Text
    # # Use GPU if it's available
    if tf.config.list_physical_devices('GPU'):
        device = '/GPU:0'
    else:
        device = '/CPU:0'

    with tf.device(device):
        # preprocessing.StringLookup converts each character into a numeric ID
        ids_from_chars = preprocessing.StringLookup(
            vocabulary=list(vocab), mask_token=None)
        chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

        # For each input sequence the corresponding target contains the same length of text but instead of the following chars
        all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
        ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

        seq_length = 7

        sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

        def split_input_target(sequence):
            input_text = sequence[:-1]
            target_text = sequence[1:]
            return input_text, target_text

        dataset = sequences.map(split_input_target)

        # Create Training Batches
        # Batch size
        BATCH_SIZE = 1024

        # Buffer size to shuffle the dataset
        # (TF data is designed to work with possibly infinite sequences,
        # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
        # it maintains a buffer in which it shuffles elements).
        BUFFER_SIZE = 10000

        dataset = (dataset.shuffle(BUFFER_SIZE).batch(
            BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE))

    end = time.time()
    g_vectorizing_time = end - start
    print(g_vectorizing_time)

    # Length of the vocabulary in chars
    vocab_size = len(vocab)
    return ids_from_chars, chars_from_ids, dataset, vocab_size


class MyModelOne(tf.keras.Model):  # Creating the model, adding the necessary layers
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            rnn_units, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x


class MyModelTwo(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru1 = tf.keras.layers.GRU(
            rnn_units, return_sequences=True, return_state=True)
        self.gru2 = tf.keras.layers.GRU(
            rnn_units, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        flag = 0
        if states is None:
            flag = 1
            states = self.gru1.get_initial_state(x)
        x, states = self.gru1(x, initial_state=states, training=training)
        if flag == 1:
            states = self.gru2.get_initial_state(x)
        x, states = self.gru2(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x


class MyModelThree(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru1 = tf.keras.layers.GRU(
            rnn_units, return_sequences=True, return_state=True)
        self.gru2 = tf.keras.layers.GRU(
            rnn_units, return_sequences=True, return_state=True)
        self.gru3 = tf.keras.layers.GRU(
            rnn_units, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        flag = 0
        if states is None:
            flag = 1
            states = self.gru1.get_initial_state(x)
        x, states = self.gru1(x, initial_state=states, training=training)
        if flag == 1:
            flag = 2
            states = self.gru2.get_initial_state(x)
        x, states = self.gru2(x, initial_state=states, training=training)
        if flag == 2:
            states = self.gru3.get_initial_state(x)
        x, states = self.gru3(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x


def create_rnn_model(vocab_size, embedding_dim, rnn_units, num_layers):
    ############################## Part 2 Creating the RNN ########################
    if num_layers == 1:
        return MyModelOne(vocab_size, embedding_dim, rnn_units)
    elif num_layers == 2:
        return MyModelTwo(vocab_size, embedding_dim, rnn_units)
    elif num_layers == 3:
        return MyModelThree(vocab_size, embedding_dim, rnn_units)
    else:
        raise ValueError("Unsupported number of layers")


def train_model(model, dataset, epoch_count, specific_file, rnn_units, num_layers, save_new_weight, checkpoint_path, example_batch_predictions):
    ############################## Part 3 - Training the RNN ########################
    start = time.time()
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    example_batch_loss = loss(target_example_batch, example_batch_predictions)
    mean_loss = example_batch_loss.numpy().mean()
    print("Prediction shape: ", example_batch_predictions.shape,
          " # (batch_size, sequence_length, vocab_size)")
    print("Mean loss:        ", mean_loss)

    model.compile(optimizer='adam', loss=loss)

    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(
        checkpoint_dir, "{num_layers} Hidden Layers/ckpt_{specific_file}-{epoch}")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    if (save_new_weight):  # If we are saving a weight from the checkpoint, simply load, save, exit
        model.load_weights(
            f'./training_checkpoints/{num_layers} Hidden Layers/ckpt_{specific_file}-{epoch_count}').expect_partial()
        model.save_weights(checkpoint_path)
        print(f"Saved weight from ckpt_{specific_file}-{epoch_count}")
        exit()

    es = EarlyStopping(monitor='loss', min_delta=0.0015,
                       mode='min', verbose=1, patience=3)
    history = model.fit(dataset, epochs=epoch_count, callbacks=[
                        es, checkpoint_callback])
    early_stop = len(history.history['loss'])
    checkpoint_path = f'./model_weights/{num_layers} Hidden Layers/{specific_file}/model({rnn_units}by{rnn_units})({early_stop})-{specific_file}'
    model.save_weights(checkpoint_path)

    # Review models loss and training for evaluation
    print(history.history['loss'])
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(
        f'Training_Graphs/{num_layers} Hidden Layers/{specific_file}_{rnn_units}_training.png')
    end = time.time()
    g_training_time = end - start
    line = f"File: {specific_file} \nImporting Data Time: {g_importing_time} \nVectorizing Data Time: {g_vectorizing_time} \nLayers: {num_layers}\n Neurons: {rnn_units}\nTraining Time: {g_training_time} seconds\nLoss: {history.history['loss'][-4:]}\n\n"
    record = open('./runtime.txt', "a", encoding='utf-8')
    record.write(line)
    record.close()


# If training, fit the model, save the weights, then save the runtime statistics

def generate_text(model, starting_words, checkpoint_path, chars_from_ids, ids_from_chars, length, number_of_lines, num_layers, specific_file):
    ############################## Step 4 - Generating Text ########################
    print("Generating Text Begun - ")
    model.load_weights(checkpoint_path)

    class OneStep(tf.keras.Model):
        def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
            super().__init__()
            self.temperature = temperature
            self.model = model
            self.chars_from_ids = chars_from_ids
            self.ids_from_chars = ids_from_chars

            # Create a mask to prevent "[UNK]" from being generated.
            skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
            sparse_mask = tf.SparseTensor(
                # Put a -inf at each bad index.
                values=[-float('inf')]*len(skip_ids),
                indices=skip_ids,
                # Match the shape to the vocabulary
                dense_shape=[len(ids_from_chars.get_vocabulary())])
            self.prediction_mask = tf.sparse.to_dense(sparse_mask)

        @tf.function
        def generate_one_step(self, inputs, states=None):
            # Convert strings to token IDs.
            input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
            input_ids = self.ids_from_chars(input_chars).to_tensor()

            # Run the model.
            # predicted_logits.shape is [batch, char, next_char_logits]
            predicted_logits, states = self.model(inputs=input_ids, states=states,
                                                  return_state=True)
            # Only use the last prediction.
            predicted_logits = predicted_logits[:, -1, :]
            predicted_logits = predicted_logits/self.temperature
            # Apply the prediction mask: prevent "[UNK]" from being generated.
            predicted_logits = predicted_logits + self.prediction_mask

            # Sample the output logits to generate token IDs.
            predicted_ids = tf.random.categorical(
                predicted_logits, num_samples=1)
            predicted_ids = tf.squeeze(predicted_ids, axis=-1)

            # Convert from token ids to characters
            predicted_chars = self.chars_from_ids(predicted_ids)

            # Return the characters and model state.
            return predicted_chars, states

    start = time.time()

    one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

    states = None

    # Sets the tensorflow object to the array of starting words, equal to the first number_of_lines passwords in the file
    next_char = tf.constant(starting_words)
    result = [next_char]
    # Saturation is the multiple of the original file. So in the case of saturation 10, we generate 10 times the trained file
    saturation = 10
    number_of_guesses = int(length*saturation/number_of_lines)
    partial_guess = int(number_of_guesses/100)
    print(number_of_guesses)
    output_path = f"./Generated_Files/{num_layers} Hidden Layers/PRED{specific_file}-{saturation*100}(RNN).txt"
    f = open(output_path, "a", encoding='utf-8')
    current_line = ""
    for n in range(number_of_guesses):
        next_char, states = one_step_model.generate_one_step(
            next_char, states=states)
        result.append(next_char)

        # Convert the TensorFlow objects into strings
        temp = tf.strings.join(result)

        # Loop through the tf matrix to decode the strings
        for pred in range(number_of_lines):
            output = temp[pred].numpy().decode('utf-8')
            current_line += output  # Add the generated text to the current line

            # Check if the current line ends with a space
            if current_line.endswith(' '):
                f.write(output+"\n")
                current_line

        if (n % partial_guess == (partial_guess-1)):  # Helps keep track of progress
            print(f'{n} completed')
            # clears the strings already written to the file, 'next_char' maintains the last character for predictions
            result.clear()

    # After the loop, check if there's any remaining text in the current line
    if current_line:
        f.write(current_line)
        
    f.close()
    end = time.time()
    generate_time = end - start
    line = f"File: {specific_file} \nImporting Data Time: {g_importing_time} \nVectorizing Data Time: {g_vectorizing_time} \nGenerating Data: {number_of_guesses} \nGenerating Time: {generate_time} seconds\n\n"

    record = open('./runtime.txt', "a", encoding='utf-8')
    record.write(line)
    record.close()
    print(line)


def main():
    specific_file = '//globs//entropy_bin_1'  # Imported training set

    training = True
    generating = False
    num_layers = 2  # 1, 2, or 3 currently
    shutoff_when_done = False

    save_new_weight = False
    epoch_count = 1

    embedding_dim = 256  # The embedding dimension
    rnn_units = 512  # Number of RNN units
    number_of_lines = 2000  # Variable for array of starter words

    current_dir = os.getcwd()  # Set the file name for opening the file
    file_end = '//Source_Files//'+specific_file+'.txt'
    file_name = current_dir + file_end

    # Path of file for the checkpoints
    checkpoint_path = f'./model_weights/{num_layers} Hidden Layers/{specific_file}/model({rnn_units})({epoch_count})-{specific_file}'

    # Checkpoint to load if loading
    check_to_load = "./training_checkpoints/{num_layers} Hidden Layers/ckpt_{specific_file}-{epoch_count}"

    if (not os.path.exists(file_name)):
        print("File Name specified does not exist.")
        exit()
    if (save_new_weight and not os.path.exists(check_to_load)):
        print("Checkpoint to load does not exist.")
        exit()

    ### Proprocess data, get dataset ###
    ids_from_chars, chars_from_ids, dataset, vocab_size = preprocess_data(
        file_name, number_of_lines)

    ### Create model based on number of hidden layers desired ###
    model = create_rnn_model(vocab_size, embedding_dim, rnn_units, num_layers)

    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)

    model.summary()

    if (training):
        train_model(model, dataset, epoch_count, specific_file,
                    rnn_units, num_layers, save_new_weight, checkpoint_path, example_batch_predictions)

    if (generating):
        generate_text(model, starting_words, checkpoint_path, chars_from_ids,
                      ids_from_chars, length, number_of_lines, num_layers, specific_file)

    if (shutoff_when_done):
        os.system("python Helpers/shutdown_comp.py")


if __name__ == "__main__":
    main()
