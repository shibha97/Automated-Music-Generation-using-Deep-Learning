
""" 
This module prepares midi file data and feeds it to the neural network for training 
"""

import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation, BatchNormalization
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os

MIDI_FOLDER = "midi_songs"
NOTES_PATH = 'data/notes'
SEQUENCE_LENGTH = 100

def get_notes_from_midi(file_path):
    """ Extracts notes and chords from a given MIDI file """
    notes = []

    midi = converter.parse(file_path)
    try:  # file has instrument parts
        parts = instrument.partitionByInstrument(midi)
        notes_to_parse = parts.parts[0].recurse()
    except AttributeError:  # file has notes in a flat structure
        notes_to_parse = midi.flat.notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))
    
    return notes

def train_network():
    """ Train a Neural Network to generate music """
    notes = get_all_notes_from_midis(MIDI_FOLDER)
    n_vocab = len(set(notes))
    network_input, network_output = prepare_sequences(notes, n_vocab)
    model = create_network(network_input, n_vocab)
    train(model, network_input, network_output)

def get_all_notes_from_midis(directory):
    """ Get all the notes and chords from the midi files in the specified directory """
    all_notes = []

    for file in glob.glob(os.path.join(directory, "*.mid")):
        print(f"Parsing {file}")
        all_notes.extend(get_notes_from_midi(file))

    with open(NOTES_PATH, 'wb') as filepath:
        pickle.dump(all_notes, filepath)

    return all_notes

def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    pitchnames = sorted(set(item for item in notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - SEQUENCE_LENGTH, 1):
        sequence_in = notes[i:i + SEQUENCE_LENGTH]
        sequence_out = notes[i + SEQUENCE_LENGTH]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, SEQUENCE_LENGTH, 1))
    # normalize input
    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)

def create_network(network_input, n_vocab):
    """ Create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(512, input_shape=(network_input.shape[1], network_input.shape[2]), recurrent_dropout=0.3, return_sequences=True))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3))
    model.add(LSTM(512))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

def train(model, network_input, network_output, epochs=200, batch_size=128):
    """ Train the neural network """
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')
    early_stop = EarlyStopping(monitor='loss', patience=10, verbose=0, mode='min')
    callbacks_list = [checkpoint, early_stop]

    model.fit(network_input, network_output, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)

if __name__ == '__main__':
    train_network()
