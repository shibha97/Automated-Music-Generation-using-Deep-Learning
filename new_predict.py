
""" 
This module generates notes for a midi file using the trained neural network 
"""

import pickle
import numpy as np
from music21 import instrument, note, stream, chord
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, BatchNormalization, Activation
import os

NOTES_PATH = 'data/notes'
SEQUENCE_LENGTH = 100

def generate():
    """ Generate a piano midi file """
    # Load the notes used to train the model
    with open(NOTES_PATH, 'rb') as filepath:
        notes = pickle.load(filepath)

    pitchnames = sorted(set(notes))
    n_vocab = len(pitchnames)

    network_input, normalized_input = prepare_sequences(notes, pitchnames, n_vocab)
    model = create_network(normalized_input, n_vocab)
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)
    create_midi(prediction_output)

def prepare_sequences(notes, pitchnames, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    for i in range(len(notes) - SEQUENCE_LENGTH):
        sequence_in = notes[i:i + SEQUENCE_LENGTH]
        network_input.append([note_to_int[char] for char in sequence_in])

    n_patterns = len(network_input)
    normalized_input = np.reshape(network_input, (n_patterns, SEQUENCE_LENGTH, 1))
    normalized_input = normalized_input / float(n_vocab)

    return (network_input, normalized_input)

def create_network(normalized_input, n_vocab):
    """ Create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(512, input_shape=(normalized_input.shape[1], normalized_input.shape[2]), recurrent_dropout=0.3, return_sequences=True))
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

def generate_notes(model, network_input, pitchnames, n_vocab):
    """ Use the trained model to generate notes """
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    start = np.random.randint(0, len(network_input)-1)
    pattern = network_input[start]
    prediction_output = []

    for _ in range(500):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        prediction_output.append(int_to_note[index])
        pattern.append(index)
        pattern = pattern[1:]

    return prediction_output

def create_midi(prediction_output):
    """ Convert the output from the prediction to notes and create a midi file """
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes_obj = [note.Note(int(n)).setStoredInstrument(instrument.Piano()) for n in notes_in_chord]
            new_chord = chord.Chord(notes_obj)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern).setStoredInstrument(instrument.Piano())
            new_note.offset = offset
            output_notes.append(new_note)

        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='test_output.mid')

if __name__ == '__main__':
    generate()
