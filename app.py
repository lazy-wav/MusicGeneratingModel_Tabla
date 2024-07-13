import streamlit as st
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import soundfile as sf
import numpy as np
import librosa
import librosa.display
import os


def define_model():
    model = Sequential([
        LSTM(units=256, input_shape=(1, 40), return_sequences=True),
        Dropout(0.3),
        LSTM(units=256, return_sequences=True),
        Dropout(0.3),
        LSTM(units=256, return_sequences=True),
        Dropout(0.3),
        LSTM(units=256, return_sequences=False),  
        Dense(units=40)
    ])
    return model


def load_model_weights(model, weights_path):
    model.load_weights(weights_path)
    return model

# load the model architecture
model = define_model()

weights_path = 'model/music_generation_model.h5'
if os.path.exists(weights_path):
    try:
        model = load_model_weights(model, weights_path)
    except ValueError as e:
        st.error(f"Error loading model weights: {e}")
else:
    st.error(f"Model weights file '{weights_path}' not found.")



# generate sequence of seeds
def generate_sequence(model, seed, sequence_length):
    generated_sequence = []
    current_sequence = seed
    for _ in range(sequence_length):
        prediction = model.predict(current_sequence)
        generated_sequence.append(prediction[0])
        prediction = np.pad(prediction, ((0, 0), (0, 40 - prediction.shape[1])), 'constant')
        current_sequence = np.concatenate((current_sequence[:, 1:, :], prediction.reshape(1, 1, 40)), axis=1)
    return np.array(generated_sequence)


# convert to .wav

def mfcc_to_audio(mfcc_sequence, sample_rate=22050):
    audio = librosa.feature.inverse.mfcc_to_audio(mfcc_sequence.T, sr=sample_rate, n_iter=512)
    return audio

st.title('Music Generation using LSTM')
st.write('Generate music using a trained LSTM model.')

seed_input = st.text_input('Enter seed sequence (comma-separated integers):', '1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40')
sequence_length = st.number_input('Enter sequence length:', min_value=10, max_value=2000, value=100, step=10)

if st.button('Generate Music'):
    seed_sequence = np.array([int(i) for i in seed_input.split(',')]).reshape(1, 1, -1)
    generated_sequence = generate_sequence(model, seed_sequence, sequence_length)
    generated_audio = mfcc_to_audio(generated_sequence)
    sf.write('generated_music.wav', generated_audio, samplerate=22050)
    st.audio('generated_music.wav', format='audio/wav')
    st.success('Music generated and saved as generated_music.wav')
