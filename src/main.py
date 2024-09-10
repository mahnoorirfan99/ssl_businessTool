from vosk import Model, KaldiRecognizer
import streamlit as sl
import soundfile as sf
import pandas as pd
import json
from pydub import AudioSegment
import numpy as np


def transcribe_audio(input_audio_path, model_path, buffer_size=4000):
    # Load Vosk model
    model = Model(model_path)
    

    audio = AudioSegment.from_file(input_audio_path).set_frame_rate(16000).set_channels(1).set_sample_width(2)
    audio_data = audio.raw_data
    
    # Initialize recognizer with the model and sample rate
    recognizer = KaldiRecognizer(model, 16000)
    
    
    transcript = ""
    
    # Process the audio data in chunks
    for i in range(0, len(audio_data), buffer_size):
        if recognizer.AcceptWaveform(audio_data[i:i+buffer_size]):
            result = recognizer.Result()
            transcript += json.loads(result)['text'] + " "
    
    # Get final transcription result
    final_result = recognizer.FinalResult()
    transcript += json.loads(final_result)['text']
    
    return transcript

def main():
    input_audio_path = "uploads/tester.wav"
    model_path = "model/vosk-model-small-en-us-0.15/vosk-model-small-en-us-0.15"
    
    # Call the transcription function
    transcript = transcribe_audio(input_audio_path, model_path)
    print("Transcription:", transcript)

# Call main function
if __name__ == "__main__":
    main()