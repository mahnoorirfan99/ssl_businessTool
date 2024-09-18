from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
import streamlit as sl
import soundfile as sf
import pandas as pd
import json
import os

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

def save_transcript(transcript, filename):
     # Define the file path for the transcript
    transcript_path = os.path.join('transcripts', filename)

    # Write the transcript to the file
    with open(transcript_path, 'w') as f:
        f.write(transcript)

    print(f"Transcript saved as: {transcript_path}")


def main():
    sl.title("Audio Transcription Tool")
    model_path = "model/vosk-model-en-us-0.22-lgraph"
    
    uploaded_file = sl.file_uploader("Upload an audio file", type=["wav"])

    if uploaded_file is not None:
        
        transcript = transcribe_audio(uploaded_file, model_path)

        # Display transcription result in Streamlit
        sl.write("Transcription:")
        sl.text(transcript)

          # Ask the user to input a file name for the transcript
        transcript_filename = sl.text_input("Enter a name for the transcript file:", "transcript.txt")

        if sl.button("Save Transcript"):
            # Save the transcript to the 'transcripts' folder
            save_transcript(transcript, transcript_filename)
            sl.success(f"Transcript saved as: transcripts/{transcript_filename}")

# Call main function
if __name__ == "__main__":
    main()
    