from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
from transformers import pipeline
import streamlit as sl
import soundfile as sf
import pandas as pd
import sqlite3
import json
import textwrap
import os



con = sqlite3.connect("transcripts/transcript.db")

cur = con.cursor()

cur.execute('''
    CREATE TABLE IF NOT EXISTS transcripts (
        id INTEGER PRIMARY KEY,
        filename TEXT NOT NULL,
        transcript TEXT NOT NULL
    )
''')
con.commit()


pipe = pipeline("summarization", model="Falconsai/text_summarization")

wrapper = textwrap.TextWrapper(width=80, initial_indent='')

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

    final_result = recognizer.FinalResult()
    transcript += json.loads(final_result)['text']
    
    wrapped_transcript = wrapper.fill(transcript)
    
    return wrapped_transcript


def save_transcript_to_db(transcript, filename):
    cur.execute("INSERT INTO transcripts (filename, transcript) VALUES (?, ?)", (filename, transcript))
    con.commit()
    print(f"Transcript saved to database with filename: {filename}")


#def save_transcript(transcript, filename):
     # Define the file path for the transcript
 #   transcript_path = os.path.join('transcripts', filename)

  #  with open(transcript_path, 'w') as f:
   #     f.write(transcript)

    #print(f"Transcript saved as: {transcript_path}")

def summarize_text(selected_text):
    summary = pipe(selected_text, max_length=130, min_length=30, do_sample=True)
    wrapped_summary = wrapper.fill(summary[0]['summary_text'])
    
    return wrapped_summary


def main():
    sl.title("Audio Transcription Tool")
    model_path = "model/vosk-model-en-us-0.22-lgraph"
    
    uploaded_file = sl.file_uploader("Upload an audio file", type=["wav"])

    if uploaded_file is not None:
        
        transcript = transcribe_audio(uploaded_file, model_path)

        sl.write("Transcription:")
        sl.text(transcript)

        #Ask the user to input a file name for the transcript
        transcript_filename = sl.text_input("Enter a name for the transcript file:", "transcript.txt")

        if sl.button("Save Transcript"):
            save_transcript_to_db(transcript, transcript_filename)
            sl.success(f"Transcript saved in database as: transcripts/{transcript_filename}")

        selected_text = sl.text_area("Select a block of text to summarize", "")
        if sl.button("Summarize Text"):
            if selected_text:
                summary = summarize_text(selected_text)
                sl.write("Summary:")
                sl.text(summary)
            else:
                sl.warning("Please select some text")


# Call main function
if __name__ == "__main__":
    main()

con.close()
    