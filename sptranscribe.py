import whisper
import os

# Load Whisper model once
whisper_model = whisper.load_model("medium")

# Function to transcribe separated speech
def transcribe_speech(speaker1_path, speaker2_path, output_dir):
    print("Transcribing speaker 1...")
    transcription_1 = whisper_model.transcribe(speaker1_path)["text"]

    print("Transcribing speaker 2...")
    transcription_2 = whisper_model.transcribe(speaker2_path)["text"]

    # Combine transcriptions
    full_transcription = f"Speaker 1: {transcription_1}\nSpeaker 2: {transcription_2}"

    # Save transcription
    transcription_file = os.path.join(output_dir, "transcription.txt")
    with open(transcription_file, "w") as file:
        file.write(full_transcription)

    print("Transcription complete.")

    return transcription_file  # Return the file path for Flask to use
