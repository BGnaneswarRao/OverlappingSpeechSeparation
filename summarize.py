from transformers import pipeline
import os

# Load summarization model once
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Function to summarize transcriptions
def summarize_text(transcription_file, output_dir):
    # Read the transcription file
    with open(transcription_file, "r") as file:
        full_transcription = file.read()

    print("Summarizing the conversation...")
    summary = summarizer(full_transcription, max_length=100, min_length=30, do_sample=False)[0]["summary_text"]

    # Save summary
    summary_file = os.path.join(output_dir, "summary.txt")
    with open(summary_file, "w") as file:
        file.write(summary)

    print("Summarization complete.")

    return summary_file  # Return the file path for Flask to use
