from flask import Flask, render_template, request, send_from_directory, redirect, url_for
import os
import torch
import librosa
import soundfile as sf
from separator import separate_speech
from sptranscribe import transcribe_speech
from summarize import summarize_text

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
TRANSCRIPT_FOLDER = "transcriptions"
SUMMARY_FOLDER = "summaries"

for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, TRANSCRIPT_FOLDER, SUMMARY_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
app.config["TRANSCRIPT_FOLDER"] = TRANSCRIPT_FOLDER
app.config["SUMMARY_FOLDER"] = SUMMARY_FOLDER

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return "No file part"
    
    file = request.files["file"]
    if file.filename == "":
        return "No selected file"
    
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # Run Speech Separation
    output_files = separate_speech(filepath, app.config["OUTPUT_FOLDER"])

    return render_template("results.html", files=output_files)

@app.route("/download/<path:filename>")
def download_file(filename):
    return send_from_directory(directory=app.config["OUTPUT_FOLDER"], path=filename, as_attachment=True)

@app.route("/transcribe/<filename>")
def transcribe(filename):
    input_path = os.path.join(app.config["OUTPUT_FOLDER"], filename)
    transcript = transcribe_speech(input_path)

    transcript_path = os.path.join(app.config["TRANSCRIPT_FOLDER"], f"{filename}.txt")
    with open(transcript_path, "w") as f:
        f.write(transcript)

    return render_template("results.html", transcript=transcript, filename=filename)

@app.route("/summarize/<filename>")
def summarize(filename):
    transcript_path = os.path.join(app.config["TRANSCRIPT_FOLDER"], f"{filename}.txt")
    with open(transcript_path, "r") as f:
        text = f.read()

    summary = summarize_text(text)

    summary_path = os.path.join(app.config["SUMMARY_FOLDER"], f"{filename}_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)

    return render_template("results.html", summary=summary, filename=filename)

if __name__ == "__main__":
    app.run(debug=True)
