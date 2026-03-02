import torch
import torchaudio
import torchaudio.transforms as T
import os
from pathlib import Path

from speechbrain.inference.separation import SepformerSeparation as separator
# Model paths
MODEL_PATH = "E:\Final main project UI\Speech separation\model\dprnn_model_10sec.pth"
MODEL_DIR = "E:\Final main project UI\Speech separation\model\Tmpdir"

Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

# Load the model architecture
model = separator.from_hparams(
    source=MODEL_DIR,
    savedir=MODEL_DIR,
    run_opts={"download": False}
)
# Load the state_dict
state_dict = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state_dict)

# Ensure the model is in evaluation mode
model.eval()

# Target sample rate
TARGET_SR = 8000

def separate_speech(audio_path, output_dir):
    # Load the input audio
    waveform, sample_rate = torchaudio.load(audio_path)

    # Resample if necessary
    if sample_rate != TARGET_SR:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=TARGET_SR)
        waveform = resampler(waveform)
        print(f"Input audio resampled from {sample_rate} Hz to {TARGET_SR} Hz.")

    # Perform speech separation
    waveform = waveform.squeeze(1)
    est_sources = model.separate_batch(waveform.unsqueeze(0))  # Add batch dimension
    est_sources = est_sources.squeeze(0)  # Remove batch dimension

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define output file paths
    speaker1_path = os.path.join(output_dir, "speaker1.wav")
    speaker2_path = os.path.join(output_dir, "speaker2.wav")

    # Save separated speakers
    torchaudio.save(speaker1_path, est_sources[0].unsqueeze(0), TARGET_SR)
    torchaudio.save(speaker2_path, est_sources[1].unsqueeze(0), TARGET_SR)

    print("Overlapping speech separation complete.")

    return speaker1_path, speaker2_path
