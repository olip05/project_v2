
from pathlib import Path

import pandas as pd
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Configuration
# --------------------------------------------------

CLIPS_DIR = Path("dataset")

CSV_FILES = {
    "train": Path("dataset/geo/train.csv"),
    "dev": Path("dataset/geo/dev.csv"),
    "test": Path("dataset/geo/test.csv"),
}

FILENAME_COLUMN = "file"
TRANSCRIPT_COLUMN = "transcript"

# --------------------------------------------------
# Analysis
# --------------------------------------------------

records = []

for partition, csv_path in CSV_FILES.items():

    data = pd.read_csv(csv_path)
    data[TRANSCRIPT_COLUMN] = data[TRANSCRIPT_COLUMN].fillna("")

    for _, row in data.iterrows():

        filename = str(row[FILENAME_COLUMN])
        transcript = str(row[TRANSCRIPT_COLUMN])

        audio_path = CLIPS_DIR / filename

        if not audio_path.exists():
            print(f"Missing audio: {audio_path}")
            continue

        # Audio properties
        info = sf.info(audio_path)

        duration = info.duration
        sample_rate = info.samplerate
        channels = info.channels

        bit_depth = None

        if info.subtype.startswith("PCM_"):
            bit_depth = int(info.subtype.split("_")[1])

        # Transcript properties
        word_count = len(transcript.split())
        character_count = len(transcript)

        records.append({
            "partition": partition,
            "filename": filename,
            "duration": duration,
            "sample_rate": sample_rate,
            "channels": channels,
            "bit_depth": bit_depth,
            "word_count": word_count,
            "character_count": character_count,
            "transcript": transcript,
        })

# --------------------------------------------------
# DataFrame
# --------------------------------------------------

df = pd.DataFrame(records)

if df.empty:
    raise ValueError("No matching audio files found.")

# --------------------------------------------------
# Summary statistics: min, max, mean
# --------------------------------------------------

print("\n========== DATASET SUMMARY ==========\n")

for partition in CSV_FILES:

    subset = df[df["partition"] == partition]

    if subset.empty:
        continue

    print(f"\nPartition: {partition}")
    print(f"Samples: {len(subset)}")

    # Audio duration
    print("\nAudio duration (seconds):")
    print(f"Min:  {subset['duration'].min():.2f}")
    print(f"Max:  {subset['duration'].max():.2f}")
    print(f"Mean: {subset['duration'].mean():.2f}")

    # Transcript length
    print("\nTranscript length (words):")
    print(f"Min:  {subset['word_count'].min()}")
    print(f"Max:  {subset['word_count'].max()}")
    print(f"Mean: {subset['word_count'].mean():.2f}")

    # Audio properties
    print("\nSampling rate (Hz):")
    print(f"Min:  {subset['sample_rate'].min()}")
    print(f"Max:  {subset['sample_rate'].max()}")
    print(f"Mean: {subset['sample_rate'].mean():.0f}")

    print("\nBit depth:")
    print(f"Min:  {subset['bit_depth'].min()}")
    print(f"Max:  {subset['bit_depth'].max()}")
    print(f"Mean: {subset['bit_depth'].mean():.2f}")

    print("\nChannels:")
    print(f"Min:  {subset['channels'].min()}")
    print(f"Max:  {subset['channels'].max()}")
    print(f"Mean: {subset['channels'].mean():.2f}")

# --------------------------------------------------
# Save detailed results
# --------------------------------------------------

df.to_csv("dataset_analysis.csv", index=False)

# --------------------------------------------------
# Combined histograms
# --------------------------------------------------

partitions = list(CSV_FILES.keys())

# Audio duration histogram
# Get common bin edges across all partitions
all_durations = df["duration"].dropna()
duration_bins = np.histogram_bin_edges(all_durations, bins=20)

plt.figure(figsize=(10, 6))

for partition in ["train", "dev", "test"]:
    values = df.loc[df["partition"] == partition, "duration"].dropna()

    plt.hist(
        values,
        bins=duration_bins,
        alpha=0.5,
        label=partition,
        histtype="bar",
        rwidth=1
    )

# plt.xlabel("Duration (seconds)")
# plt.ylabel("Number of samples")
# plt.title("Audio duration distribution by partition")
# plt.legend()
# plt.tight_layout()
# plt.savefig("audio_duration_histogram.png", dpi=300)
# plt.show()

plt.xlabel("Audio length [s]")
plt.ylabel("Number of samples")
plt.legend()

plt.tight_layout()
plt.savefig("audio_duration_histogram.png")
plt.close()

# Transcript length histogram
# plt.figure(figsize=(9, 5))

# for partition in partitions:


#     subset = df[df["partition"] == partition]

#     if not subset.empty:
#         plt.hist(
#             subset["word_count"],
#             bins=30,
#             alpha=0.5,
#             label=partition
#         )


all_lengths = df["word_count"].dropna()
length_bins = np.histogram_bin_edges(all_lengths, bins=10)

plt.figure(figsize=(10, 6))

for partition in ["train", "dev", "test"]:
    if partition == "test":
        continue
    values = df.loc[df["partition"] == partition, "word_count"].dropna()

    plt.hist(
        values,
        bins=length_bins,
        alpha=0.5,
        label=partition,
        histtype="bar",
        rwidth=1
    )

plt.xlabel("Transcript length [words]")
plt.ylabel("Number of samples")
# plt.title("Transcript length distribution")
plt.legend()

plt.tight_layout()
plt.savefig("transcript_length_histogram.png")
plt.close()

print("\nAnalysis completed.")
print("Results saved to dataset_analysis.csv")
print("Histograms saved.")
