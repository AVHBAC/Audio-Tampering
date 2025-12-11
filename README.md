# README — Automated Deepfake Tampering Pipeline

This project automates the process of generating **tampered audio** by replacing segments of deepfake audio with matched real-speech segments. It uses:

* `splice_replace.py` — performs word-level alignment, segmentation, and splicing
* `create.sh` — batches the process across many directories
* Whisper or WhisperX for transcription
* Automatic temporary working directories
* Automatic retries and logging
# How It Works

For every detected **deepfake directiry**, the script:

1. Finds the file `resampled.wav` (the donor REAL audio).
2. Processes:

   * `audio0.wav`
   * `audio1.wav`
   * `audio2.wav`
3. Runs `splice_replace.py`:

   * Transcribes both audio files using whipserx (and if that fails using whisper)
   * Finds matching words by comparing the dictionaries that are output by the transcirption
   * Extracts matching word-segments by picking from them randomly
   * Splices real audio into the deepfake file
4. Saves the tampered output into:

```
<grandparent>/<parent>/
```

Example:

```
/results/xtts-clean/df_sub099/audio0.wav
→ Output goes to: xtts-clean/df_sub099/
```

5. Cleans up intermediate files automatically (`--cleanup`).

---

# Directory Structure Requirements

Your directories should have this structure, though you can change the names of these files by manually adjusting `create.sh`:

```
<root_set>/
    ├── <speaker_or_sample_dir>/
    │       ├── audio0.wav
    │       ├── audio1.wav
    │       ├── audio2.wav
    │       ├── resampled.wav     ← donor real audio
```

---

# Editing Input Directories

Open `create.sh` and edit the `dirs=( ... )` list:

```bash
dirs=(
    "/path/to/your/data-set-A"
    "/path/to/your/data-set-B"
    "/another/path"
)
```

Add or remove as many paths as you want.

Each directory must contain many subfolders, each containing the four audio files listed above.

---

# Running the Pipeline

To run the full processing pipeline and get extra timing, use:

```bash
{ time ./create.sh 1>/dev/null ; } 2>> runtimes.txt
```

Or run normally:

```bash
./create.sh
```

---

# Output & Logs

* **Tampered audio output** is saved into:

```
<grandparent>/<parent>/
```

* **process_log.txt**
  Contains warnings + failures.

* **runtimes.log**
  Stores total runtime of each full pass.

* **Intermediate temp dirs**
  Automatically created and removed (due to `--cleanup`).

---

# Automatic Retries

Each audio file (audio0/1/2) will retry up to **3 times** if:

* WhisperX fails
* alignment fails
* splicing errors occur

This helps stabilize processing in large batches.

---

# Dependencies

You must install the required Python dependencies in the environment where you run the script. This can be done in the following method:

```bash
conda create -n tamper python=3.11
conda activate tamper
pip install -U openai-whisper whisperx librosa soundfile numpy pydub tqdm
```
