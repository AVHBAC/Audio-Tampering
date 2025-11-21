#!/usr/bin/env python3
"""
splice_replace.py

Replace matching words in a target (deepfake) audio with matching words from a donor (real) audio.

Usage:
    python splice_replace.py --target deepfake.wav --donor real.wav --outdir out_dir

Options:
    --n_replace N        number of replacements (default 10)
    --cleanup            remove the temp working directory after successful run
    --keep_temp_name NAME set a custom name for the temp working dir instead of auto-generated

Dependencies:
    pip install -U openai-whisper whisperx librosa soundfile numpy pydub tqdm
"""
import argparse
import os
import json
import random
import math
import shutil
import tempfile
import datetime
from typing import List, Dict, Tuple
import numpy as np
import soundfile as sf
from tqdm import tqdm
import librosa

# optional libs for transcription
try:
    import whisperx
    HAS_WHISPERX = True
except Exception:
    HAS_WHISPERX = False

try:
    import whisper
    HAS_WHISPER = True
except Exception:
    HAS_WHISPER = False

# conservative list of short function words to avoid
STOPWORDS = {
    "a","an","the","and","or","but","if","is","are","was","were","am","be",
    "to","of","in","on","for","with","at","by","from","as","that","this","it",
    "i","you","he","she","they","we","me","him","her","them","us"
}

# ---- Utilities ----
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def copy_to_dir(src: str, dst_dir: str):
    ensure_dir(dst_dir)
    dst = os.path.join(dst_dir, os.path.basename(src))
    shutil.copy2(src, dst)
    return dst

def load_audio_mono(path: str, sr: int):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y

def write_audio(path: str, y: np.ndarray, sr: int):
    sf.write(path, y.astype(np.float32), sr)

def now_iso():
    return datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

# approximate median f0 using librosa.pyin
def median_f0(y: np.ndarray, sr: int):
    try:
        f0, _, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
        f = f0[~np.isnan(f0)]
        return float(np.median(f)) if f.size>0 else None
    except Exception:
        return None

def rms(y: np.ndarray):
    return math.sqrt(float(np.mean(y**2))) if y.size>0 else 0.0

def match_duration(donor: np.ndarray, donor_sr: int, target_duration_s: float):
    donor_dur = donor.shape[0] / donor_sr
    if donor_dur <= 0:
        return donor
    rate = donor_dur / target_duration_s
    if abs(rate - 1.0) < 0.01:
        return donor
    try:
        # librosa 0.10+ requires keyword args
        out = librosa.effects.time_stretch(y=donor.astype(float), rate=rate)
        return out
    except Exception:
        new_len = int(round(target_duration_s * donor_sr))
        # librosa 0.10+ also requires keyword args
        return librosa.util.fix_length(data=donor, size=new_len)

def shift_pitch(donor: np.ndarray, sr: int, semitones: float):
    if abs(semitones) < 0.1:
        return donor
    try:
        return librosa.effects.pitch_shift(donor.astype(float), sr, n_steps=semitones)
    except Exception:
        return donor

def match_pitch_and_energy(donor: np.ndarray, target: np.ndarray, sr: int):
    fd = median_f0(donor, sr)
    ft = median_f0(target, sr)
    out = donor
    if fd and ft and fd > 0 and ft > 0:
        semitones = 12 * math.log2(ft / fd)
        out = shift_pitch(out, sr, semitones)
    rms_d = rms(out)
    rms_t = rms(target)
    if rms_d > 0 and rms_t > 0:
        out = out * (rms_t / (rms_d + 1e-9))
    return out

def splice_into_target(target, sr, t_start, t_end, donor):
    start = int(round(t_start * sr))
    end   = int(round(t_end * sr))

    # split target
    left  = target[:start]
    right = target[end:]

    # pad donor if needed
    donor_len = donor.shape[0]
    needed = (end - start)

    if donor_len < needed:
        pad = needed - donor_len
        donor = librosa.util.fix_length(data=donor, size=needed)

    # pad left and right if needed (this is where your error was)
    if start < 0:
        pad_before = -start
        left = librosa.util.fix_length(data=left, size=pad_before)

    if end > len(target):
        pad_after = end - len(target)
        right = librosa.util.fix_length(data=right, size=pad_after)

    return np.concatenate([left, donor, right])

# ---- Transcription & alignment ----
def transcribe_with_whisperx(audio_path: str, model_name='medium', device='cpu', language=None):
    import whisperx as wx

    # CPU cannot run float16 → force int8 for WhisperX
    compute_type = "int8" if device == "cpu" else "float16"

    model = wx.load_model(
        model_name,
        device=device,
        compute_type=compute_type
    )

    result = model.transcribe(audio_path, language=language)

    model_a, metadata = wx.load_align_model(
        language_code=result["language"],
        device=device,
    )

    aligned = wx.align(
        result["segments"], model_a, metadata, audio_path, device
    )

    word_ts = []
    for seg in aligned['segments']:
        for w in seg.get('words', []):
            word_ts.append({
                'word': w['word'].strip().lower(),
                'start': float(w['start']),
                'end': float(w['end'])
            })
    return word_ts, aligned


def transcribe_with_whisper_fallback(audio_path: str, model_name='small', sr=16000, language=None):
    import whisper
    model = whisper.load_model(model_name)
    res = model.transcribe(audio_path, language=language)
    word_ts = []
    for seg in res['segments']:
        text = seg['text'].strip()
        if text == "":
            continue
        words = text.split()
        seg_start = float(seg['start'])
        seg_end = float(seg['end'])
        seg_dur = seg_end - seg_start
        if len(words) == 0:
            continue
        per = seg_dur / len(words)
        for i, w in enumerate(words):
            w_clean = w.strip().lower()
            start = seg_start + i * per
            end = start + per
            word_ts.append({'word': w_clean, 'start': start, 'end': end})
    return word_ts, res

def get_word_timestamps(audio_path: str, prefer_whisperx=True, model_name_whisperx='medium', model_name_whisper='small', device='cpu', language=None):
    if prefer_whisperx and HAS_WHISPERX:
        try:
            return transcribe_with_whisperx(
                audio_path,
                model_name=model_name_whisperx,
                device=device,
                language=language
            )
        except Exception as e:
            print("whisperx failed, falling back to whisper:", e)
    if HAS_WHISPER:
        return transcribe_with_whisper_fallback(audio_path, model_name_whisper, sr=16000, language=language)
    raise RuntimeError("No transcription backend available. Install `whisper` (and optionally `whisperx`).")

# ---- candidate matching ----
def find_matching_words(target_ts: List[Dict], donor_ts: List[Dict], min_duration_s=0.12):
    donor_dict = {}
    for d in donor_ts:
        w = d.get('word','').strip()
        if len(w)==0: continue
        donor_dict.setdefault(w, []).append(d)
    candidates = []
    for t in target_ts:
        w = t.get('word','').strip()
        if len(w)==0: continue
        if w in STOPWORDS:
            continue
        dur = t['end'] - t['start']
        if dur < min_duration_s:
            continue
        if w in donor_dict:
            for d in donor_dict[w]:
                d_dur = d['end'] - d['start']
                if d_dur < 0.05:
                    continue
                candidates.append((t, d))
    return candidates

def choose_replacements(candidates: List[Tuple[Dict,Dict]], n_replace: int, seed: int = 0):
    random.seed(seed)
    if not candidates:
        return []
    # deduplicate by target region
    unique_by_target = {}
    for t,d in candidates:
        key = (round(t['start'],3), round(t['end'],3))
        unique_by_target.setdefault(key, []).append((t,d))
    uniq_list = [v[0] for v in unique_by_target.values()]
    random.shuffle(uniq_list)
    return uniq_list[:n_replace]

# ---- main orchestration ----
def main():
    p = argparse.ArgumentParser(description="Replace matching words from donor audio into target audio (with temp working dir).")
    p.add_argument("--target", "-t", required=True, help="Path to target (deepfake) wav file")
    p.add_argument("--donor", "-d", required=True, help="Path to donor (real) wav file")
    p.add_argument("--outdir", "-o", required=True, help="Output directory")
    p.add_argument("--n_replace", "-n", type=int, default=10, help="Number of replacements (default 10)")
    p.add_argument("--sr", type=int, default=16000, help="Audio sample rate (default 16000)")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--min_word_dur_ms", type=int, default=120, help="Minimum target word duration in ms")
    p.add_argument("--prefer_whisperx", action="store_true", help="If set, try whisperx first (recommended if installed)")
    p.add_argument("--cleanup", action="store_true", help="If set, remove temp working directory after successful run")
    p.add_argument("--keep_temp_name", type=str, default=None, help="If set, use this folder name (inside outdir) for temp workdir")
    p.add_argument("--model_whisperx", default="medium", help="whisperx model size (if used)")
    p.add_argument("--model_whisper", default="small", help="whisper model size (fallback)")
    args = p.parse_args()

    ensure_dir(args.outdir)

    # create a temp working dir inside outdir, with timestamp
    if args.keep_temp_name:
        temp_dir = os.path.join(args.outdir, args.keep_temp_name)
        ensure_dir(temp_dir)
    else:
        temp_dir = os.path.join(args.outdir, f"temp_work_{now_iso()}_{random.randint(1000,9999)}")
        ensure_dir(temp_dir)

    print(f"[+] Temp working dir: {temp_dir}")

    # copy inputs into temp dir and work on copies
    try:
        target_copy = copy_to_dir(args.target, temp_dir)
        donor_copy = copy_to_dir(args.donor, temp_dir)
    except Exception as e:
        print("Failed to copy input files to temp dir:", e)
        raise

    # save a small run-info file
    run_info = {
        'timestamp_utc': now_iso(),
        'original_target': os.path.abspath(args.target),
        'original_donor': os.path.abspath(args.donor),
        'target_copy': os.path.abspath(target_copy),
        'donor_copy': os.path.abspath(donor_copy),
        'params': {k: str(v) for k, v in vars(args).items()}
    }
    with open(os.path.join(temp_dir, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2)

    # load copies for processing
    sr = args.sr
    print("[+] Loading audio (copies)...")
    target_y = load_audio_mono(target_copy, sr=sr)
    donor_y = load_audio_mono(donor_copy, sr=sr)

    # transcribe & align (save transcription to temp dir)
    print("[+] Transcribing target...")
    try:
        target_ts, target_full = get_word_timestamps(target_copy, prefer_whisperx=args.prefer_whisperx and HAS_WHISPERX,
                                                     model_name_whisperx=args.model_whisperx, model_name_whisper=args.model_whisper,
                                                     device='cpu')
    except Exception as e:
        print("Transcription failed for target:", e)
        raise
    with open(os.path.join(temp_dir, "target_transcription.json"), "w") as f:
        json.dump({'words': target_ts}, f, indent=2)

    print("[+] Transcribing donor...")
    try:
        donor_ts, donor_full = get_word_timestamps(donor_copy, prefer_whisperx=args.prefer_whisperx and HAS_WHISPERX,
                                                   model_name_whisperx=args.model_whisperx, model_name_whisper=args.model_whisper,
                                                   device='cpu')
    except Exception as e:
        print("Transcription failed for donor:", e)
        raise
    with open(os.path.join(temp_dir, "donor_transcription.json"), "w") as f:
        json.dump({'words': donor_ts}, f, indent=2)

    print(f"[+] Found {len(target_ts)} words in target, {len(donor_ts)} words in donor.")

    min_dur_s = args.min_word_dur_ms / 1000.0
    candidates = find_matching_words(target_ts, donor_ts, min_duration_s=min_dur_s)
    print(f"[+] {len(candidates)} candidate matching word pairs (after filters).")
    if not candidates:
        print("No valid candidates found; exiting.")
        if args.cleanup:
            shutil.rmtree(temp_dir, ignore_errors=True)
        return

    chosen = choose_replacements(candidates, args.n_replace, seed=args.seed)
    print(f"[+] Selected {len(chosen)} replacements.")

    out_y = target_y.copy()
    metadata = {'target_copy': target_copy, 'donor_copy': donor_copy, 'replacements': []}

    # apply replacements; save intermediate extracted segments for auditing
    extracted_dir = os.path.join(temp_dir, "extracted_segments")
    processed_dir = os.path.join(temp_dir, "processed_segments")
    ensure_dir(extracted_dir)
    ensure_dir(processed_dir)

    for idx, (t, d) in enumerate(tqdm(chosen, desc="Applying replacements")):
        t_start, t_end = float(t['start']), float(t['end'])
        d_start, d_end = float(d['start']), float(d['end'])
        word_label = t.get('word','unknown')

        s_t = int(round(t_start * sr))
        e_t = int(round(t_end * sr))
        s_d = int(round(d_start * sr))
        e_d = int(round(d_end * sr))

        target_seg = out_y[s_t:e_t] if e_t > s_t else np.array([])
        donor_seg = donor_y[s_d:e_d] if e_d > s_d else np.array([])

        # save raw excerpts
        try:
            if target_seg.size>0:
                target_fn = os.path.join(extracted_dir, f"{idx:02d}_target_{word_label}_{s_t}_{e_t}.wav")
                write_audio(target_fn, target_seg, sr)
            else:
                target_fn = None
            if donor_seg.size>0:
                donor_fn = os.path.join(extracted_dir, f"{idx:02d}_donor_{word_label}_{s_d}_{e_d}.wav")
                write_audio(donor_fn, donor_seg, sr)
            else:
                donor_fn = None
        except Exception as e:
            print("Failed saving extracted segments:", e)

        if donor_seg.size == 0 or target_seg.size == 0:
            print(f"Skipping idx {idx} because one of segments is empty.")
            continue

        # duration match
        donor_matched = match_duration(donor_seg, sr, (t_end - t_start))
        # pitch and energy match
        donor_matched = match_pitch_and_energy(donor_matched, target_seg, sr)

        # save processed donor matched segment
        try:
            proc_fn = os.path.join(processed_dir, f"{idx:02d}_donor_matched_{word_label}.wav")
            write_audio(proc_fn, donor_matched, sr)
        except Exception as e:
            print("Failed writing processed donor segment:", e)
            proc_fn = None

        # splice
        out_y = splice_into_target(out_y, sr, t_start, t_end, donor_matched)

        metadata['replacements'].append({
            'index': idx,
            'word': word_label,
            'target_start': t_start,
            'target_end': t_end,
            'donor_start': d_start,
            'donor_end': d_end,
            'target_segment_file': os.path.relpath(target_fn, temp_dir) if target_fn else None,
            'donor_segment_file': os.path.relpath(donor_fn, temp_dir) if donor_fn else None,
            'donor_matched_file': os.path.relpath(proc_fn, temp_dir) if proc_fn else None,
            'applied_duration_s': donor_matched.shape[0] / sr if donor_matched is not None else None
        })

    # write final tampered audio and metadata into outdir (and copies into temp)
    out_basename = os.path.basename(args.target).rsplit('.',1)[0] + "_tampered.wav"
    out_path = os.path.join(args.outdir, out_basename)
    write_audio(out_path, out_y, sr)

    # also copy final into temp_dir
    final_copy = os.path.join(temp_dir, os.path.basename(out_path))
    write_audio(final_copy, out_y, sr)

    meta = {
        'out_tampered': os.path.abspath(out_path),
        'out_tampered_copy': os.path.abspath(final_copy),
        'metadata': metadata
    }
    meta_path = os.path.join(args.outdir, os.path.basename(out_path).rsplit('.',1)[0] + "_metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(temp_dir, "metadata.json"), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"[+] Wrote tampered audio to: {out_path}")
    print(f"[+] Wrote metadata to: {meta_path}")
    print(f"[+] Intermediate artifacts saved in: {temp_dir}")

    if args.cleanup:
        try:
            shutil.rmtree(temp_dir)
            print("[+] Cleaned up temp dir.")
        except Exception as e:
            print("Failed to remove temp dir:", e)

if __name__ == "__main__":
    main()
