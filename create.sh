#!/bin/bash
set -euo pipefail

# Run with: { time ./create.sh 1>/dev/null ; } 2>> runtimes.txt

# Start time
start=$(date +%s)

# Directories to search — add as many as you like
dirs=(
    "/paths/to/your/files"
    "/paths/to/your/files"
)

# Log file
logfile="process_log.txt"
: > "$logfile"   # Clear existing log file

# Max retry attempts
max_attempts=3

# Loop over directories
for dirpath in "${dirs[@]}"; do
    find "$dirpath" -type f -name "resampled.wav" | while read -r file; do
        
        filename=$(realpath "$file")
        dir=$(dirname "$filename")
        parent=$(basename "$dir")
        grandparent=$(basename "$(dirname "$dir")")

        # List of target audio files to process
        targets=(
            "$dir/audio0.wav"
            "$dir/audio1.wav"
            "$dir/audio2.wav"
        )

        for target in "${targets[@]}"; do
            
            # Skip missing audio files
            if [[ ! -f "$target" ]]; then
                echo "Skipping missing target: $target"
                continue
            fi

            attempt=1
            success=0

            while (( attempt <= max_attempts )); do
                echo "Processing: $target (Attempt $attempt/$max_attempts)"

                if python splice_replace.py \
                        --target "$target" \
                        --donor "$dir/resampled.wav" \
                        --outdir "$grandparent/$parent" \
                        --prefer_whisperx \
                        --cleanup; then

                    echo "✅ Success: $target"
                    success=1
                    break

                else
                    if (( attempt < max_attempts )); then
                        echo "⚠️ WARNING: $target failed (attempt $attempt)" | tee -a "$logfile"
                        (( attempt++ ))
                        sleep 1
                    else
                        echo "❌ ERROR: $target failed after $max_attempts attempts" | tee -a "$logfile"
                        break
                    fi
                fi
            done
        done
    done
done

# End time
end=$(date +%s)
runtime=$(( end - start ))

echo "All processing complete. Logs written to $logfile"
echo "Runtime: ${runtime} seconds" >> runtimes.log
