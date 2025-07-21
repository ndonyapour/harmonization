#!/bin/bash

INPUT_DIR="/Users/donyapourn2/Desktop/projects/datasets/ADNI/t1_mpr_n4_corrected"
OUTPUT_DIR="/Users/donyapourn2/Desktop/projects/datasets/ADNI/t1_mpr_fullbrain_mask"
mkdir -p "$OUTPUT_DIR"

for input_image in "$INPUT_DIR"/*.nii.gz; do
    filename=$(basename "$input_image" .nii.gz)
    echo "Processing $filename..."
    
    subject_dir="$OUTPUT_DIR/$filename"
    mkdir -p "$subject_dir"

    # Step 1: Generate brain mask with balanced parameters
    ThresholdImage 3 "$input_image" "$subject_dir/${filename}_otsu.nii.gz" Otsu 4
    ThresholdImage 3 "$subject_dir/${filename}_otsu.nii.gz" "$subject_dir/${filename}_mask.nii.gz" 2 4 1 0

    # Cleaning
    rm "$subject_dir/${filename}_otsu.nii.gz"

    echo "Completed processing $filename"

done

echo "Completed processing all files"