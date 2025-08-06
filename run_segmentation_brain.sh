#!/bin/bash

INPUT_DIR="/Users/donyapourn2/Desktop/projects/datasets/Craig_scans/T1_nifti_test"
OUTPUT_DIR="/Users/donyapourn2/Desktop/projects/datasets/ADNI/t1_nifti_segmentations"
mkdir -p "$OUTPUT_DIR"

for input_image in "$INPUT_DIR"/*.nii.gz; do
    filename=$(basename "$input_image" .nii.gz)
    echo "Processing $filename..."
    
    subject_dir="$OUTPUT_DIR/$filename"
    mkdir -p "$subject_dir"

    # Step 1: Generate brain mask with balanced parameters
    echo "Step 1: Generating brain mask..."
    ThresholdImage 3 "$input_image" "$subject_dir/${filename}_otsu.nii.gz" Otsu 4
    ThresholdImage 3 "$subject_dir/${filename}_otsu.nii.gz" "$subject_dir/${filename}_mask.nii.gz" 2 4 1 0

    echo "Step 2: Running Atropos segmentation..."
    Atropos \
        -d 3 \
        -a [${input_image},0] \
        -x ${subject_dir}/${filename}_mask.nii.gz \
        -i 'Kmeans[3]' \
        -m '[0.2,1x1x1]' \
        -c '[5,0]' \
        -k Gaussian \
        -o ${subject_dir}/${filename}_Segmentation.nii.gz

    echo "Step 3: Extracting white matter mask..."
    # Extract White Matter (label 2) and create binary mask
    ThresholdImage 3 \
        "$subject_dir/${filename}_Segmentation.nii.gz" \
        "$subject_dir/${filename}_white_matter_mask.nii.gz" \
        2 2 1 0

    # # Fill holes in 3D
    ImageMath 3 \
        "$subject_dir/${filename}_white_matter_mask.nii.gz" \
        FillHoles \
        "$subject_dir/${filename}_white_matter_mask.nii.gz"

    # Remove small islands (more aggressive)
    ImageMath 3 \
        "$subject_dir/${filename}_white_matter_mask.nii.gz" \
        GetLargestComponent \
        "$subject_dir/${filename}_white_matter_mask.nii.gz"

    # Smooth boundaries while preserving topology
    ImageMath 3 \
        "$subject_dir/${filename}_white_matter_mask.nii.gz" \
        MD \
        "$subject_dir/${filename}_white_matter_mask.nii.gz" \
        1
    
    ImageMath 3 \
        "$subject_dir/${filename}_white_matter_mask.nii.gz" \
        ME \
        "$subject_dir/${filename}_white_matter_mask.nii.gz" \
        1

done