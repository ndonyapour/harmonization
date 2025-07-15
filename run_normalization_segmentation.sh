#!/bin/bash

# Directory containing all T1 images
INPUT_DIR="/Users/donyapourn2/Desktop/projects/datasets/Craig_scans/T1_nifti"
# Directory for outputs
OUTPUT_DIR="/Users/donyapourn2/Desktop/projects/datasets/Craig_scans/segmentations_ants"
# Template paths
T1_TEMPLATE="./templates/tpl-OASIS30ANTs_res-01_T1w.nii.gz"
MASK_TEMPLATE="./templates/tpl-OASIS30ANTs_res-01_desc-brain_mask.nii.gz"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Process each .nii.gz file in the input directory
for input_image in "$INPUT_DIR"/*.nii.gz; do
    # Get the filename without path and extension
    filename=$(basename "$input_image" .nii.gz)
    echo "Processing $filename..."
    
    # Create output directory for this subject
    subject_dir="$OUTPUT_DIR/$filename"
    mkdir -p "$subject_dir"
    
    # Run brain extraction
    antsBrainExtraction.sh \
        -d 3 \
        -a "$input_image" \
        -e "$T1_TEMPLATE" \
        -m "$MASK_TEMPLATE" \
        -o "$subject_dir/${filename}_"
    
    # Run segmentation
    antsAtroposN4.sh \
        -d 3 \
        -a "$input_image" \
        -x "$subject_dir/${filename}_BrainExtractionMask.nii.gz" \
        -o "$subject_dir/${filename}_seg_" \
        -c 3
        
    # Create white matter binary mask from segmentation posteriors
    ThresholdImage 3 "$subject_dir/${filename}_seg_SegmentationPosteriors3.nii.gz" \
        "$subject_dir/${filename}_white_matter_mask.nii.gz" \
        0.5 1.0 1 0
        
    echo "Completed processing $filename"
done

echo "All processing complete!" 