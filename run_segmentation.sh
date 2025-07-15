#!/bin/bash

# Directory containing all T1 images
# INPUT_DIR="/Users/donyapourn2/Desktop/projects/datasets/ADNI/t1_mpr_n4_corrected"
# # Directory for outputs
# OUTPUT_DIR="/Users/donyapourn2/Desktop/projects/datasets/ADNI/t1_mpr_segmentations"
INPUT_DIR="/Users/donyapourn2/Desktop/projects/datasets/ADNI/t1_mpr_test"
# Directory for outputs
OUTPUT_DIR="/Users/donyapourn2/Desktop/projects/datasets/ADNI/t1_mpr_segmentations_v5"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Process each .nii.gz file in the input directory
for input_image in "$INPUT_DIR"/*.nii.gz; do
    # Get the filename without path and extension
    filename=$(basename "$input_image" .nii.gz)
    echo "Processing $filename..."
    echo "Input image: $input_image"
    
    # Verify input image exists
    if [ ! -f "$input_image" ]; then
        echo "ERROR: Input image not found: $input_image"
        continue
    fi
    
    # Create output directory for this subject
    subject_dir="$OUTPUT_DIR/$filename"
    mkdir -p "$subject_dir"
    echo "Output directory: $subject_dir"

    # Generate a simple brain mask using Otsu thresholding
    echo "Generating brain mask..."
    ThresholdImage 3 ${input_image} ${subject_dir}/${filename}_mask.nii.gz Otsu 4
    ThresholdImage 3 ${subject_dir}/${filename}_mask.nii.gz ${subject_dir}/${filename}_mask.nii.gz 2 3 1 0

    ImageMath 3 ${subject_dir}/${filename}_mask.nii.gz MD ${subject_dir}/${filename}_mask.nii.gz 1
    ImageMath 3 ${subject_dir}/${filename}_mask.nii.gz ME ${subject_dir}/${filename}_mask.nii.gz 1
    
    # # Run Atropos segmentation with the mask
    # echo "Running Atropos segmentation..."
    Atropos \
        -d 3 \
        -a [${input_image},0] \
        -x ${subject_dir}/${filename}_mask.nii.gz \
        -i 'Kmeans[3]' \
        -m '[0.2,1x1x1]' \
        -c '[5,0]' \
        -k Gaussian \
        -o ${subject_dir}/${filename}_Segmentation.nii.gz
    
    # Extract White Matter from segmentation (class 3)
    ThresholdImage 3 \
        "$subject_dir/${filename}_Segmentation.nii.gz" \
        "$subject_dir/${filename}_white_matter_mask.nii.gz" \
        2 2 1 0
        

    # # Create probabilistic WM mask from posterior (threshold > 0.5)
    # ThresholdImage 3 \
    #     "$subject_dir1/${filename}_SegmentationPosteriors03.nii.gz" \
    #     "$subject_dir1/${filename}_WM_prob.nii.gz" \
    #     0.5 1 1 0
        
    echo "Completed processing $filename"
done

echo "All processing complete!" 