#!/bin/bash

# Directory containing all T1 images
INPUT_DIR="/Users/donyapourn2/Desktop/projects/datasets/ADNI/t1_mpr_n4_corrected"
# Directory for outputs
OUTPUT_DIR="/Users/donyapourn2/Desktop/projects/datasets/ADNI/t1_mpr_segmentations_v3"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Process each .nii.gz file in the input directory
for input_image in "$INPUT_DIR"/*.nii.gz; do
    # Get the filename without path and extension
    filename=$(basename "$input_image" .nii.gz)
    echo "Processing $filename..."
    echo "Input image: $input_image"
    
    # Create output directory for this subject
    subject_dir="$OUTPUT_DIR/$filename"
    mkdir -p "$subject_dir"
    
    echo "Step 1: Creating better brain mask..."
    # First create Otsu with more classes for better tissue separation
    ThresholdImage 3 ${input_image} ${subject_dir}/${filename}_otsu.nii.gz Otsu 5

    # Create initial brain mask - more restrictive range
    ThresholdImage 3 ${subject_dir}/${filename}_otsu.nii.gz ${subject_dir}/${filename}_rough_mask.nii.gz 2 3 1 0

    # Clean up the mask with morphological operations
    # First dilate to connect components
    ImageMath 3 ${subject_dir}/${filename}_mask.nii.gz MD ${subject_dir}/${filename}_rough_mask.nii.gz 2
    # Then erode to remove small connections
    ImageMath 3 ${subject_dir}/${filename}_mask.nii.gz ME ${subject_dir}/${filename}_mask.nii.gz 3
    # Get largest component only
    ImageMath 3 ${subject_dir}/${filename}_mask.nii.gz GetLargestComponent ${subject_dir}/${filename}_mask.nii.gz
    # Final dilation to recover brain boundary
    ImageMath 3 ${subject_dir}/${filename}_mask.nii.gz MD ${subject_dir}/${filename}_mask.nii.gz 1

    echo "Step 2: Running tissue segmentation..."
    # Run Atropos with more iterations and smoothing
    Atropos \
        -d 3 \
        -a ${input_image} \
        -x ${subject_dir}/${filename}_mask.nii.gz \
        -i 'KMeans[3]' \
        -m '[0.3,1x1x1]' \
        -c '[10,0.00001]' \
        -k Gaussian \
        -o ${subject_dir}/${filename}_Segmentation.nii.gz

    # Check if segmentation was successful
    if [ ! -f "${subject_dir}/${filename}_Segmentation.nii.gz" ]; then
        echo "ERROR: Segmentation failed for ${filename}"
        continue
    fi

    echo "Step 3: Extracting and cleaning white matter (class 2)..."
    # Extract White Matter
    ThresholdImage 3 \
        ${subject_dir}/${filename}_Segmentation.nii.gz \
        ${subject_dir}/${filename}_white_matter_mask.nii.gz \
        2 2 1 0

    # Clean up the white matter mask
    ImageMath 3 ${subject_dir}/${filename}_white_matter_mask.nii.gz GetLargestComponent ${subject_dir}/${filename}_white_matter_mask.nii.gz
    
    # Clean up intermediate files
    rm -f ${subject_dir}/${filename}_Segmentation.nii.gz

    echo "Completed processing $filename"
done

echo "All processing complete!" 