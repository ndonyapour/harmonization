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
    
    echo "Step 1: Creating initial mask..."
    # First create a rough brain mask using Otsu with 4 classes
    ThresholdImage 3 ${input_image} ${subject_dir}/${filename}_otsu.nii.gz Otsu 4
    
    # Create a binary mask keeping only the two middle intensity classes (brain tissue)
    ThresholdImage 3 ${subject_dir}/${filename}_otsu.nii.gz ${subject_dir}/${filename}_rough_mask.nii.gz 2 3 1 0
    
    # Smooth the mask slightly to remove noise
    ImageMath 3 ${subject_dir}/${filename}_mask.nii.gz MD ${subject_dir}/${filename}_rough_mask.nii.gz 1
    ImageMath 3 ${subject_dir}/${filename}_mask.nii.gz ME ${subject_dir}/${filename}_mask.nii.gz 1

    echo "Step 2: Running tissue segmentation..."
    # Run Atropos with simpler initialization
    # In T1-weighted images:
    # - Class 1 typically corresponds to CSF (darkest)
    # - Class 2 typically corresponds to White Matter (brightest)
    # - Class 3 typically corresponds to Gray Matter (medium intensity)
    Atropos \
        -d 3 \
        -a ${input_image} \
        -x ${subject_dir}/${filename}_mask.nii.gz \
        -i 'KMeans[3]' \
        -m '[0.2,1x1x1]' \
        -c '[5,0]' \
        -k Gaussian \
        -o ${subject_dir}/${filename}_Segmentation.nii.gz

    # Check if segmentation was successful
    if [ ! -f "${subject_dir}/${filename}_Segmentation.nii.gz" ]; then
        echo "ERROR: Segmentation failed for ${filename}"
        continue
    fi

    echo "Step 3: Extracting white matter (class 2)..."
    # Extract White Matter (class 2) using both label and probability
    ThresholdImage 3 \
        ${subject_dir}/${filename}_Segmentation.nii.gz \
        ${subject_dir}/${filename}_white_matter_mask.nii.gz \
        2 2 1 0

    # Also create a probability mask from the posterior
    ThresholdImage 3 \
        ${subject_dir}/${filename}_Posteriors02.nii.gz \
        ${subject_dir}/${filename}_wm_prob.nii.gz \
        0.5 1.0 1 0

    # Combine both criteria for final white matter mask
    ImageMath 3 ${subject_dir}/${filename}_white_matter.nii.gz m ${subject_dir}/${filename}_wm_label.nii.gz ${subject_dir}/${filename}_wm_prob.nii.gz

    echo "Completed processing $filename"
done

echo "All processing complete!" 