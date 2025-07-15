#!/bin/bash

# Directory containing all T1 images
INPUT_DIR="/Users/donyapourn2/Desktop/projects/datasets/ADNI/t1_mpr_test"
# Directory for outputs
OUTPUT_DIR="/Users/donyapourn2/Desktop/projects/datasets/ADNI/t1_mpr_segmentations_v4"

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
    
    echo "Step 1: Creating brain mask..."
    # First normalize image intensity to 0-1 range
    ImageMath 3 ${subject_dir}/${filename}_norm.nii.gz RescaleImage ${input_image} 0 1

    # Create initial Otsu mask with 6 classes for finer separation
    ThresholdImage 3 ${subject_dir}/${filename}_norm.nii.gz ${subject_dir}/${filename}_otsu.nii.gz Otsu 6

    # More restrictive thresholding - only keep middle intensity classes (brain tissue)
    ThresholdImage 3 ${subject_dir}/${filename}_otsu.nii.gz ${subject_dir}/${filename}_rough_mask.nii.gz 3 4 1 0

    # Aggressive morphological operations
    # First get largest component to remove small outliers
    ImageMath 3 ${subject_dir}/${filename}_mask.nii.gz GetLargestComponent ${subject_dir}/${filename}_rough_mask.nii.gz
    
    # Fill holes in all directions
    ImageMath 3 ${subject_dir}/${filename}_mask.nii.gz FillHoles ${subject_dir}/${filename}_mask.nii.gz
    
    # Dilate to connect components
    ImageMath 3 ${subject_dir}/${filename}_mask.nii.gz MD ${subject_dir}/${filename}_mask.nii.gz 2
    
    # Get largest component again
    ImageMath 3 ${subject_dir}/${filename}_mask.nii.gz GetLargestComponent ${subject_dir}/${filename}_mask.nii.gz
    
    # Erode to remove boundary effects
    ImageMath 3 ${subject_dir}/${filename}_mask.nii.gz ME ${subject_dir}/${filename}_mask.nii.gz 3
    
    # Dilate slightly to recover brain boundary
    ImageMath 3 ${subject_dir}/${filename}_mask.nii.gz MD ${subject_dir}/${filename}_mask.nii.gz 1
    
    # Final cleanup
    ImageMath 3 ${subject_dir}/${filename}_mask.nii.gz FillHoles ${subject_dir}/${filename}_mask.nii.gz
    ImageMath 3 ${subject_dir}/${filename}_mask.nii.gz GetLargestComponent ${subject_dir}/${filename}_mask.nii.gz

    echo "Step 2: Running tissue segmentation..."
    # Run Atropos with more specific parameters
    Atropos \
        -d 3 \
        -a ${input_image} \
        -x ${subject_dir}/${filename}_mask.nii.gz \
        -i 'KMeans[3]' \
        -m '[0.2,1x1x1]' \
        -c '[15,0.000001]' \
        -k Gaussian \
        -r 1 \
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
    rm -f ${subject_dir}/${filename}_norm.nii.gz
    rm -f ${subject_dir}/${filename}_otsu.nii.gz
    rm -f ${subject_dir}/${filename}_rough_mask.nii.gz
    rm -f ${subject_dir}/${filename}_mask.nii.gz
    rm -f ${subject_dir}/${filename}_Segmentation.nii.gz

    echo "Completed processing $filename"
done

echo "All processing complete!" 