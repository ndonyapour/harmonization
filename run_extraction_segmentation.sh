#!/bin/bash

# Directory containing all T1 images
INPUT_DIR="/Users/donyapourn2/Desktop/projects/datasets/ADNI/t1_mpr_nyul_negative_removed_test" # t1_mpr_nyul_normalized_negative_removed"
# Directory for outputs
OUTPUT_DIR="/Users/donyapourn2/Desktop/projects/datasets/ADNI/t1_mpr_extraction_segmentations_negative_removed" #t1_mpr_negative_removed_extraction_segmentations"

# Template paths for brain extraction
# Using OASIS template which is commonly used with ANTs
T1_TEMPLATE="./templates/tpl-OASIS30ANTs_res-01_T1w.nii.gz"
BRAIN_TEMPLATE="./templates/tpl-OASIS30ANTs_res-01_desc-brain_mask.nii.gz"

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

    echo "Step 1: Running brain extraction..."
    # Run ANTs brain extraction
    antsBrainExtraction.sh \
        -d 3 \
        -a ${input_image} \
        -e ${T1_TEMPLATE} \
        -m ${BRAIN_TEMPLATE} \
        -o ${subject_dir}/${filename}_

    # echo "Step 2: Running tissue segmentation with Atropos..."
    # # Run Atropos segmentation on the brain-extracted image
    # # Now segmenting only brain tissue into CSF (1), GM (2), WM (3)
    Atropos \
        -d 3 \
        -a ${subject_dir}/${filename}_BrainExtractionBrain.nii.gz \
        -x ${subject_dir}/${filename}_BrainExtractionMask.nii.gz \
        -i 'Kmeans[3]' \
        -m '[0.1,1x1x1]' \
        -p 'Socrates[1]' \
        -c '[10,0.00001]' \
        -k Gaussian \
        -o ${subject_dir}/${filename}_Segmentation.nii.gz
    
    echo "Step 3: Extracting white matter mask..."
    # Extract White Matter (label 3) and create binary mask
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

    rm ${subject_dir}/${filename}_Segmentation.nii.gz
    rm ${subject_dir}/${filename}_BrainExtractionPrior0GenericAffine.mat
 
done

echo "All processing complete!" 