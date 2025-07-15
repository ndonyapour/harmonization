#!/bin/bash

INPUT_DIR="/Users/donyapourn2/Desktop/projects/datasets/ADNI/t1_mpr_test"
OUTPUT_DIR="/Users/donyapourn2/Desktop/projects/datasets/ADNI/t1_mpr_segmentations_v5"
mkdir -p "$OUTPUT_DIR"

for input_image in "$INPUT_DIR"/*.nii.gz; do
    filename=$(basename "$input_image" .nii.gz)
    echo "Processing $filename..."
    
    subject_dir="$OUTPUT_DIR/$filename"
    mkdir -p "$subject_dir"

    # Step 1: Generate brain mask with balanced parameters
    ThresholdImage 3 "$input_image" "$subject_dir/${filename}_otsu.nii.gz" Otsu 4
    ThresholdImage 3 "$subject_dir/${filename}_otsu.nii.gz" "$subject_dir/${filename}_brain_mask.nii.gz" 2 4 1 0

    # Moderate mask cleaning
    ImageMath 3 "$subject_dir/${filename}_brain_mask.nii.gz" GetLargestComponent "$subject_dir/${filename}_brain_mask.nii.gz"
    ImageMath 3 "$subject_dir/${filename}_brain_mask.nii.gz" MD "$subject_dir/${filename}_brain_mask.nii.gz" 1
    ImageMath 3 "$subject_dir/${filename}_brain_mask.nii.gz" ME "$subject_dir/${filename}_brain_mask.nii.gz" 1
    ImageMath 3 "$subject_dir/${filename}_brain_mask.nii.gz" FillHoles "$subject_dir/${filename}_brain_mask.nii.gz"

    # Step 2: Run Atropos (3 classes) with optimized parameters
    Atropos \
        -d 3 \
        -a ["$input_image",0] \
        -x "$subject_dir/${filename}_brain_mask.nii.gz" \
        -i KMeans[3] \
        -m [0.25,1x1x1] \
        -c [10,0] \
        -k Gaussian \
        -o ["$subject_dir/${filename}_Segmentation.nii.gz","$subject_dir/${filename}_Posteriors%02d.nii.gz"]

    # Step 3: Extract White Matter
    ThresholdImage 3 "$subject_dir/${filename}_Segmentation.nii.gz" "$subject_dir/${filename}_white_matter_label.nii.gz" 2 2 1 0

    # Step 4: Combine with posterior map
    ThresholdImage 3 "$subject_dir/${filename}_Posteriors02.nii.gz" "$subject_dir/${filename}_WM_prob_mask.nii.gz" 0.55 1 1 0

    ImageMath 3 "$subject_dir/${filename}_white_matter_mask.nii.gz" m \
        "$subject_dir/${filename}_white_matter_label.nii.gz" \
        "$subject_dir/${filename}_WM_prob_mask.nii.gz"

    # Step 5: Enhanced cleanup for better connectivity
    # Get largest component
    ImageMath 3 "$subject_dir/${filename}_white_matter_clean.nii.gz" GetLargestComponent "$subject_dir/${filename}_white_matter_mask.nii.gz"
    
    # Initial hole filling
    ImageMath 3 "$subject_dir/${filename}_white_matter_clean.nii.gz" FillHoles "$subject_dir/${filename}_white_matter_clean.nii.gz"
    
    # Morphological operations to improve connectivity
    ImageMath 3 "$subject_dir/${filename}_white_matter_clean.nii.gz" MD "$subject_dir/${filename}_white_matter_clean.nii.gz" 1.5
    ImageMath 3 "$subject_dir/${filename}_white_matter_clean.nii.gz" ME "$subject_dir/${filename}_white_matter_clean.nii.gz" 1.25
    
    # Final hole filling
    ImageMath 3 "$subject_dir/${filename}_white_matter_clean.nii.gz" FillHoles "$subject_dir/${filename}_white_matter_clean.nii.gz"

    # Move to final output name
    mv "$subject_dir/${filename}_white_matter_clean.nii.gz" "$subject_dir/${filename}_white_matter_mask.nii.gz"

    echo "✅ Completed: $filename"
done

echo "🎉 All processing complete!"
