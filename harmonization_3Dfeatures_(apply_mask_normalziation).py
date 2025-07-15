import os
import numpy as np
import nibabel as nib
from nyxusmed import Nyxus3DFeatureExtractor
from intensity_normalization.normalize.nyul import NyulNormalize
from typing import List, Tuple
from pathlib import Path


def extract_date_from_filename(filepath: str) -> str:
    """Extract date from filename.
    
    Args:
        filepath: Path to the file
        
    Returns:
        Extracted date string
    """
    # Implement date extraction logic based on your filename format
    return os.path.basename(filepath)


def run_nyul_normalization(image_dir: str, segmentation_dir: str, out_dir: str, 
                         filter_threshold: float = 100.0, min_percentile: float = 1.0, 
                         max_percentile: float = 99.99, output_min: float = 0.0, 
                         output_max: float = 6000.0) -> Tuple[List[np.ndarray], List[str]]:
    """Run Nyul intensity normalization on a set of images.
    
    Args:
        image_dir: Directory containing input images
        segmentation_dir: Directory containing segmentation masks
        out_dir: Output directory for normalized images
        filter_threshold: Threshold for filtering intensity values
        min_percentile: Minimum percentile for normalization
        max_percentile: Maximum percentile for normalization
        output_min: Minimum output value
        output_max: Maximum output value
        
    Returns:
        Tuple containing list of normalized images and corresponding scan dates
    """
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    
    # Get sorted list of image files
    image_files = sorted([f for f in os.listdir(image_dir) 
                         if f.endswith('.nii.gz') or f.endswith('.nii')])
    
    # Initialize lists for filtered images and scan paths
    masked_images = []
    scan_paths = []
    original_images = []  # Store original images for reconstruction
    original_masks = []   # Store original masks
    
    # Process each image
    for file_name in image_files:
        image_file = os.path.join(image_dir, file_name)

        seg_file = os.path.join(segmentation_dir, Path(file_name).stem.split(".")[0], 
                            f"{Path(file_name).stem.split('.')[0]}_white_matter_mask.nii.gz")

        print(f"Processing {file_name}...")
        
        # Load image and mask
        img_nib = nib.load(image_file)
        img = img_nib.get_fdata()
        mask_nib = nib.load(seg_file)
        mask = mask_nib.get_fdata()
        
        # Store original data
        original_images.append((img_nib, img))
        original_masks.append(mask)
        
        # Apply mask and filter
        data = img[mask > 0]
        data = data[data > filter_threshold].flatten()
        masked_images.append(data)
        
        # Extract date and append to lists
        scan_date = extract_date_from_filename(image_file)
        
        scan_paths.append(scan_date)

    # Initialize and fit normalizer
    normalizer = NyulNormalize(min_percentile=min_percentile, 
                              max_percentile=max_percentile,
                              output_min_value=output_min, 
                              output_max_value=output_max)
    normalizer.fit(masked_images)
    
    # Normalize images and save
    normalized_images = []
    for idx, (img_nib, img) in enumerate(original_images):
        # Get mask and create output array
        mask = original_masks[idx]
        normalized = np.zeros_like(img)
        
        # Get masked data and normalize
        masked_data = img[mask > 0]
        normalized_data = normalizer(masked_data)
        
        # Put normalized data back into original space
        normalized[mask > 0] = normalized_data
        
        # Create new nifti image with same header/affine
        normalized_nifti = nib.Nifti1Image(normalized, img_nib.affine, img_nib.header)
        
        # Save normalized image
        output_path = os.path.join(out_dir, f'{os.path.basename(scan_paths[idx])}')
        nib.save(normalized_nifti, output_path)
        
        normalized_images.append(normalized)
        
    return normalized_images, scan_paths


def run_3d_extraction(input_dir: str, seg_dir: str, out_dir: str) -> None:
    """Test 3D volumetric feature extraction.
    
    Args:
        input_files: Path pattern for input intensity images
        seg_files: Path pattern for segmentation masks
        out_dir: Output directory for results
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    nifti_files = [f for f in os.listdir(input_dir) 
                   if f.endswith('.nii.gz') or f.endswith('.nii')]
    print("\nTesting 3D Feature Extraction...")
    for file in nifti_files:
        input_file = os.path.join(input_dir, file)
        seg_file = os.path.join(seg_dir, Path(file).stem.split(".")[0], 
                            f"{Path(file).stem.split('.')[0]}_white_matter_mask.nii.gz")
        extractor = Nyxus3DFeatureExtractor(
            input_files=input_file,
            seg_files=seg_file,
            out_dir=out_dir,
            features=["ALL"],
            per_slice=False,
            out_format="csv",
            metadata_file=None
        )
        print(f"Running 3D feature extraction for {file}...")
        extractor.run_parallel()
        print(f"3D results saved to: {out_dir}")



if __name__ == "__main__":
    input_dir = "../datasets/Craig_scans/T1_nifti"
    seg_dir = "../datasets/Craig_scans/atropos_segmentations_no_n4"
    normalized_dir = "../datasets/Craig_scans/nyul_normalized_images"
    original_features_dir = "../datasets/Craig_scans/features_atropos_no_n4_wm_mask"
    normalized_features_dir = "../datasets/Craig_scans/normalized_features_atropos_no_n4_nyul"

    # print("Running Nyul normalization...")
    #run_nyul_normalization(input_dir, seg_dir, normalized_dir)
    # print("Running 3D feature extraction on original images...")
    run_3d_extraction(input_dir, seg_dir, original_features_dir)
    print("Running 3D feature extraction on normalized images...")
    #run_3d_extraction(normalized_dir, seg_dir, normalized_features_dir)

