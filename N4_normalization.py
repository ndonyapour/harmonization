import os
import numpy as np
import nibabel as nib
from nyxusmed import Nyxus3DFeatureExtractor
from typing import List, Tuple, Dict
from pathlib import Path
from collections import defaultdict
import re
import random
from tqdm import tqdm
import ants


def is_n4_normalized(nifti_path: str) -> bool:
    """Check if a NIfTI file has been N4 normalized by examining its header.
    
    Args:
        nifti_path: Path to the NIfTI file
        
    Returns:
        bool: True if the file appears to be N4 normalized, False otherwise
    """
    try:
        # Load the NIfTI file
        img = nib.load(nifti_path)
        header = img.header
        
        # Check description field in header
        descrip = header.get('descrip', '').tobytes().decode('utf-8').lower()
        db_name = header.get('db_name', '').tobytes().decode('utf-8').lower()
        
        # Look for indicators of N4/B1 correction in header fields
        n4_indicators = ['n4', 'b1', 'bias_corrected', 'bias corrected', 'bias-corrected']
        
        # Check both description and database name fields
        for field in [descrip, db_name]:
            if any(indicator in field for indicator in n4_indicators):
                return True
                
        # Also check filename for B1 indicator
        if 'B1' in os.path.basename(nifti_path):
            return True
            
        return False
        
    except Exception as e:
        print(f"Warning: Error checking N4 status for {nifti_path}: {str(e)}")
        return False


def find_uncorrected_files(image_dir: str) -> List[str]:
    """Find NIfTI files that need N4 correction.
    
    Args:
        image_dir: Directory containing input images
        
    Returns:
        List of filenames that need N4 correction
    """
    all_files = sorted([f for f in os.listdir(image_dir) 
                       if f.endswith('.nii.gz') or f.endswith('.nii')])
    
    uncorrected_files = []
    
    for file_name in all_files:
        file_path = os.path.join(image_dir, file_name)
        if not is_n4_normalized(file_path):
            uncorrected_files.append(file_name)
            
    print(f"Found {len(uncorrected_files)} files needing N4 correction")
    print(f"Found {len(all_files) - len(uncorrected_files)} files already N4-corrected")
    
    return uncorrected_files


def apply_n4_correction(image_dir: str, out_dir: str, shrink_factor: int = 4) -> None:
    """Apply N4 bias field correction to a set of images.
    
    Args:
        image_dir: Directory containing input images
        out_dir: Output directory for N4 corrected images
        shrink_factor: Shrink factor to speed up N4 correction (default=4)
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    image_files = sorted([f for f in os.listdir(image_dir) 
                         if f.endswith('.nii.gz') or f.endswith('.nii')])
    
    for file_name in tqdm(image_files, total=len(image_files)):
        print(f"Processing {file_name}...")
        
        # Load image
        img_path = os.path.join(image_dir, file_name)
        img = ants.image_read(img_path)
        
        # Generate mask using Otsu's method
        mask = ants.get_mask(img)
        
        # Shrink image and mask to speed up correction
        input_shrinked = ants.resample_image(img, [s//shrink_factor for s in img.shape])
        mask_shrinked = ants.resample_image(mask, [s//shrink_factor for s in mask.shape])
        
        # Perform N4 bias field correction
        corrected = ants.n4_bias_field_correction(input_shrinked, 
                                                mask=mask_shrinked,
                                                shrink_factor=1)  # Already shrinked above
        
        # Get bias field at full resolution
        bias_field = ants.n4_bias_field_correction(img,
                                                 mask=mask,
                                                 return_bias_field=True)
        
        # Apply bias field correction at full resolution
        corrected_full = img / bias_field
        
        # Save corrected image
        output_path = os.path.join(out_dir, f"{Path(file_name).stem.split('.')[0]}_N4.nii.gz")
        ants.image_write(corrected_full, output_path)


if __name__ == "__main__":
    input_dir = "../datasets/ADNI/t1_mpr"
    n4_dir = "../datasets/ADNI/t1_mpr_n4_corrected"
    apply_n4_correction(input_dir, n4_dir)
    
    # # Continue with other processing...
    # seg_dir = "../datasets/ADNI/t1_mpr_segmentations"
    # normalized_dir = f"../datasets/ADNI/t1_mpr_nyul_normalized_2"

    # uncorrected_files = find_uncorrected_files(input_dir)
    # print(f"Found {len(uncorrected_files)} files needing N4 correction")
    # print(uncorrected_files)
    # train_files = select_training_images(input_dir) 
    # standard_histogram_path = os.path.join('../datasets/ADNI/standard_histogram', 'standard_histogram_27_2.npy')
    # test_dir = "../datasets/ADNI/t1_mpr_test"
    # apply_nyul_normalization(test_dir, normalized_dir, standard_histogram_path)