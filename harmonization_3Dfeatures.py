import os
import numpy as np
import nibabel as nib
from nyxusmed import Nyxus3DFeatureExtractor
from intensity_normalization.normalize.nyul import NyulNormalize
from typing import List, Tuple
from pathlib import Path
from intensity_normalization.normalize.kde import KDENormalize
from tqdm import tqdm

# https://www.medrxiv.org/content/10.1101/2021.02.24.21252322v2.full
def get_percentile(image_data, percentile: float = 98):
    # Flatten the image array to 1D
    flat_data = image_data.flatten()
    
    # Remove background/zero values
    nonzero_data = flat_data[flat_data > 0]
    
    # Calculate the percentile
    p = np.percentile(nonzero_data, percentile)
    return p

def extract_date_from_filename(filepath: str) -> str:
    """Extract date from filename.
    
    Args:
        filepath: Path to the file
        
    Returns:
        Extracted date string
    """
    # Implement date extraction logic based on your filename format
    return os.path.basename(filepath)


def run_nyul_normalization(image_dir: str, out_dir: str, min_percentile: float = 1.0, 
                         max_percentile: float = 99.99, output_min: float = 0.0, 
                         batch_size: int = 30) -> None:
    """Run Nyul intensity normalization on a set of images.
    
    Args:
        image_dir: Directory containing input images
        out_dir: Output directory for normalized images
        min_percentile: Minimum percentile for normalization
        max_percentile: Maximum percentile for normalization
        output_min: Minimum output value
        batch_size: Number of images to process at once for learning landmarks
    """
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    
    # Get sorted list of image files
    image_files = sorted([f for f in os.listdir(image_dir) 
                         if f.endswith('.nii.gz') or f.endswith('.nii')])
    
    print(f"Found {len(image_files)} images")
    
    # Calculate output_max using a subset of images
    print("Calculating output max...")
    sample_size = min(batch_size, len(image_files))
    sample_files = np.random.choice(image_files, sample_size, replace=False)
    
    data_for_percentile = []
    for file_name in sample_files:
        image_file = os.path.join(image_dir, file_name)
        img = nib.load(image_file).get_fdata()
        data_for_percentile.extend(img[img > 0].flatten())  # Only include non-zero values
        del img  # Explicitly delete to free memory
    
    output_max = get_percentile(np.array(data_for_percentile), percentile=98)
    del data_for_percentile  # Free memory
    
    # Initialize normalizer
    normalizer = NyulNormalize(min_percentile=min_percentile, 
                              max_percentile=max_percentile,
                              output_min_value=output_min, 
                              output_max_value=output_max)
    
    # Process images in batches for fitting
    print("Fitting normalizer...")
    num_batches = (len(image_files) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(image_files))
        batch_files = image_files[start_idx:end_idx]
        
        batch_data = []
        for file_name in batch_files:
            image_file = os.path.join(image_dir, file_name)
            img = nib.load(image_file).get_fdata()
            batch_data.append(img)
        
        # Fit on this batch
        if batch_idx == 0:
            normalizer.fit(batch_data)
        else:
            normalizer.fit(batch_data, update_landmarks=True)
        
        # Clear batch data
        del batch_data
    
    # Process and save each image individually
    print("Normalizing images...")
    for file_name in image_files:
        try:
            image_file = os.path.join(image_dir, file_name)
            print(f"Processing {file_name}...")
            
            # Load and normalize single image
            img_nib = nib.load(image_file)
            img = img_nib.get_fdata()
            normalized = normalizer(img)
            
            # Save normalized image
            normalized_nifti = nib.Nifti1Image(normalized, img_nib.affine, img_nib.header)
            output_path = os.path.join(out_dir, file_name)
            nib.save(normalized_nifti, output_path)
            
            # Clear memory
            del img, normalized, normalized_nifti
            
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            continue
    
    print("Normalization complete!")


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


def apply_kde_normalization(image_dir: str, out_dir: str, norm_value: float = 100.0) -> None:    
    """Run KDE normalization on a set of images.
    
    Args:
        image_dir: Directory containing input images
        out_dir: Output directory for normalized images
    """
    if not os.path.exists(out_dir): 
        os.makedirs(out_dir)

    image_files = sorted([f for f in os.listdir(image_dir) 
                         if f.endswith('.nii.gz') or f.endswith('.nii')])
    
    kde_normalizer = KDENormalize(norm_value=norm_value)

    for file_name in tqdm(image_files, total=len(image_files)):
        print(f"Processing {file_name}...")
        img_nib = nib.load(os.path.join(image_dir, file_name))
        img = img_nib.get_fdata()
        normalized = kde_normalizer(img)
        normalized_nifti = nib.Nifti1Image(normalized, img_nib.affine, img_nib.header)
        output_path = os.path.join(out_dir, f"{Path(file_name).stem.split('.')[0]}.nii.gz")
        nib.save(normalized_nifti, output_path)

if __name__ == "__main__":
    # input_dir = "../datasets/Craig_scans/T1_nifti"
    # seg_dir = "../datasets/Craig_scans/atropos_segmentations_no_n4"
    # normalized_dir = f"../datasets/Craig_scans/nyul_normalized_images"
    # original_features_dir = "../datasets/Craig_scans/original_features"
    # normalized_features_dir = f"../datasets/Craig_scans/nyul_normalized_features"

    # input_dir = "../datasets/ADNI/t1_mpr"
    # normalized_dir = f"../datasets/ADNI/t1_mpr_normalized"
    # seg_dir = "../datasets/ADNI/t1_mpr_segmentations"

    # print("Running Nyul normalization...")
    # run_nyul_normalization(input_dir, normalized_dir)
    # print("Running 3D feature extraction on original images...")
    # # run_3d_extraction(input_dir, seg_dir, original_features_dir)
    # print("Running 3D feature extraction on normalized images...")
    # run_3d_extraction(normalized_dir, seg_dir, normalized_features_dir)

    input_dir = "../datasets/Craig_scans/T1_nifti"
    seg_dir = "../datasets/Craig_scans/atropos_segmentations_no_n4"
    normalized_dir = f"../datasets/Craig_scans/kde_normalized_images"
    normalized_features_dir = f"../datasets/Craig_scans/kde_normalized_features"

    apply_kde_normalization(input_dir, normalized_dir, norm_value=100.0)
    run_3d_extraction(normalized_dir, seg_dir, normalized_features_dir)