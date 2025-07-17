import os
import numpy as np
import nibabel as nib
from nyxusmed import Nyxus3DFeatureExtractor
from intensity_normalization import NyulNormalizer, WhiteStripeNormalizer, KDENormalizer
from intensity_normalization.adapters.images import create_image 
from intensity_normalization.adapters.io import  save_image
from typing import List, Tuple
from pathlib import Path
from collections import defaultdict
import re
import random
from tqdm import tqdm
import ants

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


 
def learn_Nyul_normalization(image_dir: str, min_percentile: float = 1.0, 
                         max_percentile: float = 99.99, output_min: float = 0.0, train_files: list[str] = [], 
                         standard_histogram_path: str = None) -> Tuple[List[np.ndarray], List[str]]:
    
    train_images = []
    
    
    # Process each image
    for file_name in train_files:
        image_file = os.path.join(image_dir, file_name)

        print(f"Processing {file_name}...")
        
        # Load image 
        img_nib = nib.load(image_file)
        img = img_nib.get_fdata()
        
        train_images.append(img)
        


    # Do the learning using the training images
    # For the now, we use the 98th percentile as the output max. We get back to this later.
    # https://www.medrxiv.org/content/10.1101/2021.02.24.21252322v2.full

    
    data_flattened = []
    for img in train_images:
        data_flattened.extend(img.flatten())

    
    # print("Calculating output max...")
    # output_max = get_percentile(np.array(data_flattened), percentile=98)

    # print(f"Output max: {output_max}")
    # Initialize and fit normalizer
    normalizer = NyulNormalizer(min_percentile=min_percentile, 
                              max_percentile=max_percentile,
                              output_min_value=output_min, 
                              output_max_value=1200)
    #normalizer = NyulNormalizer()
    images = [create_image(os.path.join(image_dir, file_name)) for file_name in train_files]
    print(f"Training on {len(train_images)} images")
    print("Fitting Nyul normalizer...")
    normalizer.fit_population(images)
    print("Fitting Nyul normalizer done")
    normalizer.save_standard_histogram(standard_histogram_path)
    return normalizer
    
    
    
def apply_nyul_normalization(image_dir: str, out_dir: str, normalizer_path: str) -> None:
    """Run Nyul intensity normalization on a set of images.
    
    Args:
        image_dir: Directory containing input images
        out_dir: Output directory for normalized images
        
    Returns:
        Tuple containing list of normalized images and corresponding scan dates
    """
   
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Get sorted list of image files
    image_files = sorted([f for f in os.listdir(image_dir) 
                         if f.endswith('.nii.gz') or f.endswith('.nii')])

    normalizer = NyulNormalizer()
    normalizer.load_standard_histogram(normalizer_path)
    # Normalize all images and save
    # add tqdm
    for file_name in tqdm(image_files, total=len(image_files)):
        print(f"Processing {file_name}...")
        # Get mask and create output array
        img_nib = nib.load(os.path.join(image_dir, file_name))
        img = img_nib.get_fdata()
        
        normalized = normalizer.transform(create_image(os.path.join(image_dir, file_name)))
        
        # Create new nifti image with same header/affine
        normalized_nifti = nib.Nifti1Image(normalized, img_nib.affine, img_nib.header)
        
        # Save normalized image
        output_path = os.path.join(out_dir, f"{Path(file_name).stem.split('.')[0]}.nii.gz")
        save_image(normalized, output_path)
        
        


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

def get_subject_id(nifti_img, filename: str = '') -> str:
    header = nifti_img.header
    descrip = header.get('descrip', '').tobytes().decode('utf-8').lower()
    db_name = header.get('db_name', '').tobytes().decode('utf-8').lower()
    
    # Try to find subject ID
    subject_patterns = [
        r'(sub-\w+)',           # BIDS format
        r'(adni_\d+)',          # ADNI format
        r'(s\d{4})',            # ADNI S#### format
        r'(subject[-_]\w+)'     # General format
    ]
    
    # Look in both header fields and filename
    text_to_search = f"{descrip} {db_name} {filename}".lower()
    
    for pattern in subject_patterns:
        match = re.search(pattern, text_to_search)
        if match:
            subject_id = match.group(1)
            break
    return subject_id

def select_training_images(input_dir: str, filename: str = '') -> list[str]:
    """
    Get subject ID and scan date from a NIfTI file.
    
    Args:
        nifti_img: A NiBabel image object
        filename: Optional filename to extract information from
        
    Returns:
        tuple: (subject_id, scan_date) where either could be 'unknown' if not found
    """
    # Set random seed for reproducibility
    random.seed(42)
    
    subject_id = 'unknown'
    subject_ids = defaultdict(list)
    for file in os.listdir(input_dir):
        if file.endswith('.nii.gz') or file.endswith('.nii'):
            file_path = os.path.join(input_dir, file)
            img = nib.load(file_path)
            subject_id = get_subject_id(img, file)
            subject_ids[subject_id].append(file)

    # Select 54 images for training
    
    train_files = []
    for subject_id, files in subject_ids.items():
        train_files.extend(random.sample(files, 1))
 
    return train_files

def learn_Nyul_normalization_masked(image_dir: str, seg_dir: str, min_percentile: float = 1.0, 
                              max_percentile: float = 99.99, output_min: float = 0.0, train_files: list[str] = [], 
                              standard_histogram_path: str = None) -> NyulNormalizer:
    """Learn Nyul normalization parameters using white matter masks directly in fit.
    
    Args:
        image_dir: Directory containing input images
        seg_dir: Directory containing segmentation masks
        out_dir: Output directory
        min_percentile: Minimum percentile for normalization
        max_percentile: Maximum percentile for normalization
        output_min: Minimum output value
        train_files: List of training files
        standard_histogram_path: Path to save standard histogram
    """
    train_images = []
    train_masks = []
    
    # Process each image and its mask
    for file_name in train_files:
        image_file = os.path.join(image_dir, file_name)
        mask_file = os.path.join(seg_dir, Path(file_name).stem.split(".")[0], 
                              f"{Path(file_name).stem.split('.')[0]}_mask.nii.gz")

        print(f"Processing {file_name}...")
        
        # Load image
        # img_nib = nib.load(image_file)
        # img = img_nib.get_fdata()
        
        # Load white matter mask
        # if os.path.exists(mask_file):
        #     mask_nib = nib.load(mask_file)
        #     mask = mask_nib.get_fdata() > 0
        #     print(f"Found mask for {file_name}")
        # else:
        #     print(f"Warning: No mask found for {file_name}, using whole brain")
        #     mask = np.ones_like(img, dtype=bool)
        
        train_images.append(create_image(image_file))
        train_masks.append(create_image(mask_file))

    print(f"Training on {len(train_images)} images with masks")
    
    # Initialize and fit normalizer with images and masks
    #
    normalizer = NyulNormalizer(min_percentile=min_percentile, 
                              max_percentile=max_percentile,
                              output_min_value=output_min, 
                              output_max_value=1200)
    
    # Directly use masks in fit
    normalizer.fit_population(train_images, masks=train_masks)
    
    if standard_histogram_path:
        normalizer.save_standard_histogram(standard_histogram_path)
    
    return normalizer

def apply_nyul_normalization_masked(image_dir: str, seg_dir: str, out_dir: str, normalizer_path: str) -> None:
    """Apply Nyul normalization using white matter masks.
    
    Args:
        image_dir: Directory containing input images
        seg_dir: Directory containing segmentation masks
        out_dir: Output directory for normalized images
        normalizer_path: Path to saved normalizer
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    image_files = sorted([f for f in os.listdir(image_dir) 
                         if f.endswith('.nii.gz') or f.endswith('.nii')])

    normalizer = NyulNormalizer()
    normalizer.load_standard_histogram(normalizer_path)
    
    for file_name in tqdm(image_files, total=len(image_files)):
        print(f"Processing {file_name}...")
        
        # Load image and mask
        img_nib = nib.load(os.path.join(image_dir, file_name))
        img = img_nib.get_fdata()
        
        mask_file = os.path.join(seg_dir, Path(file_name).stem.split(".")[0], 
                              f"{Path(file_name).stem.split('.')[0]}_mask.nii.gz")
        
        if os.path.exists(mask_file):
            mask_nib = nib.load(mask_file)
            mask = mask_nib.get_fdata() > 0
        else:
            print(f"Warning: No mask found for {file_name}, using whole brain")
            mask = np.ones_like(img, dtype=bool)
        
        # Apply normalization with mask
        normalized = normalizer.transform(create_image(os.path.join(image_dir, file_name)), 
                                          mask=create_image(mask_file))
        
        # Save normalized image
        output_path = os.path.join(out_dir, f"{Path(file_name).stem.split('.')[0]}.nii.gz")
        save_image(normalized, output_path)

# def apply_whitestripe_normalization(image_dir: str, seg_dir: str, out_dir: str, scale: float = 100.0) -> None:
#     """Apply WhiteStripe normalization using the WhiteStripeNormalize class.
    
#     Args:
#         image_dir: Directory containing input images
#         seg_dir: Directory containing segmentation masks
#         out_dir: Output directory for normalized images
#         scale: Scale factor for normalization (default=100.0)
#     """
#     if not os.path.exists(out_dir):
#         os.makedirs(out_dir)
        
#     image_files = sorted([f for f in os.listdir(image_dir) 
#                          if f.endswith('.nii.gz') or f.endswith('.nii')])

#     # Initialize WhiteStripe normalizer
#     whitestripe = WhiteStripeNormalize(norm_value=scale)
    
#     for file_name in tqdm(image_files, total=len(image_files)):
#         print(f"Processing {file_name}...")
        
#         # Load image and mask
#         img_nib = nib.load(os.path.join(image_dir, file_name))
#         img = img_nib.get_fdata()
        
#         mask_file = os.path.join(seg_dir, Path(file_name).stem.split(".")[0], 
#                               f"{Path(file_name).stem.split('.')[0]}_white_matter_mask.nii.gz")
        
#         if os.path.exists(mask_file):
#             mask_nib = nib.load(mask_file)
#             mask = mask_nib.get_fdata() > 0
#         else:
#             print(f"Warning: No mask found for {file_name}, using whole brain")
#             mask = np.ones_like(img, dtype=bool)
        
#         # Apply WhiteStripe normalization
#         normalized = whitestripe(img, mask=mask)
        
#         # Create new nifti image
#         normalized_nifti = nib.Nifti1Image(normalized, img_nib.affine, img_nib.header)
        
#         # Save normalized image
#         output_path = os.path.join(out_dir, f"{Path(file_name).stem.split('.')[0]}.nii.gz")
#         nib.save(normalized_nifti, output_path)

def apply_kde_normalization(image_dir: str, seg_dir: str, out_dir: str) -> None:    
    """Run KDE normalization on a set of images.
    
    Args:
        image_dir: Directory containing input images
        out_dir: Output directory for normalized images
    """
    if not os.path.exists(out_dir): 
        os.makedirs(out_dir)

    image_files = sorted([f for f in os.listdir(image_dir) 
                         if f.endswith('.nii.gz') or f.endswith('.nii')])
    
    kde_normalizer = KDENormalizer()
    kde_normalizer.norm_value = 100.0

    for file_name in tqdm(image_files, total=len(image_files)):
        print(f"Processing {file_name}...")
        img_file = os.path.join(image_dir, file_name)
        mask_file = os.path.join(seg_dir, Path(file_name).stem.split(".")[0], 
                              f"{Path(file_name).stem.split('.')[0]}_mask.nii.gz")
   
        normalized = kde_normalizer.fit_transform(create_image(img_file), mask=create_image(mask_file))
        output_path = os.path.join(out_dir, f"{Path(file_name).stem.split('.')[0]}.nii.gz")
        save_image(normalized, output_path)

def remove_negative_values(input_dir: str, output_dir: str) -> None:
    """
    Remove negative values from the images in the input directory and save the results in the output directory.
    """
    min_value = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("Starting to find minimum value...")
    for file in tqdm(os.listdir(input_dir), total=len(os.listdir(input_dir))):
        if file.endswith('.nii.gz') or file.endswith('.nii'):
            file_path = os.path.join(input_dir, file)
            img = create_image(file_path)
            img_data = img.get_data()
            min_value.append(img_data.min())

    shift = - min(min_value)
    print(f"Shift: {shift}")
    print("Starting to remove negative values...")
    for file in tqdm(os.listdir(input_dir), total=len(os.listdir(input_dir))):
        if file.endswith('.nii.gz') or file.endswith('.nii'):
            file_path = os.path.join(input_dir, file)
            img = create_image(file_path)
            img_data = img.get_data()
            img_data = img_data + shift
            img = img.with_data(img_data)
            save_image(img, os.path.join(output_dir, file))

    print(f"Removed negative values for all images in {input_dir} and saved to {output_dir}")
       

if __name__ == "__main__":
    
    input_dir = "../datasets/ADNI/t1_mpr_n4_corrected"
    seg_dir = "../datasets/ADNI/t1_mpr_segmentations"
    nyul_normalized_dir = f"../datasets/ADNI/t1_mpr_nyul_normalized"
    kde_normalized_dir = f"../datasets/ADNI/t1_mpr_kde_normalized"
    original_features_dir = f"../datasets/ADNI/t1_mpr_original_features"
    nyul_normalized_features_dir = f"../datasets/ADNI/t1_mpr_nyul_normalized_features"
    kde_normalized_features_dir = f"../datasets/ADNI/t1_mpr_kde_normalized_features"

    train_files = select_training_images(input_dir) 
    standard_histogram_path = os.path.join('../datasets/ADNI/standard_histogram', 'standard_histogram_27_masked.npy')
    # normalizer = learn_Nyul_normalization(image_dir=input_dir,
    #                                     train_files=train_files,
    #                                     standard_histogram_path=standard_histogram_path)
    # test_dir = "../datasets/ADNI/t1_mpr_test"
    # # apply_nyul_normalization(test_dir, normalized_dir, standard_histogram_path)

    # normalizer = learn_Nyul_normalization_masked(image_dir=input_dir,
    #                                     seg_dir=seg_dir,
    #                                     train_files=train_files,
    #                                     standard_histogram_path=standard_histogram_path)
    # apply_nyul_normalization_masked(input_dir, 
    #                                 seg_dir, 
    #                                 normalized_dir, 
    #                                 standard_histogram_path)

   


    #normalized_dir = f"../datasets/ADNI/t1_mpr_normalized"
    # seg_dir = "../datasets/ADNI/t1_mpr_segmentations"
    # original_features_dir = "../datasets/ADNI/original_features"
    # normalized_features_dir = f"../datasets/ADNI/nyul_normalized_features"
    # standard_histogram_path = os.path.join("../datasets/ADNI", "standard_histogram.npy")

    # train_files = select_training_images(input_dir)
    # normalizer = learn_Nyul_normalization(input_dir, normalized_dir, train_files=train_files, standard_histogram_path=standard_histogram_path)
    # apply_nyul_normalization(input_dir, normalized_dir, standard_histogram_path)


  
    # print("Running 3D feature extraction on original images...")
    # run_3d_extraction(input_dir, seg_dir, original_features_dir)
    # print("Running 3D feature extraction on Nyulnormalized images...")
    # run_3d_extraction(nyul_normalized_dir, seg_dir, nyul_normalized_features_dir)
    # print("Running 3D feature extraction on KDE normalized images...")
    # run_3d_extraction(kde_normalized_dir, seg_dir, kde_normalized_features_dir)

    # apply_kde_normalization(input_dir, seg_dir, normalized_dir)
    # #run_3d_extraction(normalized_dir, seg_dir, normalized_features_dir)

    nyul_normalized_dir = f"../datasets/ADNI/t1_mpr_nyul_normalized"
    nyul_normalized_negative_removed = f"../datasets/ADNI/t1_mpr_nyul_normalized_negative_removed"
    remove_negative_values(nyul_normalized_dir, nyul_normalized_negative_removed)