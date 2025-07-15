import os
import numpy as np
import nibabel as nib
from nyul import nyul_train_standard_scale, nyul_apply_standard_scale
from collections import defaultdict
import random
import re
import matplotlib.pyplot as plt

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
            subject_ids[subject_id].append(file_path)

    # Select 54 images for training
    
    train_files = []
    for subject_id, files in subject_ids.items():
        train_files.extend(random.sample(files, 1))
 
    return train_files

def learn_Nyul_normalization(train_scans: list[str], mask_scans: list[str]=None, standard_histogram_path: str=None):
    standard_scale, perc = nyul_train_standard_scale(train_scans, mask_scans, i_s_max=1000)
    np.save(standard_histogram_path, [standard_scale, perc])

#def apply_nyul_normalization(test_scans: list[str], standard_histogram_path: str):

if __name__ == "__main__":

    input_dir = "../datasets/ADNI/t1_mpr_n4_corrected"
    seg_dir = "../datasets/ADNI/t1_mpr_segmentations"
    normalized_dir = f"../datasets/ADNI/t1_mpr_nyul_normalized_2"
    train_files = select_training_images(input_dir) 
    standard_histogram_path = os.path.join('../datasets/ADNI/standard_histogram', 'standard_histogram_27_3.npy')
    mask_scans = select_training_images(seg_dir)
    learn_Nyul_normalization(train_scans=train_files, standard_histogram_path=standard_histogram_path)


    file = '../datasets/ADNI/t1_mpr_test/ADNI_123_S_0108_MR_MPR-R____N3__Scaled_2_Br_20081006160013571_S11415_I119310_N4.nii.gz'
    original_image = nib.load(file).get_fdata() 

    normalized_image = nyul_apply_standard_scale(original_image, standard_histogram_path)

    plt.hist(normalized_image.flatten(), bins=200)
    plt.ylim(0, 5e5)
    plt.show()

    # normalizer = apply_nyul_normalization(train_files, mask_scans, standard_histogram_path)
    # test_dir = "../datasets/ADNI/t1_mpr_test"
    # apply_nyul_normalization(test_dir, normalized_dir, standard_histogram_path)
   


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
    # print("Running 3D feature extraction on normalized images...")
    # run_3d_extraction(normalized_dir, seg_dir, normalized_features_dir)

    
    # input_dir = "../datasets/ADNI/t1_mpr"
    # normalized_dir = f"../datasets/ADNI/t1_mpr_kde_normalized"
    # seg_dir = "../datasets/ADNI/t1_mpr_segmentations"
    # normalized_features_dir = f"../datasets/ADNI/kde_normalized_features"
    # apply_kde_normalization(input_dir, normalized_dir, norm_value=100.0)
    # run_3d_extraction(normalized_dir, seg_dir, normalized_features_dir)
