import os
import numpy as np
from utils import *

if __name__ == "__main__":
    
    input_dir = "../datasets/ADNI/t1_mpr_n4_corrected"
    seg_dir = "../datasets/ADNI/t1_mpr_segmentations"
    nyul_normalized_dir = f"../datasets/ADNI/t1_mpr_nyul_normalized"
    kde_normalized_dir = f"../datasets/ADNI/t1_mpr_kde_normalized"
    original_features_dir = f"../datasets/ADNI/t1_mpr_original_features"
    nyul_normalized_features_dir = f"../datasets/ADNI/t1_mpr_nyul_normalized_features"
    kde_normalized_features_dir = f"../datasets/ADNI/t1_mpr_kde_normalized_features"
    fullbrain_masks_dir = f"../datasets/ADNI/t1_mpr_fullbrain_mask"

    # train_files = select_training_images(input_dir) 
    # standard_histogram_path = os.path.join('../datasets/ADNI/standard_histogram', 'standard_histogram_27_masked.npy')


    # normalizer = learn_Nyul_normalization(image_dir=input_dir,
    #                                     seg_dir=seg_dir,
    #                                     train_files=train_files,
    #                                     standard_histogram_path=standard_histogram_path)
    # apply_nyul_normalization(input_dir, 
    #                                 seg_dir, 
    #                                 nyul_normalized_dir, 
    #                                 standard_histogram_path)

    # apply_kde_normalization(input_dir, fullbrain_masks_dir, kde_normalized_dir)

    # print("Running 3D feature extraction on original images...")
    # run_3d_extraction(input_dir, seg_dir, original_features_dir)
    print("Running 3D feature extraction on Nyul normalized images...")
    run_3d_extraction(nyul_normalized_dir, seg_dir, nyul_normalized_features_dir)
    # print("Running 3D feature extraction on KDE normalized images...")
    # run_3d_extraction(kde_normalized_dir, seg_dir, kde_normalized_features_dir)
