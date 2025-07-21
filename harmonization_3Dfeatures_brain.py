from utils import *


if __name__ == "__main__":

    input_dir = "../datasets/Craig_scans/T1_nifti"
    seg_dir = "../datasets/Craig_scans/T1_nifti_segmentations"
    nyul_normalized_dir = f"../datasets/Craig_scans/nyul_normalized_images"
    kde_normalized_dir = f"../datasets/Craig_scans/kde_normalized_images"
    standard_histogram_path = "../datasets/Craig_scans/standard_histogram/standard_histogram.npy"
    normalized_features_dir = f"../datasets/Craig_scans/nyul_normalized_features"
    fullbrain_mask_dir = f"../datasets/Craig_scans/T1_nifti_fullbrain_mask"

    train_files = [f for f in os.listdir(input_dir) 
                         if f.endswith('.nii.gz') or f.endswith('.nii')]

    # normalizer = learn_Nyul_normalization(image_dir=input_dir,
    #                                     seg_dir=seg_dir,
    #                                     train_files=train_files,
    #                                     standard_histogram_path=standard_histogram_path,
    #                                     percentile_step=1.0)
    # apply_nyul_normalization(input_dir, seg_dir, nyul_normalized_dir, standard_histogram_path)
    #run_3d_extraction(nyul_normalized_dir, seg_dir, normalized_features_dir)

    apply_kde_normalization(input_dir, None, kde_normalized_dir)

    # print("Running Nyul normalization...")
    # run_nyul_normalization(input_dir, normalized_dir)
    # print("Running 3D feature extraction on original images...")
    # # run_3d_extraction(input_dir, seg_dir, original_features_dir)
    # print("Running 3D feature extraction on normalized images...")
    # run_3d_extraction(normalized_dir, seg_dir, normalized_features_dir)

    