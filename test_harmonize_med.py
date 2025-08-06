import sys
import logging
from pathlib import Path

# Configure logging BEFORE importing HarmonizeMed
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Import after logging configuration
from harmonizemed import HarmonizeMed

input_dir = "../datasets/ADNI/ADNI_samples/kde_normalized"
seg_dir = "../datasets/ADNI/ADNI_samples/wm_masks"
output_dir = "../datasets/ADNI/ADNI_samples/harmonized"
standard_histogram_path = "../datasets/ADNI/ADNI_samples/standard_histogram.npy"

harmonizer = HarmonizeMed()
harmonizer.create_nyul_standard_scale(input_dir, seg_dir, standard_histogram_path)
filename = "ADNI_005_S_0814_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20081022101607053_S56661_I122649_N4"

# Keep filename as a string and construct paths properly
input_file = Path(input_dir) / f"{filename}.nii.gz"
mask_file = Path(seg_dir) / f"{filename}_white_matter_mask.nii.gz"
output_file_nyul = Path(output_dir) / f"{filename}_nyul.nii.gz"
output_file_kde = Path(output_dir) / f"{filename}_kde.nii.gz"

harmonizer.nyul_normalize(input_file, mask_file, output_file_nyul)

harmonizer.kde_normalize(input_file, mask_file, output_file_kde)