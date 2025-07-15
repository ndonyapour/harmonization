import os
import nibabel as nib
from pathlib import Path
import re
from intensity_normalization.normalize.nyul import NyulNormalize
from nyxusmed import Nyxus3DFeatureExtractor, Nyxus2DFeatureExtractor
from segmed import process_single_file
from totalsegmentator.python_api import totalsegmentator

def identify_t1_scans(base_dir: str) -> dict:
    """Identify T1 scans from PAR/REC files in the given directory.
    
    Args:
        base_dir: Base directory containing scan folders
        
    Returns:
        Dictionary mapping subject folders to their T1 scan paths
    """
    t1_scans = {}
    
    # Common T1 sequence identifiers
    t1_identifiers = [
        r'T1TFE',  # T1 Turbo Field Echo
        r'T1',
        r'T1W',
        r'T1-weighted',
        r'T1_weighted',
        r'T1W_',
        r'T1_',
        r'T1W-',
        r'T1-',
        r'T1W ',
        r'T1 ',
        r'T1TSE',  # T1 Turbo Spin Echo
        r'T1FFE',  # T1 Fast Field Echo
        r'T1GRE',  # T1 Gradient Echo
        r'T1MPRAGE'  # T1 MPRAGE
    ]
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(base_dir):
        # Look for PAR files
        par_files = [f for f in files if f.endswith('.PAR')]
        
        if par_files:
            print(f"\nFound PAR files in: {root}")
            for par_file in par_files:
                par_path = os.path.join(root, par_file)
                
                # Read PAR file header
                try:
                    # Load PAR/REC file using nibabel
                    par_img = nib.load(par_path)
                    header = par_img.header
                    
                    # Get scan parameters from header
                    scan_type = str(header.general_info.get("tech", ""))
                    protocol = str(header.general_info.get("protocol_name", ""))
                    scan_mode = str(header.general_info.get("scan_mode", ""))
                    
                    # Print header information
                    # print(f"\nHeader info for {par_file}:")
                    # print(f"Scan type: {scan_type}")
                    # print(f"Protocol: {protocol}")
                    # print(f"Scan mode: {scan_mode}")
                    
                    # Check if it's a T1 scan
                    is_t1 = False
                    
                    # Check scan type name
                    for identifier in t1_identifiers:
                        if re.search(identifier, scan_type, re.IGNORECASE) or \
                           re.search(identifier, protocol, re.IGNORECASE):
                            is_t1 = True
                            break
                    
                    if is_t1:
                        # print(f"Found T1 scan: {par_file}")
                        # print(f"Scan type: {scan_type}")
                        # print(f"Protocol: {protocol}")
                        # print(f"Scan mode: {scan_mode}")
                        print(f"\nHeader info for {par_file}:")
                        print(f"Scan type: {scan_type}")
                        # Store the path
                        subject_name = os.path.basename(root)
                        t1_scans[subject_name] = par_path
                        print(subject_name)
                    
                except Exception as e:
                        pass
    
    return t1_scans

def convert_to_nifti(images: dict, out_dir: str) -> None:
    """Convert PAR/REC files to NIfTI format.
    
    Args:
        images: List of paths to PAR/REC files
        out_dir: Directory to save NIfTI files
    """
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    
    for date, image in images.items():
        try:
            # Load PAR/REC file
            par_img = nib.load(image)
            
            # Get the data and affine
            data = par_img.get_fdata()
            affine = par_img.affine
            
            # Create new NIfTI image
            nifti_image = nib.Nifti1Image(data, affine)
            
            # Generate output filename
            output_path = os.path.join(out_dir, f"human_brain_{date}.nii.gz")
            
            # Save as NIfTI
            nib.save(nifti_image, output_path)
            print(f"Converted {image} to {output_path}")
            
        except Exception as e:
            print(f"Error converting {image}: {str(e)}")

def create_feature_vector(input_dir, out_dir):
    features = ["GLCM_CLUSHADE", "GLCM_ASM"]

    extractor = Nyxus2DFeatureExtractor(
        input_files=input_dir,
        out_dir=out_dir,
        features=features,
        per_slice=False,
        out_format="csv",
        metadata_file=None
        )
    extractor.run_parallel()

# def run_segmed(input_dir, output_dir):
#     file = "../datasets/Craig_scans/T1_nifti/20100304-human-stability_4_1.nii.gz"
#     config = {
#         "task": "total_mr",
#         "roi_subset": ["brain"],
#         "fast": True,
#     }
#     segmed = process_single_file( 
#         file_path=file,
#         specific_out_dir=output_dir,
#         config=config,
#     )

def test_3d_extraction(input_dir: str, seg_dir: str, out_dir: str) -> None:
    """Test 3D volumetric feature extraction.
    
    Args:
        input_files: Path pattern for input intensity images
        seg_files: Path pattern for segmentation masks
        out_dir: Output directory for results
    """
    nifti_files = [f for f in os.listdir(input_dir) 
                   if f.endswith('.nii.gz') or f.endswith('.nii')]
    print("\nTesting 3D Feature Extraction...")
    for file in nifti_files:
        input_file = os.path.join(input_dir, file)
        seg_file = os.path.join(seg_dir, file)
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

def run_total_segmentator(input_dir, output_dir):
    # List only NIfTI files (.nii or .nii.gz)
    nifti_files = [f for f in os.listdir(input_dir) 
                   if f.endswith('.nii.gz') or f.endswith('.nii')]
    
    for file in nifti_files:
        input_img = nib.load(os.path.join(input_dir, file))  
        output_path = os.path.join(output_dir, file)
        output_img = totalsegmentator(input_img, task="total_mr", roi_subset=["brain"], fast=True)
        nib.save(output_img, output_path)

if __name__ == "__main__":
    input_dir = "../datasets/Craig_scans/images"
    t1_nifti_output_dir = "../datasets/Craig_scans/T1_nifti"
    features_dir = "../datasets/Craig_scans/features"
    segmentation_dir = "../datasets/Craig_scans/segmentation"
    # Identify T1 scans
    print("Identifying T1 scans...")
    t1_scans = identify_t1_scans(input_dir)
    print(f"Found {t1_scans.values()} T1 scans")

    if not os.path.exists(t1_nifti_output_dir):
        os.makedirs(t1_nifti_output_dir)

    convert_to_nifti(t1_scans, t1_nifti_output_dir)    

    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    if not os.path.exists(segmentation_dir):
        os.makedirs(segmentation_dir)

    run_total_segmentator(t1_nifti_output_dir, segmentation_dir) # run total segmentator to get brain masks
    
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    test_3d_extraction(t1_nifti_output_dir, segmentation_dir, features_dir)
    
    # if not os.path.exists(t1_nifti_output_dir):
    #     os.makedirs(t1_nifti_output_dir)

    # convert_to_nifti(t1_scans.values(), t1_nifti_output_dir)

    # if not os.path.exists(nyxusmed_2d_output_dir):
    #     os.makedirs(nyxusmed_2d_output_dir)

    # create_feature_vector(t1_nifti_output_dir, nyxusmed_2d_output_dir)
