import os
import nibabel as nib
from pathlib import Path
import re

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
                        # Get the last part of the path as subject identifier
                        subject_name = os.path.basename(root)
                        t1_scans[subject_name] = par_path
                    
                except Exception as e:
                    pass
    
    return t1_scans

def main():
    input_dir = "../datasets/ADNI/images"
    # Identify T1 scans
    print("Identifying T1 scans...")
    t1_scans = identify_t1_scans(input_dir)
    
    # Print results
    print(f"\nFound T1 scans: {len(t1_scans)}")
    for subject, scan_path in t1_scans.items():
        print(f"Subject: {subject}")
        print(f"Scan path: {scan_path}")

if __name__ == "__main__":
    main() 