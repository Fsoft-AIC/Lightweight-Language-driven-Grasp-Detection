# Kiểm tra xem tất cả các subdirectories trong một directory có dạng YYYY-MM-DD_HH-MM-SS không tại depth = 4
# Nghĩa là {input_path}/*/*/*/YYYY-MM-DD_HH-MM-SS/
# Show ra các subdirectories không phải dạng YYYY-MM-DD_HH-MM-SS

import os
import glob
import argparse
import re
import shutil
import numpy as np
from tqdm import tqdm

# Create argument parser
parser = argparse.ArgumentParser(description="Debug check folder")
parser.add_argument("--input_path", type=str, help="Input path")
parser.add_argument("--output_path", type=str, help="Depth")
parser.add_argument("--number_copy", type=int, help="Number of copy")
args = parser.parse_args()


# Count all trajectory in a directory

def debug_count_and_copy(input_path, output_path, number_copy):
    count_all = 0
    # List all sub directories in input_path have ".*traj[0-9]" pattern
    paths = glob.glob(f'{input_path}/**/traj[0-9]*', recursive=True)
    print(len(paths))
    os.makedirs(output_path, exist_ok=True)
    # Choose randomly number_copy directories to copy
    paths_copy = np.random.choice(paths, number_copy, replace=False)
    
    # print(paths_copy)
    for i, path in tqdm(enumerate(paths_copy)):
        # Split input_path to get the last directory name
        # Example: input_path = data/demo_8_17/raw/<PATH_TO_TRAJECTORY>/traj0
        # Split input_path to get <PATH_TO_TRAJECTORY>/trajxxxxx
        path_target = path.replace(input_path, output_path)
        shutil.copytree(str(path), str(path_target), dirs_exist_ok=True)
        
# Test

debug_count_and_copy(args.input_path, args.output_path, args.number_copy)