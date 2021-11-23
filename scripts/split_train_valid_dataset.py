# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import os
import random
import shutil

from tqdm import tqdm


def main(args):
    train_inputs_image_dir = f"{args.inputs_dir}/train/inputs"
    train_target_image_dir = f"{args.inputs_dir}/train/target"
    valid_inputs_image_dir = f"{args.inputs_dir}/valid/inputs"
    valid_target_image_dir = f"{args.inputs_dir}/valid/target"

    if not os.path.exists(train_inputs_image_dir):
        os.makedirs(train_inputs_image_dir)
    if not os.path.exists(train_target_image_dir):
        os.makedirs(train_target_image_dir)
    if not os.path.exists(valid_inputs_image_dir):
        os.makedirs(valid_inputs_image_dir)
    if not os.path.exists(valid_target_image_dir):
        os.makedirs(valid_target_image_dir)

    train_files = os.listdir(train_inputs_image_dir)
    valid_files = random.sample(train_files, int(len(train_files) * args.valid_samples_ratio))

    process_bar = tqdm(valid_files, total=len(valid_files))

    for image_file_name in process_bar:
        train_inputs_image_path = f"{train_inputs_image_dir}/{image_file_name}"
        train_target_image_path = f"{train_target_image_dir}/{image_file_name}"
        valid_inputs_image_path = f"{valid_inputs_image_dir}/{image_file_name}"
        valid_target_image_path = f"{valid_target_image_dir}/{image_file_name}"
        shutil.copyfile(train_inputs_image_path, valid_inputs_image_path)
        shutil.copyfile(train_target_image_path, valid_target_image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split train and valid dataset scripts.")
    parser.add_argument("--inputs_dir", type=str, default="TB291/VDSR", help="Path to input image directory. (Default: ``TB291/VDSR``)")
    parser.add_argument("--valid_samples_ratio", type=float, default=0.1, help="What percentage of the data is extracted from the training set into the validation set.  (Default: 0.1)")
    args = parser.parse_args()

    main(args)
