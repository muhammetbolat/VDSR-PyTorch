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
import multiprocessing
import os

import cv2
import numpy as np
from tqdm import tqdm

import data_utils

from scipy import fftpack


def dct2(img):
    dim0 = fftpack.dct(img, axis=0, norm='ortho')
    dim1 = fftpack.dct(dim0, axis=1, norm='ortho')
    dim2 = fftpack.dct(dim1, axis=2, norm='ortho')

    return dim2


def idct2(img):
    dim0 = fftpack.idct(img, axis=0, norm='ortho')
    dim1 = fftpack.idct(dim0, axis=1, norm='ortho')
    dim2 = fftpack.idct(dim1, axis=2, norm='ortho')

    return dim2


def dropHighFrequencies(hr_image, rate) -> np.array:
    FIs = dct2(hr_image)
    Nx, Ny, _ = hr_image.shape

    Rx, Ry = np.int16([rate * Nx, rate * Ny])
    crop_FIs = FIs[0:Rx, 0:Ry]

    DCT_temp = np.zeros((Nx, Ny, 3))

    DCT_temp[0:Rx, 0:Ry] = crop_FIs

    I_r = idct2(DCT_temp).astype(int)

    Final_Img = np.zeros((Nx, Ny, 3))

    Final_Img[:, :, 0] = I_r[:, :, 0]
    Final_Img[:, :, 1] = I_r[:, :, 1]
    Final_Img[:, :, 2] = I_r[:, :, 2]

    lr_image = Final_Img.astype(int)

    return lr_image

def main(args) -> None:
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.exists(f"{args.output_dir}/hr"):
        os.makedirs(f"{args.output_dir}/hr")
    if not os.path.exists(f"{args.output_dir}/lr"):
        os.makedirs(f"{args.output_dir}/lr")

    # Get all image paths
    image_file_names = os.listdir(args.images_dir)

    # Splitting images with multiple threads
    progress_bar = tqdm(total=len(image_file_names), unit="image", desc="Prepare split image")
    workers_pool = multiprocessing.Pool(args.num_workers)
    for image_file_name in image_file_names:
        workers_pool.apply_async(worker, args=(image_file_name, args), callback=lambda arg: progress_bar.update(1))
    workers_pool.close()
    workers_pool.join()
    progress_bar.close()


def worker(image_file_name, args) -> None:
    image = cv2.imread(f"{args.images_dir}/{image_file_name}", cv2.IMREAD_UNCHANGED)

    image_height, image_width = image.shape[0:2]

    index = 1
    if image_height >= args.image_size and image_width >= args.image_size:
        for pos_y in range(0, image_height - args.image_size + 1, args.step):
            for pos_x in range(0, image_width - args.image_size + 1, args.step):
                # Crop hr image
                hr_image = image[pos_y: pos_y + args.image_size, pos_x:pos_x + args.image_size, ...]
                hr_image = np.ascontiguousarray(hr_image)
                # Resize lr image
                # lr_image = data_utils.imresize(hr_image, 1 / args.scale)
                # lr_image = data_utils.imresize(lr_image, args.scale)
                lr_image = dropHighFrequencies(hr_image, 1 / args.scale)
                # Save image
                cv2.imwrite(
                    f"{args.output_dir}/hr/{image_file_name.split('.')[-2]}_x{args.scale}_{index:04d}.{image_file_name.split('.')[-1]}",
                    hr_image)
                cv2.imwrite(
                    f"{args.output_dir}/lr/{image_file_name.split('.')[-2]}_x{args.scale}_{index:04d}.{image_file_name.split('.')[-1]}",
                    lr_image)

                index += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare database scripts.")
    parser.add_argument("--images_dir", type=str, help="Path to input image directory.", default="..\data\T91\original")
    parser.add_argument("--output_dir", type=str, help="Path to generator image directory.", default="..\data\T91\VDSR\\train")
    parser.add_argument("--image_size", type=int, help="Low-resolution image size from raw image.", default=42)
    parser.add_argument("--step", type=int, help="Crop image similar to sliding window.", default=42)
    parser.add_argument("--scale", type=int, help="Image down-scale factor.", default=2)
    parser.add_argument("--num_workers", type=int, help="How many threads to open at the same time.", default=1)

    args = parser.parse_args()
    main(args)
