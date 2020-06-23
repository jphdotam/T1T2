import os
import cv2
import pydicom
import multiprocessing
import numpy as np
import skimage.io
import skimage.transform
import albumentations as A
from glob import glob
from tqdm import tqdm
from scipy import interpolate

import matplotlib.pyplot as plt

from utils.dicoms import load_pickle, dicom_to_img
from utils.dataset import T1T2Dataset
from utils.mask import shape_to_mask
from utils.cfg import load_config

CONFIG = "../experiments/020.yaml"


def check_pixel_spacings(list_of_dicompaths):
    pixel_spacings = []
    for dicompath in list_of_dicompaths:
        dcm = pydicom.dcmread(dicompath)
        pixel_spacings.extend(dcm.PixelSpacing)
    if len(set(pixel_spacings)) > 1:
        raise ValueError(f"PixelSpacings for T1 and T2 images should be square and equal, but got {pixel_spacings}")
    return pixel_spacings[0]


def get_radius_matrix(image_dimensions):
    img_height, img_width = image_dimensions[:2]
    radius_matrix_width = img_width * 2 + 10  # made up extra size just in case
    radius_matrix_height = img_height * 2 + 10
    radius_matrix = np.zeros((radius_matrix_height, radius_matrix_width))
    for x in range(img_width):
        for y in range(img_height):
            dist = np.sqrt(x * x + y * y)
            radius_matrix[img_height + y, img_width + x] = dist
            radius_matrix[img_height + y, img_width - x] = dist
            radius_matrix[img_height - y, img_width + x] = dist
            radius_matrix[img_height - y, img_width - x] = dist
    return radius_matrix


def pad_if_needed(img, min_height, min_width, centre_pad):
    input_height, input_width = img.shape[:2]
    new_shape = list(img.shape)
    new_shape[0] = max(input_height, min_height)
    new_shape[1] = max(input_width, min_width)
    out = np.zeros(new_shape, dtype=img.dtype)
    if centre_pad:
        row_from, col_from = 0, 0
        if input_height < min_height:
            row_from = (min_height - input_height) // 2
        if input_width < min_width:
            col_from = (min_width - input_width) // 2
        out[row_from:row_from + input_height, col_from:col_from + input_width] = img
    else:
        out[:input_height, :input_width] = img
    return out


def calc_gauss_on_a_scalar_or_matrix(distance, sigma):
    # partly gaussian but partly a hyperbola or something, so there is some gradient, however feeble, *everwhere*
    return 0.8 * np.exp(-(distance ** 2) / (2 * sigma ** 2)) + 0.2 * (1 / (1 + distance))


def export_label(labelpath):
    label = load_pickle(labelpath)
    if any(label_class not in label for label_class in label_classes):
        print(f"Labels missing for study {labelpath} (only {label.keys()} present)")
        return None

    dicompaths = glob(os.path.join(os.path.dirname(labelpath), "*.dcm"))
    pixel_spacing = check_pixel_spacings(dicompaths)
    datedir = os.path.basename(os.path.dirname(os.path.dirname(labelpath)))
    studydir = os.path.basename(os.path.dirname(labelpath))

    # 1. GENERATE IMAGE IN MM^2 FORMAT
    export_mm = {}
    t1, t2 = False, False
    for dicompath in dicompaths:
        if 'T1' in os.path.basename(dicompath):
            i_seq = 1
            t1 = True
        elif 'T2' in os.path.basename(dicompath):
            i_seq = 2
            t2 = True
        else:
            continue
        dcm = pydicom.dcmread(dicompath)
        seq = dicom_to_img(dcm)

        seq_mm = skimage.transform.rescale(seq, pixel_spacing, order=3)
        height_mm, width_mm = seq_mm.shape[:2]
        seq_mm = (seq_mm * 255).astype(np.uint8)
        export_mm[i_seq] = seq_mm

    if not t1 or not t2:  # If missing a seq, skip
        print(f"Missing DICOM files for {labelpath}: {t1}, {t2}")
        return None

    # 2. GENERATE MASK IN MM^2 FORMAT
    mask_mm = np.zeros((height_mm, width_mm, len(label_classes)))
    radius_matrix = get_radius_matrix((height_mm, width_mm))
    for i_label_class, label_class in enumerate(label_classes):
        points_labelled = [tuple((c[1] * pixel_spacing, c[0] * pixel_spacing)) for c in label[label_class]]
        points_labelled.append(points_labelled[0])  # Add first point to end to make full loop
        points_labelled = np.array(points_labelled)

        n_points = len(points_labelled)

        points_labelled_i = np.arange(n_points)
        points_labelled_x = points_labelled[:, 0]
        points_labelled_y = points_labelled[:, 1]

        fx = interpolate.make_interp_spline(points_labelled_i, points_labelled_x, k=3, bc_type=([(2, 0)], [(2, 0)]))
        fy = interpolate.make_interp_spline(points_labelled_i, points_labelled_y, k=3, bc_type=([(2, 0)], [(2, 0)]))

        interp_t = np.linspace(0, n_points - 1, 200)
        interp_x = fx(interp_t)
        interp_y = fy(interp_t)

        multi_radius_stack = np.zeros((len(interp_x), height_mm, width_mm))

        for i_interp_point, (x, y) in enumerate(zip(interp_x, interp_y)):
            x = int(max(0, min(width_mm - 1, x)))
            y = int(max(0, min(height_mm - 1, y)))

            multi_radius_stack[i_interp_point, :, :] = radius_matrix[
                                                height_mm - int(y):2 * height_mm - int(y),
                                                width_mm - int(x):2 * width_mm - int(x)]
            distances_to_sausage = np.amin(multi_radius_stack, 0)
            mask_mm[:, :, i_label_class] += calc_gauss_on_a_scalar_or_matrix(distances_to_sausage, gaussian_sigma)

    # Scale between 0 and 1
    mask_mm = mask_mm - np.min(mask_mm, axis=(0, 1))
    mask_mm = mask_mm / np.max(mask_mm)
    mask_mm = mask_mm[:height_mm, :width_mm]

    # SAVE
    # Dicom
    imgoutpath_t1 = f"t1_{datedir}_{studydir}.png"
    imgoutpath_t2 = f"t2_{datedir}_{studydir}.png"
    skimage.io.imsave(os.path.join(output_dir, imgoutpath_t1), export_mm[1], check_contrast=False)
    skimage.io.imsave(os.path.join(output_dir, imgoutpath_t2), export_mm[2], check_contrast=False)

    # Mask
    maskoutpath = f"m_{datedir}_{studydir}.npz"
    np.savez_compressed(os.path.join(output_dir, maskoutpath), mask=mask_mm)

    # Combined
    combined_mm = np.dstack((export_mm[2]/255, mask_mm[:,:,0], mask_mm[:,:,1]))
    combined_mm = (combined_mm * 255).astype(np.uint8)
    comboutpath = f"c_{datedir}_{studydir}.png"
    skimage.io.imsave(os.path.join(output_dir, comboutpath), combined_mm, check_contrast=False)


# RUN
# Load config
cfg, vis_dir, model_dir = load_config(CONFIG)
dicom_dir = cfg['data']['dicomdir']
output_dir = os.path.join(cfg['data']['pngdir_pose'])
label_classes = cfg['data']['input_classes']
gaussian_sigma = cfg['data']['gaussian_sigma']


if __name__ == "__main__":

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    labelpaths = glob(os.path.join(dicom_dir, "**", "label.pickle"), recursive=True)

    N_WORKERS = multiprocessing.cpu_count()

    with multiprocessing.Pool(N_WORKERS) as p:
        for _ in tqdm(p.imap(export_label, labelpaths), total=len(labelpaths)):
            pass
