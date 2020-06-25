import os
import cv2
import pydicom
import numpy as np
import skimage.io
import skimage.transform
import albumentations as A
from glob import glob
from tqdm import tqdm

from utils.dicoms import load_pickle, dicom_to_img
from utils.dataset import T1T2Dataset
from utils.mask import shape_to_mask
from utils.cfg import load_config

CONFIG = "../experiments/011.yaml"

# Load config
cfg, vis_dir, model_dir = load_config(CONFIG)

dicom_dir = cfg['data']['dicomdir']
output_dir_lowres = os.path.join(cfg['data']['pngdir'], 'low')
output_dir_highres = os.path.join(cfg['data']['pngdir'], 'high')
label_classes = cfg['data']['input_classes']
height_lowres, width_lowres = cfg['data']['low_res_crop']
height_highres, width_highres = cfg['data']['high_res_crop']
if not os.path.exists(output_dir_lowres):
    os.makedirs(output_dir_lowres)
if not os.path.exists(output_dir_highres):
    os.makedirs(output_dir_highres)


def pad_if_needed(img, min_height, min_width):
    input_height, input_width = img.shape[:2]
    new_shape = list(img.shape)
    new_shape[0] = max(input_height, min_height)
    new_shape[1] = max(input_width, min_width)
    row_from, col_from = 0, 0
    if input_height < min_height:
        row_from = (min_height - input_height) // 2
    if input_width < min_width:
        col_from = (min_width - input_width) // 2
    out = np.zeros(new_shape, dtype=img.dtype)
    out[row_from:row_from+input_height, col_from:col_from+input_width] = img
    return out


def center_crop(img, crop_height, crop_width, centre=None):
    """Either crops by the center of the image, or around a supplied point.
    Does not pad; if the supplied centre is towards the egde of the image, the padded
    area is shifted so crops start at 0 and only go up to the max row/col"""
    input_height, input_width = img.shape[:2]
    if centre is None:
        row_from = (input_height - crop_height)//2
        col_from = (input_width - crop_width)//2
    else:
        row_centre, col_centre = centre
        row_from = max(row_centre - (crop_height//2), 0)
        if (row_from + crop_height) > input_height:
            row_from -= (row_from + crop_height - input_height)
        col_from = max(col_centre - (crop_width//2), 0)
        if (col_from + crop_width) > input_width:
            col_from -= (col_from + crop_width - input_width)
    return img[row_from:row_from+crop_height, col_from:col_from+crop_width]


def check_pixel_spacings(list_of_dicompaths):
    pixel_spacings = []
    for dicompath in list_of_dicompaths:
        dcm = pydicom.dcmread(dicompath)
        pixel_spacings.extend(dcm.PixelSpacing)
    if len(set(pixel_spacings)) > 1:
        raise ValueError(f"PixelSpacings for T1 and T2 images should be square and equal, but got {pixel_spacings}")
    return pixel_spacings[0]


labelpaths = glob(os.path.join(dicom_dir, "**", "label.pickle"), recursive=True)
for labelpath in tqdm(labelpaths):

    label = load_pickle(labelpath)
    if any(label_class not in label for label_class in label_classes):
        print(f"Labels missing for study {labelpath} (only {label.keys()} present)")
        continue

    dicompaths = glob(os.path.join(os.path.dirname(labelpath), "*.dcm"))
    pixel_spacing = check_pixel_spacings(dicompaths)
    datedir = os.path.basename(os.path.dirname(os.path.dirname(labelpath)))
    studydir = os.path.basename(os.path.dirname(labelpath))
    imgoutpath = f"i_{datedir}_{studydir}.png"
    input_height, input_width = pydicom.dcmread(dicompaths[0]).pixel_array.shape

    # MASK
    # Do mask first so we can use the heart's location for the highres cropping
    # Create mask at native res
    mask = np.zeros((input_height, input_width), dtype=np.uint8)
    for i_label_class, label_class in enumerate(label_classes):
        coords = label[label_class]
        points = [tuple((c[1],c[0])) for c in coords]
        mask = mask + shape_to_mask((input_height, input_width), i_label_class + 1, points)
    # Upsample mask so pixelsize is eg 1mm2
    mask_mm = skimage.transform.rescale(mask, pixel_spacing, order=0, anti_aliasing=False)
    mask_mm = (mask_mm * 255).astype(np.uint8)
    # Find centre
    centre_row_mm, centre_col_mm = T1T2Dataset.get_high_res_coords_for_mask(mask_mm, get='centre')
    # Centercrop
    mask_lowres = center_crop(pad_if_needed(mask_mm, height_lowres, width_lowres), height_lowres, width_lowres)
    mask_highres = center_crop(mask_mm, height_highres, width_highres, centre=(centre_row_mm, centre_col_mm))
    # Export
    maskoutpath = f"m_{datedir}_{studydir}.png"
    skimage.io.imsave(os.path.join(output_dir_lowres, maskoutpath), mask_lowres, check_contrast=False)
    skimage.io.imsave(os.path.join(output_dir_highres, maskoutpath), mask_highres, check_contrast=False)

    # IMAGE
    # Create exported image
    export_img_lowres = np.zeros((height_lowres, width_lowres, 3), dtype=np.uint8)
    export_img_highres = np.zeros((height_highres, width_highres, 3), dtype=np.uint8)
    # Loop over dicoms, ensure we find T1 and T2 sequences
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

        # Upsample e.g. to pixel per mm2
        seq_mm = skimage.transform.rescale(seq, pixel_spacing, order=3)
        seq_mm = (seq_mm * 255).astype(np.uint8)

        # Crop
        seq_lowres = center_crop(pad_if_needed(seq_mm, height_lowres, width_lowres), height_lowres, width_lowres)
        seq_highres = center_crop(seq_mm, height_highres, width_highres, centre=(centre_row_mm, centre_col_mm))

        export_img_lowres[:, :, i_seq] = seq_lowres
        export_img_highres[:, :, i_seq] = seq_highres

    if not t1 or not t2:  # If missing a seq, skip
        print(f"Missing DICOM files for {labelpath}: {t1}, {t2}")
        continue

    # Save IMAGE
    skimage.io.imsave(os.path.join(output_dir_lowres, imgoutpath), export_img_lowres, check_contrast=False)
    skimage.io.imsave(os.path.join(output_dir_highres, imgoutpath), export_img_highres, check_contrast=False)

    # COMBINED
    # Add mask to red channel which has been left empty
    combinedoutpath = f"c_{datedir}_{studydir}.png"

    mask_scaled_lowres = mask_lowres.astype(np.float32) * (255 / np.max(mask_lowres)).astype(np.uint8)  # Scale mask to 255
    combined_lowres = export_img_lowres.copy()
    combined_lowres[:,:,0] = mask_scaled_lowres
    skimage.io.imsave(os.path.join(output_dir_lowres, combinedoutpath), combined_lowres, check_contrast=False)

    mask_scaled_highres = mask_highres.astype(np.float32) * (255 / np.max(mask_highres)).astype(np.uint8)  # Scale mask to 255
    combined_highres = export_img_highres.copy()
    combined_highres[:,:,0] = mask_scaled_highres
    skimage.io.imsave(os.path.join(output_dir_highres, combinedoutpath), combined_highres, check_contrast=False)


