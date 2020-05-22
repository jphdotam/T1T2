import os
import skimage.io
import numpy as np
from glob import glob
from tqdm import tqdm

from utils.dicoms import load_pickle, dicompath_to_img
from utils.mask import shape_to_mask

DICOM_DIR = "../data/dicoms/by_date_by_study"
OUTPUT_DIR = "../data/pngs"
LABEL_CLASSES = ('epi', 'endo')

labelpaths = glob(os.path.join(DICOM_DIR, "**","label.pickle"), recursive=True)

for labelpath in tqdm(labelpaths):
    label = load_pickle(labelpath)

    if any(label_class not in label for label_class in LABEL_CLASSES):
        print(f"Labels missing for study {labelpath} (only {label.keys()} present)")
        continue

    dicompaths = glob(os.path.join(os.path.dirname(labelpath), "*.dcm"))
    if len(dicompaths) < 2:
        print(f"Found label {labelpath} but only {len(dicompaths)} dicoms")
        continue

    if 'dims' in label:
        img_height, img_width = label['dims']
    else:
        dcm = dicompath_to_img(dicompaths[0])
        img_height, img_width = dcm.shape[:2]

    # Create input image
    img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    t1,t2 = False, False
    for dicompath in dicompaths:
        if 'T1' in os.path.basename(dicompath):
            i_seq = 1
            t1 = True
        elif 'T2' in os.path.basename(dicompath):
            i_seq = 2
            t2 = True
        else:
            continue

        img[:,:,i_seq] = dicompath_to_img(dicompath)

    if not t1 or not t2:  # If haven't found either of them, skip
        print(f"Missing DICOM files for {labelpath}: {t1}, {t2}")
        continue

    datedir = os.path.basename(os.path.dirname(os.path.dirname(labelpath)))
    studydir = os.path.basename(os.path.dirname(labelpath))
    imgoutpath = f"i_{datedir}_{studydir}.png"
    skimage.io.imsave(os.path.join(OUTPUT_DIR, imgoutpath), img, check_contrast=False)

    # Create mask
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    for i_label_class, label_class in enumerate(LABEL_CLASSES):
        coords = label[label_class]
        points = [tuple((c[1],c[0])) for c in coords]
        mask = mask + shape_to_mask((img_height, img_width), i_label_class+1, points)

    maskoutpath = f"m_{datedir}_{studydir}.png"
    skimage.io.imsave(os.path.join(OUTPUT_DIR, maskoutpath), mask, check_contrast=False)

    # Combined - add mask to red channel which has been left empty
    mask_scaled = mask.astype(np.float32)*(255/np.max(mask)).astype(np.uint8)  # Scale mask to 255
    combined = img.copy()
    combined[:,:,0] = mask_scaled
    combinedoutpath = f"c_{datedir}_{studydir}.png"
    skimage.io.imsave(os.path.join(OUTPUT_DIR, combinedoutpath), combined, check_contrast=False)
