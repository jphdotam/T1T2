import os
import skimage.io
import numpy as np

from utils.dicoms import get_sequences_by_report_for_folder, load_pickle, dicompath_to_img
from utils.mask import shape_to_mask

DICOM_DIR = "../../data/dicoms"
OUTPUT_DIR = "../../data/pngs"
LABELS = ('epi', 'endo')

sequences_by_report = get_sequences_by_report_for_folder(DICOM_DIR)

for reportpicklepath, sequences in sequences_by_report.items():
    report = load_pickle(reportpicklepath)

    if any(label not in report for label in LABELS):
        continue

    if 'dims' not in report:
        dcm = dicompath_to_img(sequences[0]['path'])
        img_height, img_width = dcm.shape[:2]
        print(f"{reportpicklepath}: {dcm.shape}")
    else:
        img_height, img_width = report['dims']

    # Create input image
    img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    t1,t2 = False, False
    for seq in sequences:
        if 'T1' in seq['type']:
            i_seq = 1
            t1 = True
        elif 'T2' in seq['type']:
            i_seq = 2
            t2 = True
        else:
            continue

        img[:,:,i_seq] = dicompath_to_img(seq['path'])

    if not t1 or not t2:  # If haven't found both of them, skip
        continue

    imgoutpath = os.path.join(OUTPUT_DIR, f"i_{os.path.splitext(os.path.basename(reportpicklepath))[0]}.png")
    skimage.io.imsave(os.path.join(OUTPUT_DIR, imgoutpath), img, check_contrast=False)

    # Create mask
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    for i_label, label in enumerate(LABELS):
        coords = report[label]
        points = [tuple((c[1],c[0])) for c in coords]
        # points = numpy_indices_from_coords(coords, img_height, img_width)
        mask = mask + shape_to_mask((img_height, img_width), i_label+1, points)

    maskoutpath = os.path.join(OUTPUT_DIR, f"m_{os.path.splitext(os.path.basename(reportpicklepath))[0]}.png")
    skimage.io.imsave(os.path.join(OUTPUT_DIR, maskoutpath), mask, check_contrast=False)

    # Combined - add mask to red channel which has been left empty
    mask_scaled = mask.astype(np.float32)*(255/np.max(mask)).astype(np.uint8)  # Scale mask to 255
    combined = img.copy()
    combined[:,:,0] = mask_scaled
    combinedoutpath = os.path.join(OUTPUT_DIR, f"c_{os.path.splitext(os.path.basename(reportpicklepath))[0]}.png")
    skimage.io.imsave(os.path.join(OUTPUT_DIR, combinedoutpath), combined, check_contrast=False)
