import os
import re
import pickle
import numpy as np
from glob import glob
from tqdm import tqdm
from collections import defaultdict

try:
    import pydicom
except ImportError:
    pass

REGEX_HUI = "(.*)_([0-9]{3,})_([0-9]{5,})_([0-9]{5,})_([0-9]{2,})_([0-9]{8})-([0-9]{6,})_([0-9]{1,})_?(.*)?"
REGEX_PETER = "(.*)_([0-9]{4,})_([0-9]{4,})_([0-9]{4,})_([0-9]{1,})_([0-9]{8})-([0-9]{6})"


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pickle(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def dicom_to_img(dicom):
    if type(dicom) == str:
        dcm = pydicom.dcmread(dicom)
    else:
        dcm = dicom
    window_min = max(0, dcm.WindowCenter - dcm.WindowWidth)
    frame = dcm.pixel_array - window_min
    frame = frame / dcm.WindowWidth
    frame = np.clip(frame, 0, 1)
    frame = (frame * 255).astype(np.uint8)
    return frame


def get_studies_peter(path):
    sequences = {}
    numpy_paths = glob(os.path.join(path, "**", "*.npy"), recursive=True)
    for numpy_path in numpy_paths:
        # PETER FORMAT
        # DIR:
        #   20200313 \ T1T2_141613_25752396_25752404_256_20200711-135533 \ ...
        #   ^ date     ^seq ^scanr ^sid     ^pid     ^meas_id ^datetime
        #
        #
        # ... T1_T2_PD_SLC0_CON0_PHS0_REP0_SET0_AVE0_1.npy
        # ... T1_T2_PD_SLC1_CON0_PHS0_REP0_SET0_AVE0_2.npy
        # ... T1_T2_PD_SLC2_CON0_PHS0_REP0_SET0_AVE0_3.npy

        dirname = os.path.basename(os.path.dirname(numpy_path))
        matches = re.findall(REGEX_PETER, dirname)[0]  # Returns a list of len 1, with a tuple of 7
        assert len(matches) == 7, f"Expected 7 matches but got {len(matches)}: {matches}"

        seq_name, scanner_id, study_id, patient_id, meas_id, date, time = matches
        run_id = os.path.splitext(os.path.basename(numpy_path))[0].rsplit('_', 1)[1]

        report_path = numpy_path + '.pickle'
        reported = 'peter' if os.path.exists(report_path) else 'no'

        scan_id = f"{scanner_id}_{patient_id}_{study_id}_{meas_id}_{date}-{time} - {run_id}"
        assert scan_id not in sequences, f"Found clashing ID {scan_id}"
        sequences[scan_id] = {
            'numpy_path': numpy_path,
            'report_path': report_path,
            'reported': reported,
        }
    return sequences
