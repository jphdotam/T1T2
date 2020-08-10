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


def get_studies_hui(path):
    sequences = {}
    hospital_dirs = [f for f in sorted(glob(os.path.join(path, "*"))) if os.path.isdir(f)]
    for hospital_dir in hospital_dirs:
        t1map_paths = glob(os.path.join(hospital_dir, "t1_map_numpy", "*.npy"))
        for t1map_path in t1map_paths:
            file_id = os.path.splitext(os.path.basename(t1map_path))[0]
            # HUI FORMAT
            # DIR:
            #   HAMMERSMITH_T1T2_2020_AI_1mm \ t2_last_image_numpy \ <filename> \ ...
            # ...  T1SR_Mapping_SASHA_HC_T1T2_141316_06902422_06902429_434_20200602-150921_1.npy
            # ...  T1SR_Mapping_SASHA_HC_T1T2_42363_688280696_688280701_245_20200626-131035_1_slc1.npy"
            #      ^ seq_name                 ^scanner  ^ pid ^ sid    ^ meas_id ^time     ^run    ^slice

            matches = re.findall(REGEX_HUI, file_id)[0]  # Returns a list of len 1, with a tuple of 9
            assert len(matches) == 9, f"Expected 9 matches but got {len(matches)}: {matches}"
            seq_name, scanner_id, patient_id, study_id, meas_id, date, time, run_id, slice_id = matches

            if slice_id:
                slice_id = int(slice_id[-1])  # slc3 -> 3
            else:
                slice_id = 0

            t2map_path = os.path.join(hospital_dir, "t2_map_numpy", os.path.basename(t1map_path))
            t2img_path = os.path.join(hospital_dir, "t2_last_image_numpy", os.path.basename(t1map_path))
            report_path = os.path.join(hospital_dir, "seg_labels", os.path.basename(t1map_path) + '.pickle')
            if not os.path.exists(t2map_path):
                print(f"Unable to find T2 map matching {t1map_path}")
                continue
            if not os.path.exists(t2img_path):
                print(f"Unable to find last T2 image matching {t1map_path}")
                continue
            reported = True if os.path.exists(report_path) else False

            scan_id = f"{scanner_id}_{patient_id}_{study_id}_{meas_id}_{date}-{time}_{run_id} - {slice_id}"
            assert scan_id not in sequences, f"Found clashing ID {scan_id}"
            sequences[scan_id] = {
                't1map_path': t1map_path,
                't2map_path': t2map_path,
                't2img_path': t2img_path,
                'report_path': report_path,
                'reported': reported,
            }

    return sequences


def get_studies_peter(path, check_hui_label):
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

        if reported == 'no' and check_hui_label:
            hui_label_path = get_hui_report_path(numpy_path, hui_data_dir=check_hui_label)
            if hui_label_path:
                reported = 'hui'

        scan_id = f"{scanner_id}_{patient_id}_{study_id}_{meas_id}_{date}-{time} - {run_id}"
        assert scan_id not in sequences, f"Found clashing ID {scan_id}"
        sequences[scan_id] = {
            'numpy_path': numpy_path,
            'report_path': report_path,
            'reported': reported,
        }
    return sequences


def get_hui_report_path(peter_path, hui_data_dir):
    seq_name, scanner_id, study_id, patient_id, meas_id, date, time = re.findall(REGEX_PETER, os.path.basename(os.path.dirname(peter_path)))[0]  # Returns a list of len 1, with a tuple of 7
    run_id = os.path.splitext(os.path.basename(peter_path))[0].rsplit('_', 1)[1]
    matching_reports = glob(os.path.join(hui_data_dir, "**", "seg_labels", f"*{scanner_id}*{study_id}*{meas_id}*{date}*{time}*{run_id}*.npy.pickle"))
    if len(matching_reports) == 0:
        return False
    elif len(matching_reports) == 1:
        return matching_reports[0]
    else:
        print(f"Found > 1 matching report for {peter_path}: {matching_reports}")
        return False


def convert_hui_coords_to_peter_coords(hui_coords):
    peter_coords = defaultdict(list)
    dim0, dim1 = hui_coords['dims'][:2]
    for label_name, points in hui_coords.items():
        if type(points) == tuple:  # dims
            peter_coords[label_name] = points
        else:
            for point in points:
                peter_coords[label_name].append([point[1], dim0-point[0]])
    return peter_coords
