import os
import re
import pickle
import pydicom
import numpy as np
from glob import glob
from tqdm import tqdm
from collections import defaultdict


def split_dicoms_into_patients_and_sequences(dicompaths):
    dicoms = defaultdict(lambda: defaultdict(list))
    for dicompath in tqdm(dicompaths):
        dcm = pydicom.dcmread(dicompath)
        if dcm.SequenceName != 'tfi2d1_72':
            continue
        if 'T1 MAP' not in dcm.ImageType and 'T2 MAP' not in dcm.ImageType:
            continue
        name = dcm.PatientName
        actime = dcm.AcquisitionTime
        imgtype = dcm.ImageType
        dicoms[name][actime].append({'path': dicompath, 'type': imgtype})
    return dicoms

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


def get_sequences(path, old_path=None):
    sequences = {}
    hospital_dirs = [f for f in sorted(glob(os.path.join(path, "*"))) if os.path.isdir(f)]
    for hospital_dir in hospital_dirs:
        t1map_paths = glob(os.path.join(hospital_dir, "t1_map_numpy", "*.npy"))
        for t1map_path in t1map_paths:
            file_id = os.path.splitext(os.path.basename(t1map_path))[0]
            # e.g. T1SR_Mapping_SASHA_HC_T1T2_141316_06902422_06902429_434_20200602-150921_1.npy
            #      T1SR_Mapping_SASHA_HC_T1T2_42363_688280696_688280701_245_20200626-131035_1_slc1.npy"
            #      ^ seq_name                 ^scanner  ^ pid ^ sid    ^ meas_id ^time     ^run    ^slice

            template = "(.*)_([0-9]{3,})_([0-9]{5,})_([0-9]{5,})_([0-9]{2,})_([0-9]{8})-([0-9]{6,})_([0-9]{1,})_?(.*)?"
            matches = re.findall(template, file_id)[0]  # Returns a list of len 1, with a tuple of 9
            assert len(matches) == 9, f"Expected 9 matches but got {len(matches)}: {matches}"
            seq_name, scanner_id, patient_id, study_id, meas_id, date, time, run_id, slice_id = matches

            if slice_id:
                slice_id = int(slice_id[-1])  # slc3 -> 3
            else:
                slice_id = 0

            t2map_path = os.path.join(hospital_dir, "t2_map_numpy", os.path.basename(t1map_path))
            t2img_path = os.path.join(hospital_dir, "t2_last_image_numpy", os.path.basename(t1map_path))
            report_path = os.path.join(hospital_dir, "seg_labels", os.path.basename(t1map_path)+'.pickle')
            if not os.path.exists(t2map_path):
                print(f"Unable to find T2 map matching {t1map_path}")
                continue
            if not os.path.exists(t2img_path):
                print(f"Unable to find last T2 image matching {t1map_path}")
                continue
            reported = True if os.path.exists(report_path) else False

            # if old_path:
            #     print(f"{file_id} -> {date} -> {date}")
            #     old_study_folder = os.path.basename(t1map_path).rsplit('_',1)[0] + "_dicom"
            #     old_report_path = os.path.join(old_path, date, old_study_folder, f"label_{max(slice_id, 1)}.pickle")
            #     print(f"Looking for {old_report_path} -> {os.path.exists(old_report_path)}")
            #     if os.path.exists(old_report_path):
            #         reported_old = True
            #     else:
            #         reported_old = False
            # else:
            #     reported_old = False

            scan_id = f"{scanner_id}_{patient_id}_{study_id}_{meas_id}_{date}-{time}_{run_id} - {slice_id}"
            assert scan_id not in sequences, f"Found clashing ID {scan_id}"
            sequences[scan_id] = {
                't1map_path': t1map_path,
                't2map_path': t2map_path,
                't2img_path': t2img_path,
                'report_path': report_path,
                'reported': reported,
                # 'report_path_old': old_report_path,
                # 'reported_old': reported_old
            }




    # for seqpath in sequencepaths:
    #
    #     t1_paths = glob(os.path.join(seqpath, "T1_*.dcm"))
    #
    #     slice_ids = [os.path.splitext(os.path.basename(path))[0].rsplit('_',1)[1] for path in t1_paths]
    #
    #     for slice_id, t1_path in zip(slice_ids, t1_paths):
    #
    #         entry = f"{os.path.basename(os.path.dirname(seqpath))} - {os.path.basename(seqpath)} - {slice_id}"
    #         reported = True if os.path.exists(os.path.join(seqpath, f"label_{slice_id}.pickle")) else False
    #
    #         sequences[entry] = {
    #             't1_path': t1_path,
    #             'reported': reported
    #         }

    return sequences