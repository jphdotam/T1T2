import os
import pickle
import pydicom
import numpy as np
from glob import glob
from tqdm import tqdm
from collections import defaultdict


def get_dicoms_for_dir(path, ext=".IMA"):
    dicoms = glob(os.path.join(path, f"*{ext}"))
    return dicoms

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

def get_sequences_by_studies_for_folder(path):
    entries = {}
    dicompaths = get_dicoms_for_dir(path)
    dicoms = split_dicoms_into_patients_and_sequences(dicompaths)
    for ptname, actimes in dicoms.items():
        for actime, listofseqs in actimes.items():
            studyname = f"{ptname} - {actime}"
            reported = "reported" if os.path.exists(os.path.join(path, f"{studyname}.pickle")) else "unreported"
            displayname = f"{ptname} - {actime} ({len(listofseqs)} sequences) - {reported}"
            entries[displayname] = listofseqs
    return entries

def get_sequences_by_report_for_folder(path):
    entries = {}
    dicompaths = get_dicoms_for_dir(path)
    dicoms = split_dicoms_into_patients_and_sequences(dicompaths)
    for ptname, actimes in dicoms.items():
        for actime, listofseqs in actimes.items():
            studyname = f"{ptname} - {actime}"
            picklepath = os.path.join(path, f"{studyname}.pickle")
            if os.path.exists(picklepath):
                entries[picklepath] = listofseqs
    return entries

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pickle(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def dicompath_to_img(dicompath):
    dcm = pydicom.dcmread(dicompath)
    window_min = max(0, dcm.WindowCenter - dcm.WindowWidth)
    frame = dcm.pixel_array - window_min
    frame = frame / dcm.WindowWidth
    frame = np.clip(frame, 0, 1)
    frame = (frame * 255).astype(np.uint8)
    return frame