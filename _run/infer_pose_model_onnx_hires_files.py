import pydicom
import numpy as np
import skimage.transform
import matplotlib.pyplot as plt

import onnxruntime as ort

from utils.inference import center_crop, dicom_to_img, pad_if_needed, pose_mask_to_coords

# Settings
MEAN = [0.3023, 0.4016]
STD = [0.2582, 0.3189]
FOV_CROP = 256
DOUBLE_INPUT_RES = True
MODELPATH = "../output/models/022/hrnet_pose.onnx"

# Plot settings
RAISE_TO_POWER = 3
NORMALISE_PREDS = True

# Load data
# path_t1 = "../data/dicoms/by_date_by_study/20200423/T1SR_Mapping_SASHA_HC_T1T2_141613_5906470_5906478_97_20200423-102518_dicom/T1_SLC0_CON0_PHS0_REP0_SET0_AVE0_1.dcm"
# path_t2 = "../data/dicoms/by_date_by_study/20200423/T1SR_Mapping_SASHA_HC_T1T2_141613_5906470_5906478_97_20200423-102518_dicom/T2_SLC0_CON0_PHS0_REP0_SET0_AVE0_1.dcm"
# path_t1 = "../data/dicoms/by_date_by_study/20200505/T1SR_Mapping_SASHA_HC_T1T2_42363_622646938_622646943_2398_20200505-120210_dicom/T1_SLC0_CON0_PHS0_REP0_SET0_AVE0_1.dcm"
# path_t2 = "../data/dicoms/by_date_by_study/20200505/T1SR_Mapping_SASHA_HC_T1T2_42363_622646938_622646943_2398_20200505-120210_dicom/T2_SLC0_CON0_PHS0_REP0_SET0_AVE0_1.dcm"
# path_t1 = "../data/dicoms/bdbs_new/20200615/T1SR_Mapping_SASHA_HC_T1T2_42363_671978570_671978575_58_20200615-103059_dicom/T1_SLC0_CON0_PHS0_REP0_SET0_AVE0_1.dcm"
# path_t2 = "../data/dicoms/bdbs_new/20200615/T1SR_Mapping_SASHA_HC_T1T2_42363_671978570_671978575_58_20200615-103059_dicom/T2_SLC0_CON0_PHS0_REP0_SET0_AVE0_1.dcm"

path_t1s = ["E:/hui/T1_SLC0_CON0_PHS0_REP0_SET0_AVE0_1.dcm", "E:/hui/T1_SLC1_CON0_PHS0_REP0_SET0_AVE0_4.dcm", "E:/hui/T1_SLC2_CON0_PHS0_REP0_SET0_AVE0_7.dcm"]
path_t2s = ["E:/hui/T2_SLC0_CON0_PHS0_REP0_SET0_AVE0_1.dcm", "E:/hui/T2_SLC1_CON0_PHS0_REP0_SET0_AVE0_4.dcm", "E:/hui/T2_SLC2_CON0_PHS0_REP0_SET0_AVE0_7.dcm"]

for i_seq, (path_t1, path_t2) in enumerate(zip(path_t1s, path_t2s)):
    # Load DICOM
    img_t1_native = dicom_to_img(path_t1)
    img_t2_native = dicom_to_img(path_t2)
    input_height, input_width = img_t1_native.shape[:2]
    input_pixelspacing = pydicom.dcmread(path_t1).PixelSpacing

    # Rescale to MM
    img_t1_mm = skimage.transform.rescale(img_t1_native, input_pixelspacing, order=3)
    img_t2_mm = skimage.transform.rescale(img_t2_native, input_pixelspacing, order=3)
    img_mm_cl = np.dstack((img_t1_mm, img_t2_mm))

    # Crop and double res if needed (for HRNet which downsamples masks)
    img_crop_cl, topleft_crop = center_crop(pad_if_needed(img_mm_cl, FOV_CROP, FOV_CROP), FOV_CROP, FOV_CROP)
    img_crop_cl_norm = (img_crop_cl - MEAN) / STD
    if DOUBLE_INPUT_RES:
        img_crop_cf = skimage.transform.rescale(img_crop_cl_norm, 2, order=3, multichannel=True).transpose((2, 0, 1))
    else:
        img_crop_cf = img_crop_cl_norm.transpose((2, 0, 1))

    # Predict
    # Load the ONNX low-res model
    sess = ort.InferenceSession(MODELPATH)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    img_batch = np.expand_dims(img_crop_cf, 0).astype(np.float32)
    pred_batch = sess.run([output_name], {input_name: img_batch})[0]  # Returns a list of len 1

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(20, 20), gridspec_kw={'wspace': 0, 'hspace': 0})

    for i_sequence, sequence in enumerate(('T1', 'T2')):
        # img
        img_dcm = img_crop_cl[:, :, i_sequence]
        img_dcm = np.dstack((img_dcm, img_dcm, img_dcm))
        img_plot = img_dcm.copy()

        for i_channel in range(2):
            # pred mask
            pred = pred_batch[0, i_channel]
            #np.save(f"E:/{i_seq}.npy", pred_batch)
            if RAISE_TO_POWER > 1:
                pred = np.power(pred, RAISE_TO_POWER)
            if NORMALISE_PREDS:
                pred = pred - pred.min()
                pred = pred / (pred.max() + 1e-8)

            # pred coords
            pred_coords = pose_mask_to_coords(prediction_mask=pred_batch[0, i_channel])
            pred_coords = [[coord['x'], coord['y']] for coord in pred_coords]
            xs, ys = zip(*pred_coords)

            # top row - image & raw predicted masks
            img_plot[:, :, i_channel] = img_dcm[:, :, i_channel] + pred

            axes[0][i_sequence].imshow(img_plot, cmap='gray')
            axes[0][i_sequence].set_title(sequence)
            axes[0][i_sequence].set_axis_off()

            # bottom row - image & interpreted masks
            axes[1][i_sequence].imshow(img_dcm, cmap='gray')
            axes[1][i_sequence].plot(xs, ys)

    plt.show()
