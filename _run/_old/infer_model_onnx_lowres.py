import cv2
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import albumentations as A
import onnxruntime as ort

def dicompath_to_img(dicompath):
    dcm = pydicom.dcmread(dicompath)
    window_min = max(0, dcm.WindowCenter - dcm.WindowWidth)
    frame = dcm.pixel_array - window_min
    frame = frame / dcm.WindowWidth
    frame = np.clip(frame, 0, 1)
    frame = (frame * 255).astype(np.uint8)
    return frame

# Settings
MEAN = [0.3023, 0.4016]
STD = [0.2582, 0.3189]
DIM = 256

# Load the ONNX model
sess = ort.InferenceSession("../../output/models/009/t1t2_fcn.onnx")
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# Load data
# path_t1 = "../../data/dicoms/by_date_by_study/20200423/T1SR_Mapping_SASHA_HC_T1T2_141613_5906470_5906478_97_20200423-102518_dicom/T1_SLC0_CON0_PHS0_REP0_SET0_AVE0_1.dcm"
# path_t2 = "../../data/dicoms/by_date_by_study/20200423/T1SR_Mapping_SASHA_HC_T1T2_141613_5906470_5906478_97_20200423-102518_dicom/T2_SLC0_CON0_PHS0_REP0_SET0_AVE0_1.dcm"
path_t1 = "../../data/dicoms/bdbs_new/20200615/T1SR_Mapping_SASHA_HC_T1T2_42363_671978570_671978575_58_20200615-103059_dicom/T1_SLC0_CON0_PHS0_REP0_SET0_AVE0_1.dcm"
path_t2 = "../../data/dicoms/bdbs_new/20200615/T1SR_Mapping_SASHA_HC_T1T2_42363_671978570_671978575_58_20200615-103059_dicom/T2_SLC0_CON0_PHS0_REP0_SET0_AVE0_1.dcm"

# Transforms
low_transf = A.Compose([A.LongestMaxSize(DIM, interpolation=cv2.INTER_CUBIC),
                        A.PadIfNeeded(min_height=DIM, min_width=DIM)])

# Load image and create batch
img_low_cl = [dicompath_to_img(path_t1), dicompath_to_img(path_t2)]
img_low_cl = np.dstack(img_low_cl)  # H * W * 2
img_low_cl = low_transf(image=img_low_cl)['image']
img_low_cl = ((img_low_cl / 255) - MEAN) / STD
img_low_cf = img_low_cl.transpose((2, 0, 1))  # 2 * H * W
img_batch_low = np.expand_dims(img_low_cf, 0).astype(np.float32)

# Forward pass & get classes for each pixel
pred_batch_low = sess.run([output_name], {input_name:img_batch_low})[0]  # Returns a list of len 1
pred_cls_low = np.argmax(pred_batch_low[0], axis=0)  # Classes

fig, axes = plt.subplots(1, 3, figsize=(12,4))

axes[0].imshow(img_low_cf[0])
axes[0].set_title("T1")
axes[0].set_axis_off()

axes[1].imshow(img_low_cf[1])
axes[1].set_title("T2")
axes[1].set_axis_off()

axes[2].imshow(pred_cls_low, vmin=0, vmax=2)
axes[2].set_title("Predictions")
axes[2].set_axis_off()

plt.show()