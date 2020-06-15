import numpy as np
import pydicom
from skimage.transform import rescale
import matplotlib.pyplot as plt
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

# Load the ONNX model
sess = ort.InferenceSession("../../output/models/009/t1t2_fcn.onnx")
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# Load data
path_t1 = "../../data/dicoms/by_date_by_study/20200423/T1SR_Mapping_SASHA_HC_T1T2_141613_5906470_5906478_97_20200423-102518_dicom/T1_SLC0_CON0_PHS0_REP0_SET0_AVE0_1.dcm"
path_t2 = "../../data/dicoms/by_date_by_study/20200423/T1SR_Mapping_SASHA_HC_T1T2_141613_5906470_5906478_97_20200423-102518_dicom/T2_SLC0_CON0_PHS0_REP0_SET0_AVE0_1.dcm"

img = [dicompath_to_img(path_t1), dicompath_to_img(path_t2)]
img = np.dstack(img)  # 2 * H * W
img = ((img/255)-MEAN)/STD
img = img.transpose((2,0,1))

img_batch = np.expand_dims(img, 0).astype(np.float32)

pred_batch = sess.run([output_name], {input_name:img_batch})[0]  # Returns a list of len 1
pred = pred_batch[0].transpose((1,2,0))  # Channels last for upsampling
#pred = rescale(pred, (4, 4), multichannel=True)  # Upsample by 4
pred_cls = np.argmax(pred, axis=2)  # Classes

fig, axes = plt.subplots(1, 3, figsize=(12,4))

axes[0].imshow(img[0])
axes[0].set_title("T1")
axes[0].set_axis_off()

axes[1].imshow(img[1])
axes[1].set_title("T2")
axes[1].set_axis_off()

axes[2].imshow(pred_cls, vmin=0, vmax=2)
axes[2].set_title("Predictions")
axes[2].set_axis_off()

plt.show()