import numpy as np
import skimage.transform
import onnxruntime as ort

MODELPATH = "../output/models/026/110_0.0010794.pt.onnx"
SRC = "E:/Data/T1T2_peter/20200427/T1T2_42363_622646506_622646511_675_20200427-163827/T1_T2_PD_SLC0_CON0_PHS0_REP0_SET0_AVE0_1.npy"
FOV = 256


def normalize_data(data, WindowCenter, WindowWidth):
    window_min = max(0, WindowCenter - WindowWidth // 2)
    frame = data - window_min
    frame = frame / (WindowWidth)
    frame = np.clip(frame, 0, 1)
    frame = frame.astype(np.float32)
    return frame


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
    area is shifted so crops start at 0 and only go up to the max row/col
    Returns both the new crop, and the top-left coords as a row,col tuple"""
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
    return img[row_from:row_from+crop_height, col_from:col_from+crop_width], (row_from, col_from)


npy = np.load(SRC)
t1w, t2w, pd, t1, t2 = np.transpose(npy, (2, 0, 1))

t1_pre = normalize_data(t1, WindowCenter=1300.0, WindowWidth=1300.0)
t1_post = normalize_data(t1, WindowCenter=500.0, WindowWidth=1000.0)
t2 = normalize_data(t2, WindowCenter=60.0, WindowWidth=120.0)
t1w = t1w - t1w.min()
t1w /= t1w.max()
t2w = t2w - t2w.min()
t2w /= t2w.max()
pd = pd - pd.min()
pd /= pd.max()
t1_pre = (t1_pre*255).astype(np.uint8)
t1_post = (t1_post*255).astype(np.uint8)
t2 = (t2*255).astype(np.uint8)
t1w = (t1w*255).astype(np.uint8)
t2w = (t2w*255).astype(np.uint8)
pd = (pd*255).astype(np.uint8)

t1_t2 = np.dstack((t1w, t2w, pd, t1_pre, t1_post, t2))
t1_t2_crop, _top_left = center_crop(pad_if_needed(t1_t2, min_height=FOV, min_width=FOV), crop_height=FOV, crop_width=FOV)
t1_t2_double = skimage.transform.rescale(t1_t2_crop, 2, order=3, multichannel=True)

t1_t2_in = t1_t2_double.transpose((2, 0, 1))

print("t1t2 segmentation im is ", t1_t2_in.shape)

sess = ort.InferenceSession(MODELPATH)
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
img_batch = np.expand_dims(t1_t2_in, 0).astype(np.float32)
pred_batch = sess.run([output_name], {input_name: img_batch})[0]  # Returns a list of len 1


# preview
import matplotlib.pyplot as plt

for pred_channel in pred_batch[0]:
    # normalize predictions
    pred_channel = pred_channel - pred_channel.min()
    pred_channel = pred_channel / (pred_channel.max() + 1e-8)

    plt.imshow(pred_channel)
    plt.title("hui")
    plt.show()