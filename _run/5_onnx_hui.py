import numpy as np

SRC = "E:/Data/T1T2_peter/20200427/T1T2_42363_622646506_622646511_675_20200427-163827/T1_T2_PD_SLC0_CON0_PHS0_REP0_SET0_AVE0_1.npy"


def normalize_data(data, WindowCenter, WindowWidth):
    window_min = max(0, WindowCenter - WindowWidth)
    frame = data - window_min
    frame = frame / (WindowWidth)
    frame = np.clip(frame, 0, 1)
    frame = frame.astype(np.float32)
    return frame


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
t1_pre *= 255.0
t1_post *= 255.0
t2 *= 255.0
t1w *= 255.0
t2w *= 255.0
pd *= 255.0

t1_t2 = np.zeros((RO, E1, 6, N))
t1_t2[:, :, 0, :] = np.reshape(t1w, (RO, E1, N))
t1_t2[:, :, 1, :] = np.reshape(t2w, (RO, E1, N))
t1_t2[:, :, 2, :] = np.reshape(pd, (RO, E1, N))
t1_t2[:, :, 3, :] = np.reshape(t1_pre, (RO, E1, N))
t1_t2[:, :, 4, :] = np.reshape(t1_post, (RO, E1, N))
t1_t2[:, :, 5, :] = np.reshape(t2, (RO, E1, N))

CHA = t1_t2.shape[2]

if (plot_flag):
    plot_image_array(t1_t2.reshape((RO, E1, CHA * N)), columns=N, figsize=[16, 8], cmap='gray')

# ------------------------------------------
# cut center FOV

data_used = np.zeros((FOV_HIGHRES, FOV_HIGHRES, CHA, N))

for n in range(N):
    a_data_used, s_ro, s_e1 = cpad_2d(t1_t2[:, :, :, n], FOV_HIGHRES, FOV_HIGHRES)
    data_used[:, :, :, n] = a_data_used

# upsample by x2

data_used_high_res = np.zeros((2 * FOV_HIGHRES, 2 * FOV_HIGHRES, CHA, N), dtype=np.float32)

# for n in range(N):
#     for m in range(2):
#         #data_used_high_res[:,:,m,n] = cv2.resize(data_used[:,:, m,n], dsize=(2*FOV_HIGHRES, 2*FOV_HIGHRES), interpolation=cv2.INTER_CUBIC)
#         data_used_high_res[:,:,m,n] = skimage.transform.rescale(data_used[:,:, m,n], 2, order=3, multichannel=False)


for n in range(N):
    data_used_high_res[:, :, :, n] = skimage.transform.rescale(data_used[:, :, :, n], 2, order=3, multichannel=True)

if (plot_flag):
    plot_image_array(data_used_high_res.reshape((2 * FOV_HIGHRES, 2 * FOV_HIGHRES, CHA * N)), columns=N,
                     figsize=[16, 8], cmap='gray')

# ------------------------------------------
# call the model
# convert data to [N M RO E1]

im = np.transpose(data_used_high_res, (3, 2, 0, 1))

print("t1t2 segmentation im is ", im.shape)

input_name = model.get_inputs()[0].name
output_name = model.get_outputs()[0].name
x = im.astype(np.float32)
result = model.run([output_name], {input_name: x})
pred_batch = result[0]
print("t1t2 segmentation scores is ", pred_batch.shape, file=sys.stderr)