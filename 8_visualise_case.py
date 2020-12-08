import os
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from utils.cmaps import default_cmap
from utils.windows import normalize_data
from utils.inference import center_crop, pad_if_needed

FOV = 256
PRED_DIR = r"E:\Dropbox\Work\Other projects\T1T2\data\028"
INTERESTING_CASES = [
    ('post', r"E:/Data/T1T2_peter_test_james\20200701\T1T2_42363_694572106_694572111_52_20200701-094956\T1_T2_PD_SLC0_CON0_PHS0_REP0_SET0_AVE0_1.npy"),
    # ('pre', r"E:\Data\T1T2_peter_test_james\20200701\T1T2_42363_694572106_694572111_36_20200701-091016\T1_T2_PD_SLC0_CON0_PHS0_REP0_SET0_AVE0_1.npy"),
    # ('pre', r"E:\Data\T1T2_peter_test_james\20200701\T1T2_42363_694572106_694572111_34_20200701-090923\T1_T2_PD_SLC0_CON0_PHS0_REP0_SET0_AVE0_1.npy"),
    # ('post', r"E:/Data/T1T2_peter_test_james\20200710\T1T2_42363_706042184_706042189_983_20200710-174532\T1_T2_PD_SLC4_CON0_PHS0_REP0_SET0_AVE0_5.npy"),
    # ('post', r"E:/Data/T1T2_peter_test_james\20200710\T1T2_42363_706042184_706042189_983_20200710-174532\T1_T2_PD_SLC5_CON0_PHS0_REP0_SET0_AVE0_6.npy"),
    # ('post', r"E:/Data/T1T2_peter_test_james\20200710\T1T2_42363_706042184_706042189_983_20200710-174532\T1_T2_PD_SLC6_CON0_PHS0_REP0_SET0_AVE0_7.npy"),
    # # """E:/Data/T1T2_peter\20200723\T1T2_42363_725571738_725571743_347_20200723-151929\T1_T2_PD_SLC7_CON0_PHS0_REP0_SET0_AVE0_8.npy,
    # # E:/Data/T1T2_peter\20200723\T1T2_42363_725571738_725571743_347_20200723-151929\T1_T2_PD_SLC8_CON0_PHS0_REP0_SET0_AVE0_9.npy,
    # # E:/Data/T1T2_peter\20200723\T1T2_42363_725571738_725571743_347_20200723-151929\T1_T2_PD_SLC9_CON0_PHS0_REP0_SET0_AVE0_10.npy""",
    # # """E:/Data/T1T2_peter\20200721\T1T2_42363_721065144_721065149_317_20200721-164132\T1_T2_PD_SLC2_CON0_PHS0_REP0_SET0_AVE0_3.npy,
    # # E:/Data/T1T2_peter\20200721\T1T2_42363_721065144_721065149_317_20200721-164132\T1_T2_PD_SLC3_CON0_PHS0_REP0_SET0_AVE0_4.npy,
    # # E:/Data/T1T2_peter\20200721\T1T2_42363_721065144_721065149_317_20200721-164132\T1_T2_PD_SLC6_CON0_PHS0_REP0_SET0_AVE0_7.npy
    # # """,
    # # """E:/Data/T1T2_peter\20200718\T1T2_141613_32895822_32895830_167_20200718-110722\T1_T2_PD_SLC0_CON0_PHS0_REP0_SET0_AVE0_1.npy,
    # # E:/Data/T1T2_peter\20200718\T1T2_141613_32895822_32895830_167_20200718-110722\T1_T2_PD_SLC1_CON0_PHS0_REP0_SET0_AVE0_2.npy,
    # # E:/Data/T1T2_peter\20200718\T1T2_141613_32895822_32895830_167_20200718-110722\T1_T2_PD_SLC2_CON0_PHS0_REP0_SET0_AVE0_3.npy""",
    # # """E:/Data/T1T2_peter\20200713\T1T2_141613_27292702_27292710_497_20200713-180803\T1_T2_PD_SLC2_CON0_PHS0_REP0_SET0_AVE0_3.npy,
    # # E:/Data/T1T2_peter\20200713\T1T2_141613_27292702_27292710_497_20200713-180803\T1_T2_PD_SLC1_CON0_PHS0_REP0_SET0_AVE0_2.npy,
    # # E:/Data/T1T2_peter\20200713\T1T2_141613_27292702_27292710_497_20200713-180803\T1_T2_PD_SLC0_CON0_PHS0_REP0_SET0_AVE0_1.npy"""
    # ('pre', r'E:/Data/T1T2_peter\20200713\T1T2_141613_27292702_27292710_471_20200713-174359\T1_T2_PD_SLC2_CON0_PHS0_REP0_SET0_AVE0_3.npy'),
    # ('pre', r'E:/Data/T1T2_peter\20200713\T1T2_141613_27292702_27292710_471_20200713-174359\T1_T2_PD_SLC1_CON0_PHS0_REP0_SET0_AVE0_2.npy'),
    # ('pre', r'E:/Data/T1T2_peter\20200713\T1T2_141613_27292702_27292710_471_20200713-174359\T1_T2_PD_SLC0_CON0_PHS0_REP0_SET0_AVE0_1.npy')
]


for pre_or_post, case_path in INTERESTING_CASES:
    seq_id = f"{os.path.basename(os.path.dirname(case_path))}__{os.path.basename(case_path)}"
    dcm = np.load(case_path)
    dcm, _ = center_crop(pad_if_needed(dcm, FOV, FOV), FOV, FOV)
    seg = skimage.io.imread(os.path.join(PRED_DIR, seq_id + ".png"))

    t1w, t2w, pd, t1, t2 = dcm.transpose((2, 0, 1))
    t1_pre = normalize_data(t1, window_centre=1300.0, window_width=1300.0)
    t1_post = normalize_data(t1, window_centre=500.0, window_width=1000.0)
    t2 = normalize_data(t2, window_centre=60.0, window_width=120.0)

    t1_pre_seg = t1_pre.copy()
    t1_post_seg = t1_post.copy()
    t2_seg = t2.copy()

    for i_seg in range(1, 6 + 1):
        mask_seg = seg == i_seg
        border_mask = binary_dilation(mask_seg)
        border_mask = border_mask ^ mask_seg
        t1_post_seg[mask_seg] = np.median(t1_post[mask_seg])
        t1_pre_seg[mask_seg] = np.median(t1_pre[mask_seg])
        t2_seg[seg == i_seg] = np.median(t2[seg == i_seg])
        t1_post_seg[border_mask] = 1
        t1_pre_seg[border_mask] = 1
        t2_seg[border_mask] = 1

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0, 0].imshow(t1_post if pre_or_post == 'post' else t1_pre, cmap=default_cmap)
    axes[0, 0].set_title('T1 map')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(t2, cmap=default_cmap)
    axes[0, 1].set_title('T2 map')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(t1_post_seg if pre_or_post == 'post' else t1_pre_seg, cmap=default_cmap)
    axes[1, 0].set_title('AI segmentation of T1 map')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(t2_seg, cmap=default_cmap)
    axes[1, 1].set_title('AI segmentation of T2 map')
    axes[1, 1].axis('off')

    fig.show()

