import os
import json
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
from statsmodels.graphics.agreement import mean_diff_plot
from utils.vis import iou

PATH_AI = "./data/028"
PATH_JH = "./data/T1T2_peter_test_james"
PATH_HX = "./data/T1T2_peter_test_hui"

with open(os.path.join(PATH_AI, 'segments.json')) as f:
    preds_ai = json.load(f)
with open(os.path.join(PATH_JH, 'segments.json')) as f:
    preds_jh = json.load(f)
with open(os.path.join(PATH_HX, 'segments.json')) as f:
    preds_hx = json.load(f)

for study_id, study_dict in preds_ai.copy().items():
    for segment_id, segment_dict in study_dict.items():
        if np.isnan(segment_dict['median']):
            print(f"Segmentation fail for {study_id}")
            del preds_ai[study_id]
            break

studies_valid = list(set(preds_ai.keys()).intersection(preds_jh.keys(), preds_hx.keys()))
print(f"Found {len(studies_valid)} across the 3")

patients = list(set([s.split('__')[0] for s in studies_valid]))
print(f"Found {len(patients)} patients")

###################### MEDIAN

######
# t1 #
######

ai_t1, jh_t1, hx_t1 = [], [], []
for study in studies_valid:
    for preds, list_to in zip((preds_ai, preds_jh, preds_hx), (ai_t1, jh_t1, hx_t1)):
        for segment_id, segment_dict in preds[study].items():
            if 't1' in segment_id:
                list_to.append(segment_dict['median'])

ai_t1 = np.array(ai_t1)
jh_t1 = np.array(jh_t1)
hx_t1 = np.array(hx_t1)

fig, axes = plt.subplots(1,3,sharey='row', figsize=(15,6))
for i, (ax, title) in enumerate(zip(axes, ('\nAI versus E1', '\nAI versus E2', '\nE1 versus E2'))):
    ax.set_title(title, fontsize=16)
    if i > 0:
        ax.axes.get_yaxis().set_visible(False)

    ax.xaxis.set_ticks(np.arange(0, 2000, 400))
    ax.yaxis.set_ticks(np.arange(-800, 1800, 400))

mean_diff_plot(ai_t1, jh_t1, ax=axes[0], scatter_kwds={'alpha': 0.25})
mean_diff_plot(ai_t1, hx_t1, ax=axes[1], scatter_kwds={'alpha': 0.25})
mean_diff_plot(jh_t1, hx_t1, ax=axes[2], scatter_kwds={'alpha': 0.25})

for ax in axes:
    ax.axvline(x=850, linestyle='--')

plt.setp(axes, xlim=(0, 1800))
plt.setp(axes, ylim=(-900, 900))
plt.subplots_adjust(wspace=0.1, hspace=0, left=0.075)
fig.suptitle('T1 (ms) - Bland-Altman plot comparing AI & expert measurements\n', fontsize=20)
fig.show()

######
# T2 #
######

ai_t2, jh_t2, hx_t2 = [], [], []
for study in studies_valid:
    for preds, list_to in zip((preds_ai, preds_jh, preds_hx), (ai_t2, jh_t2, hx_t2)):
        for segment_id, segment_dict in preds[study].items():
            if 't2' in segment_id:
                list_to.append(segment_dict['median'])

ai_t2 = np.array(ai_t2)
jh_t2 = np.array(jh_t2)
hx_t2 = np.array(hx_t2)

fig, axes = plt.subplots(1,3,sharey='row', figsize=(15,6))
for i, (ax, title) in enumerate(zip(axes, ('\nAI versus E1', '\nAI versus E2', '\nE1 versus E2'))):
    ax.set_title(title, fontsize=16)
    if i > 0:
        ax.axes.get_yaxis().set_visible(False)

    ax.xaxis.set_ticks(np.arange(0, 100, 20))
    ax.yaxis.set_ticks(np.arange(-40, 60, 20))

mean_diff_plot(ai_t2, jh_t2, ax=axes[0], scatter_kwds={'alpha': 0.25})
mean_diff_plot(ai_t2, hx_t2, ax=axes[1], scatter_kwds={'alpha': 0.25})
mean_diff_plot(jh_t2, hx_t2, ax=axes[2], scatter_kwds={'alpha': 0.25})
plt.setp(axes, xlim=(30, 80))
plt.setp(axes, ylim=(-25, 25))
plt.subplots_adjust(wspace=0.1, hspace=0, left=0.075)
fig.suptitle('T2 (ms) - Bland-Altman plot comparing AI & expert measurements\n', fontsize=20)
fig.show()


# ########### MEAN #############
#
# ######
# # t1 #
# ######
#
# ai_t1, jh_t1, hx_t1 = [], [], []
# for study in studies_valid:
#     for preds, list_to in zip((preds_ai, preds_jh, preds_hx), (ai_t1, jh_t1, hx_t1)):
#         for segment_id, segment_dict in preds[study].items():
#             if 't1' in segment_id:
#                 list_to.append(segment_dict['mean'])
#
# ai_t1 = np.array(ai_t1)
# jh_t1 = np.array(jh_t1)
# hx_t1 = np.array(hx_t1)
#
# fig, axes = plt.subplots(1,3,sharey='row', figsize=(12,6))
# for i, (ax, title) in enumerate(zip(axes, ('AI vs JH', 'AI vs HX', 'JH vs HX'))):
#     ax.set_title(title)
#     if i > 0:
#         ax.axes.get_yaxis().set_visible(False)
#
#
# mean_diff_plot(ai_t1, jh_t1, ax=axes[0], scatter_kwds={'alpha': 0.25})
# mean_diff_plot(ai_t1, hx_t1, ax=axes[1], scatter_kwds={'alpha': 0.25})
# mean_diff_plot(jh_t1, hx_t1, ax=axes[2], scatter_kwds={'alpha': 0.25})
# plt.setp(axes, ylim=(-400, 400))
# plt.subplots_adjust(wspace=0, hspace=0)
# fig.suptitle('T1 mean')
# fig.show()
#
# ######
# # T2 #
# ######
#
# ai_t2, jh_t2, hx_t2 = [], [], []
# for study in studies_valid:
#     for preds, list_to in zip((preds_ai, preds_jh, preds_hx), (ai_t2, jh_t2, hx_t2)):
#         for segment_id, segment_dict in preds[study].items():
#             if 't2' in segment_id:
#                 list_to.append(segment_dict['mean'])
#
# ai_t2 = np.array(ai_t2)
# jh_t2 = np.array(jh_t2)
# hx_t2 = np.array(hx_t2)
#
# fig, axes = plt.subplots(1,3,sharey='row', figsize=(12,6))
# for i, (ax, title) in enumerate(zip(axes, ('AI vs E1', 'AI vs T2', 'E1 vs E2'))):
#     ax.set_title(title)
#     if i > 0:
#         ax.axes.get_yaxis().set_visible(False)
# mean_diff_plot(ai_t2, jh_t2, ax=axes[0], scatter_kwds={'alpha': 0.25})
# mean_diff_plot(ai_t2, hx_t2, ax=axes[1], scatter_kwds={'alpha': 0.25})
# mean_diff_plot(jh_t2, hx_t2, ax=axes[2], scatter_kwds={'alpha': 0.25})
# plt.setp(axes, ylim=(-50, 50))
# plt.subplots_adjust(wspace=0, hspace=0)
# fig.suptitle('T2 mean')
# fig.show()
#
#
# ###### IOU
#
# iou_ai_jh, iou_ai_hx, iou_jh_hx = [], [], []
# for study in studies_valid:
#     ai_seg = skimage.io.imread(os.path.join(PATH_AI, study+'.png'))
#     jh_seg = skimage.io.imread(os.path.join(PATH_JH, study+'.png'))
#     hx_seg = skimage.io.imread(os.path.join(PATH_HX, study+'.png'))
#
#     iou_ai_jh.append(iou(ai_seg, jh_seg))
#     iou_ai_hx.append(iou(ai_seg, hx_seg))
#     iou_jh_hx.append(iou(jh_seg, hx_seg))
