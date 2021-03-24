import os
import json
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from lib.plots import mean_diff_plot
from lib.vis import iou, dice
from scipy.stats import wilcoxon, spearmanr
from sklearn.metrics import r2_score

# PATH_AI = "./data/predictions_all_030_tta"
# PATH_AI = "./data/predictions_all_035_tta"
PATH_AI = "./data/predictions_all_036_mini_tta"
# PATH_AI = "./data/predictions_all_034_mini_tta"
PATH_JH = "./data/predictions_james"
# PATH_JH = "./data/predictions_all_030_tta"
PATH_HX = "./data/predictions_hui"
# PATH_JH = "./data/predictions_hui"

MEAN_OR_MEDIAN = 'median'

with open(os.path.join(PATH_AI, 'segments.json')) as f:
    preds_ai = json.load(f)
with open(os.path.join(PATH_JH, 'segments.json')) as f:
    preds_jh = json.load(f)
with open(os.path.join(PATH_HX, 'segments.json')) as f:
    preds_hx = json.load(f)

for study_id, study_dict in preds_ai.copy().items():
    for segment_id, segment_dict in study_dict.items():
        if np.isnan(segment_dict['mean']):
            print(f"Segmentation fail for {study_id}")
            del preds_ai[study_id]
            break

studies_valid = list(set(preds_ai.keys()).intersection(preds_jh.keys(), preds_hx.keys()))
# studies_valid = list(set(preds_ai.keys()).intersection(preds_hx.keys()))
print(f"Found {len(studies_valid)} across the 3")

stacks = list(set([s.split('__')[0] for s in studies_valid]))
print(f"Found {len(stacks)} stacks")

patients = list(set([s.split('_')[2] for s in stacks]))
print(f"Found {len(patients)} patients")

###################### mean

######
# t1 #
######

ai_t1, jh_t1, hx_t1 = [], [], []
for study in studies_valid:
    for preds, list_to in zip((preds_ai, preds_jh, preds_hx), (ai_t1, jh_t1, hx_t1)):
        # for preds, list_to in zip((preds_ai, preds_hx), (ai_t1, hx_t1)):
        for segment_id, segment_dict in preds[study].items():
            if 't1' in segment_id:
                list_to.append(segment_dict[MEAN_OR_MEDIAN])


ai_t1 = np.array(ai_t1)
jh_t1 = np.array(jh_t1)
hx_t1 = np.array(hx_t1)

ai_t1_native = np.array([v for v in ai_t1 if v >= 820])
ai_t1_gad = np.array([v for v in ai_t1 if v < 820])

ai_t2, jh_t2, hx_t2 = [], [], []
for study in studies_valid:
    for preds, list_to in zip((preds_ai, preds_jh, preds_hx), (ai_t2, jh_t2, hx_t2)):
        # for preds, list_to in zip((preds_ai, preds_hx), (ai_t2, hx_t2)):
        for segment_id, segment_dict in preds[study].items():
            if 't2' in segment_id:
                list_to.append(segment_dict[MEAN_OR_MEDIAN])

ai_t2 = np.array(ai_t2)
jh_t2 = np.array(jh_t2)
hx_t2 = np.array(hx_t2)

mae_jh_ai_t1, mae_hx_ai_t1, mae_jh_hx_t1, mae_jh_ai_t2, mae_hx_ai_t2, mae_jh_hx_t2 = [], [], [], [], [], []
for jt1, jt2, ht1, ht2, at1, at2 in zip(jh_t1, jh_t2, hx_t1, hx_t2, ai_t1, ai_t2):
    mae_jh_ai_t1.append(abs(jt1 - at1))
    mae_hx_ai_t1.append(abs(ht1 - at1))
    mae_jh_hx_t1.append(abs(jt1 - ht1))
    mae_jh_ai_t2.append(abs(jt2 - at2))
    mae_hx_ai_t2.append(abs(ht2 - at2))
    mae_jh_hx_t2.append(abs(jt2 - ht2))

spearman_jh_ai_t1 = spearmanr(jh_t1, ai_t1).correlation
spearman_hx_ai_t1 = spearmanr(hx_t1, ai_t1).correlation
spearman_jh_hx_t1 = spearmanr(jh_t1, hx_t1).correlation
spearman_jh_ai_t2 = spearmanr(jh_t2, ai_t2).correlation
spearman_hx_ai_t2 = spearmanr(hx_t2, ai_t2).correlation
spearman_jh_hx_t2 = spearmanr(jh_t2, hx_t2).correlation

r2_jh_ai_t1 = r2_score(jh_t1, ai_t1)
r2_hx_ai_t1 = r2_score(hx_t1, ai_t1)
r2_jh_hx_t1 = r2_score(jh_t1, hx_t1)
r2_jh_ai_t2 = r2_score(jh_t2, ai_t2)
r2_hx_ai_t2 = r2_score(hx_t2, ai_t2)
r2_jh_hx_t2 = r2_score(jh_t2, hx_t2)

mae_jh_ai_t1 = np.mean(mae_jh_ai_t1)
mae_hx_ai_t1 = np.mean(mae_hx_ai_t1)
mae_jh_hx_t1 = np.mean(mae_jh_hx_t1)
mae_jh_ai_t2 = np.mean(mae_jh_ai_t2)
mae_hx_ai_t2 = np.mean(mae_hx_ai_t2)
mae_jh_hx_t2 = np.mean(mae_jh_hx_t2)


def add_jitter(values, max_jitter):
    return np.array([v + np.random.uniform(-max_jitter, max_jitter, 1)[0] for v in values])


fig_t1, axes_t1 = plt.subplots(2, 3, sharey='row', sharex='col', figsize=(15, 6))
for i, (ax, title) in enumerate(
        zip(axes_t1[0], ('\nAI versus Expert 1', '\nAI versus Expert 2', '\nExpert 1 versus Expert 2'))):
    ax.set_title(title, fontsize=16)
    if i > 0:
        ax.axes.get_yaxis().set_visible(False)

    ax.yaxis.set_ticks(np.arange(0, 2000, 400))
    ax.xaxis.set_ticks(np.arange(0, 2000, 400))

for i, ax in enumerate(axes_t1[1]):
    if i > 0:
        ax.axes.get_yaxis().set_visible(False)

    ax.xaxis.set_ticks(np.arange(0, 2000, 400))
    ax.yaxis.set_ticks(np.arange(-800, 1800, 400))

axes_t1[0, 0].scatter(x=jh_t1, y=ai_t1, alpha=0.1)
axes_t1[0, 1].scatter(x=hx_t1, y=ai_t1, alpha=0.1)
axes_t1[0, 2].scatter(x=jh_t1, y=hx_t1, alpha=0.1)

axes_t1[0, 0].plot([0, 2000], [0, 2000], 'k--')
axes_t1[0, 1].plot([0, 2000], [0, 2000], 'k--')
axes_t1[0, 2].plot([0, 2000], [0, 2000], 'k--')

mean_diff_plot(ai_t1, jh_t1, ax=axes_t1[1, 0], scatter_kwds={'alpha': 0.1}, mae=np.round(mae_jh_ai_t1, 1))
mean_diff_plot(ai_t1, hx_t1, ax=axes_t1[1, 1], scatter_kwds={'alpha': 0.1}, mae=np.round(mae_hx_ai_t1, 1))
mean_diff_plot(jh_t1, hx_t1, ax=axes_t1[1, 2], scatter_kwds={'alpha': 0.1}, mae=np.round(mae_jh_hx_t1, 1))

for ax in axes_t1[0]:
    ax.axvline(x=850, linestyle='--')
for ax in axes_t1[1]:
    ax.axvline(x=850, linestyle='--')

plt.setp(axes_t1, xlim=(0, 1800))
plt.setp(axes_t1[0], ylim=(0, 1700))
plt.setp(axes_t1[1], ylim=(-900, 900))
plt.subplots_adjust(wspace=0.1, hspace=0.1, left=0.075)
fig_t1.suptitle('T1 (ms) - Scatter and Bland-Altman plots comparing AI & expert measurements\n', fontsize=20)

######
# T2 #
######


fig_t2, axes_t2 = plt.subplots(2, 3, sharey='row', figsize=(15, 6))
for i, (ax, title) in enumerate(zip(axes_t2[0], ('\nAI versus E1', '\nAI versus E2', '\nE1 versus E2'))):
    ax.set_title(title, fontsize=16)
    if i > 0:
        ax.axes.get_yaxis().set_visible(False)

    ax.xaxis.set_ticks(np.arange(0, 100, 20))
    ax.yaxis.set_ticks(np.arange(0, 100, 20))

for i, ax in enumerate(axes_t2[1]):
    if i > 0:
        ax.axes.get_yaxis().set_visible(False)

    ax.xaxis.set_ticks(np.arange(0, 100, 20))
    ax.yaxis.set_ticks(np.arange(-40, 60, 20))

axes_t2[0, 0].scatter(x=jh_t2, y=ai_t2, alpha=0.1)
axes_t2[0, 1].scatter(x=hx_t2, y=ai_t2, alpha=0.1)
axes_t2[0, 2].scatter(x=jh_t2, y=hx_t2, alpha=0.1)

axes_t2[0, 0].plot([0, 100], [0, 100], 'k--')
axes_t2[0, 1].plot([0, 100], [0, 100], 'k--')
axes_t2[0, 2].plot([0, 100], [0, 100], 'k--')

# mean_diff_plot(add_jitter(ai_t2, 0.5), add_jitter(jh_t2, 0.5), ax=axes[1, 0], scatter_kwds={'alpha': 0.05})
# mean_diff_plot(add_jitter(ai_t2, 0.5), add_jitter(hx_t2, 0.5), ax=axes[1, 1], scatter_kwds={'alpha': 0.05})
# mean_diff_plot(add_jitter(jh_t2, 0.5), add_jitter(hx_t2, 0.5), ax=axes[1, 2], scatter_kwds={'alpha': 0.05})
mean_diff_plot(ai_t2, jh_t2, ax=axes_t2[1, 0], scatter_kwds={'alpha': 0.05}, r2=r2_jh_ai_t1, rho=spearman_jh_ai_t1,
               mae=np.round(mae_jh_ai_t2, 2))
mean_diff_plot(ai_t2, hx_t2, ax=axes_t2[1, 1], scatter_kwds={'alpha': 0.05}, mae=np.round(mae_hx_ai_t2, 2))
mean_diff_plot(jh_t2, hx_t2, ax=axes_t2[1, 2], scatter_kwds={'alpha': 0.05}, mae=np.round(mae_jh_hx_t2, 2))
plt.setp(axes_t2, xlim=(30, 80))
plt.setp(axes_t2[0], ylim=(30, 80))
plt.setp(axes_t2[1], ylim=(-25, 25))
plt.subplots_adjust(wspace=0.1, hspace=0.1, left=0.075)
fig_t2.suptitle('T2 (ms) - Scatter and Bland-Altman plot comparing AI & expert measurements\n', fontsize=20)

iou_ai_jh, iou_ai_hx, iou_jh_hx = [], [], []
dice_ai_jh, dice_ai_hx, dice_jh_hx = [], [], []
for study in tqdm(studies_valid):
    ai_seg = skimage.io.imread(os.path.join(PATH_AI, study + '.png'))
    jh_seg = skimage.io.imread(os.path.join(PATH_JH, study + '.png'))
    hx_seg = skimage.io.imread(os.path.join(PATH_HX, study + '.png'))

    iou_ai_jh.append(iou(ai_seg, jh_seg))
    iou_ai_hx.append(iou(ai_seg, hx_seg))
    iou_jh_hx.append(iou(jh_seg, hx_seg))
    dice_ai_jh.append(dice(ai_seg, jh_seg))
    dice_ai_hx.append(dice(ai_seg, hx_seg))
    dice_jh_hx.append(dice(jh_seg, hx_seg))

axes_t1[0, 0].annotate(f"ρ {spearman_jh_ai_t1:.4f}\nR² {r2_jh_ai_t1:.4f}", xy=(0.99, 0.6), horizontalalignment='right', verticalalignment='center', fontsize=10, xycoords='axes fraction')
axes_t1[0, 1].annotate(f"ρ {spearman_hx_ai_t1:.4f}\nR² {r2_hx_ai_t1:.4f}", xy=(0.99, 0.6), horizontalalignment='right', verticalalignment='center', fontsize=10, xycoords='axes fraction')
axes_t1[0, 2].annotate(f"ρ {spearman_jh_hx_t1:.4f}\nR² {r2_jh_hx_t1:.4f}", xy=(0.99, 0.6), horizontalalignment='right', verticalalignment='center', fontsize=10, xycoords='axes fraction')

axes_t2[0, 0].annotate(f"ρ {spearman_jh_ai_t2:.4f}\nR² {r2_jh_ai_t2:.4f}", xy=(0.99, 0.6), horizontalalignment='right', verticalalignment='center', fontsize=10, xycoords='axes fraction')
axes_t2[0, 1].annotate(f"ρ {spearman_hx_ai_t2:.4f}\nR² {r2_hx_ai_t2:.4f}", xy=(0.99, 0.6), horizontalalignment='right', verticalalignment='center', fontsize=10, xycoords='axes fraction')
axes_t2[0, 2].annotate(f"ρ {spearman_jh_hx_t2:.4f}\nR² {r2_jh_hx_t2:.4f}", xy=(0.99, 0.6), horizontalalignment='right', verticalalignment='center', fontsize=10, xycoords='axes fraction')

fig_t1.show()
fig_t2.show()

# Outliers
deltas_t1, deltas_t2 = defaultdict(list), defaultdict(list)
for study in studies_valid:
        preds_study = preds_ai[study]
        for segment_id, segment_dict in preds_study.items():
            if 't1' in segment_id:
                delta = preds_study[segment_id][MEAN_OR_MEDIAN] - np.mean([preds_jh[study][segment_id][MEAN_OR_MEDIAN],
                                                                          preds_hx[study][segment_id][MEAN_OR_MEDIAN]])
                deltas_t1[study].append(delta)
            elif 't2' in segment_id:
                delta = preds_study[segment_id][MEAN_OR_MEDIAN] - np.mean(
                    [preds_jh[study][segment_id][MEAN_OR_MEDIAN],
                     preds_hx[study][segment_id][MEAN_OR_MEDIAN]])
                deltas_t2[study].append(delta)

# for study, ai1, jh1, hx1, ai2, jh2, hx2 in zip(studies_valid, ai_t1, jh_t1, hx_t1, ai_t2, jh_t2, hx_t2):
#     deltas_t1[study].append(ai1 - np.mean([jh1, hx1]))
#     deltas_t2[study].append(ai2 - np.mean([jh2, hx2]))

deltas_t1 = dict(sorted(deltas_t1.items(), key=lambda item: np.mean(np.abs(item[1])), reverse=True))
deltas_t2 = dict(sorted(deltas_t2.items(), key=lambda item: np.mean(np.abs(item[1])), reverse=True))

print(list(deltas_t2.keys())[:10])