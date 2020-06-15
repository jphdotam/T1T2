import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from collections import OrderedDict

import torch
import torch.nn.functional as F

try:
    import rasterio.features
    import shapely.geometry
except ImportError:
    print(f"rasterio or shapely not installed; cannot generate polygons")


# def get_image(t, mean, std):
#     """Imshow for Tensor."""
#     img = t.numpy().transpose((1, 2, 0))
#     img = std * img + mean
#     img = np.clip(img, 0, 1)
#     return img


def get_polygons(y, channel):
    shape_gen = rasterio.features.shapes(y.astype(np.float32))  # 0=background, 1=cav, 2=wall
    polygons = [shapely.geometry.Polygon(shape[0]["coordinates"][0]) for shape in shape_gen if shape[1] == channel]
    return polygons

def vis_seg(dataloader, model, epoch, output_dir, cfg, show=False, writer=None, save=False):
    device = cfg['training']['device']
    if epoch % cfg['output']['vis_every']:
        return

    generate_polygons = cfg['output'].get('generate_polygons', False)
    n_classes = cfg['data']['n_classes']
    mean, std = cfg['data']['mean'], cfg['data']['std']
    n_rows = cfg['output']['vis_n']
    n_cols = 4

    cmap = matplotlib.colors.ListedColormap(['black', 'gray', 'green', 'yellow', 'orange', 'red'])

    batch_x, batch_y_true = next(iter(dataloader))  # Last batch will have more pathology than 1st as highest grade
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 40), gridspec_kw={'wspace': 0, 'hspace': 0})
    with torch.no_grad():
        batch_y_pred = model(batch_x.to(device))
        if type(batch_y_pred) == OrderedDict:
            batch_y_pred = batch_y_pred['out']
        ph, pw = batch_y_pred.size(2), batch_y_pred.size(3)
        h, w = batch_y_true.size(1), batch_y_true.size(2)
        if ph != h or pw != w:
            batch_y_pred = F.upsample(
                    input=batch_y_pred, size=(h, w), mode='bilinear')

    for i, (frame, y_true, y_pred) in enumerate(zip(batch_x, batch_y_true, batch_y_pred)):

        frame_t1 = frame[0] * std[1] + mean[1]
        frame_t2 = frame[1] * std[2] + mean[2]

        true_np = y_true.numpy()
        pred_np = np.argmax(y_pred.cpu().numpy(), axis=0)

        axes[i][0].imshow(frame_t1, cmap='gray')
        axes[i][1].imshow(frame_t2, cmap='gray')
        axes[i][2].imshow(true_np, vmin=0, vmax=n_classes - 1, cmap=cmap)
        axes[i][3].imshow(pred_np, vmin=0, vmax=n_classes - 1, cmap=cmap)
        axes[i][0].axis('off')
        axes[i][1].axis('off')
        axes[i][2].axis('off')
        axes[i][3].axis('off')

        if generate_polygons:
            for data, linestyle in zip((y_true, torch.argmax(y_pred, dim=0)), (':', '-')):
                for channel, colour in zip((1, 2), ('r', 'y')):  # LV cav exterior & wall exterior
                    polys = get_polygons(data.cpu().numpy(), channel=channel)
                    style = f"{colour}{linestyle}"
                    for poly in polys:
                        ext_x, ext_y = poly.exterior.coords.xy
                        axes[i][0].plot(ext_x, ext_y, style, alpha=0.5)
                        axes[i][1].plot(ext_x, ext_y, style, alpha=0.5)

        if i >= n_rows - 1:
            break

    if show:
        fig.show()
    if save:
        plt.savefig(os.path.join(output_dir, f"{epoch:03d}.png"))
    if writer:
        writer.add_figure(f"Epoch {epoch:03d}", fig, epoch)
