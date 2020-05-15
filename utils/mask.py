import numpy as np
import PIL.Image
import PIL.ImageDraw

def shape_to_mask(img_shape, i_label, points):
    """Thanks to https://github.com/wkentaro/labelme/blob/master/labelme/utils/shape.py"""
    mask = np.zeros(img_shape, dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    draw.polygon(xy=xy, outline=1, fill=i_label)
    mask = np.array(mask, dtype=bool)
    return mask
