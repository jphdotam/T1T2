import os
import numpy as np
import skimage.io
import skimage.transform
from scipy import interpolate
from utils.labeling import load_pickle
from utils.windows import window_numpy


def get_radius_matrix(image_dimensions):
    img_height, img_width = image_dimensions[:2]
    radius_matrix_width = img_width * 2 + 10  # made up extra size just in case
    radius_matrix_height = img_height * 2 + 10
    radius_matrix = np.zeros((radius_matrix_height, radius_matrix_width))
    for x in range(img_width):
        for y in range(img_height):
            dist = np.sqrt(x * x + y * y)
            radius_matrix[img_height + y, img_width + x] = dist
            radius_matrix[img_height + y, img_width - x] = dist
            radius_matrix[img_height - y, img_width + x] = dist
            radius_matrix[img_height - y, img_width - x] = dist
    return radius_matrix


def pad_if_needed(img, min_height, min_width, centre_pad):
    input_height, input_width = img.shape[:2]
    new_shape = list(img.shape)
    new_shape[0] = max(input_height, min_height)
    new_shape[1] = max(input_width, min_width)
    out = np.zeros(new_shape, dtype=img.dtype)
    if centre_pad:
        row_from, col_from = 0, 0
        if input_height < min_height:
            row_from = (min_height - input_height) // 2
        if input_width < min_width:
            col_from = (min_width - input_width) // 2
        out[row_from:row_from + input_height, col_from:col_from + input_width] = img
    else:
        out[:input_height, :input_width] = img
    return out


def calc_gauss_on_a_scalar_or_matrix(distance, sigma):
    # partly gaussian but partly a hyperbola or something, so there is some gradient, however feeble, *everwhere*
    return 0.8 * np.exp(-(distance ** 2) / (2 * sigma ** 2)) + 0.2 * (1 / (1 + distance))


def export_label(labelpath, sequences, label_classes, output_dir, gaussian_sigma):
    """Typical labelpath:

    E:/Data/T1T2_peter/
        20200313/
        T1T2_42363_588453382_588453387_1639_20200313-105857/
        T1_T2_PD_SLC0_CON0_PHS0_REP0_SET0_AVE0_1.npy.pickle

    if windows is not None, the dictionary will be used to normalise each image between 0 and 255

    """
    seq_path = os.path.splitext(labelpath)[0]
    date_dir = os.path.basename(os.path.dirname(os.path.dirname(labelpath)))  # 20200313
    study_dir = os.path.basename(os.path.dirname(labelpath))  # T1T2_42363_588453382_588453387_1639_20200313-105857
    npy_name = os.path.splitext(os.path.basename(seq_path))[0]  # T1_T2_PD_SLC0_CON0_PHS0_REP0_SET0_AVE0_1.npy

    npy = np.load(seq_path)
    src_height, src_width, src_channels = npy.shape

    # Image
    assert len(sequences) <= 4, "Number of sequences must be <= 4 to save as PNG"
    out_channels_img = 3 if len(sequences) <= 3 else 4  # Don't use alpha channel unless needed
    seq_out = np.zeros((src_height, src_width, out_channels_img), dtype=np.uint8)

    for i_seq_out, (seq_name, seq_dict) in enumerate(sequences.items()):
        source_channel = seq_dict['channel']
        wc = seq_dict['wc']
        ww = seq_dict['ww']

        seq = npy[:, :, source_channel]
        seq = window_numpy(seq, wc, ww, rescale_255=True)
        seq_out[:, :, i_seq_out] = seq

    # Label
    label = load_pickle(labelpath)
    if any(label_class not in label for label_class in label_classes):
        print(f"Labels missing for study {labelpath} (only {label.keys()} present)")
        return None
    radius_matrix = get_radius_matrix((src_height, src_width))

    assert len(label_classes) <= 4, "Number of output classes must be <= 4 to save as PNG"
    out_channels_lab = 3 if len(label_classes) <= 3 else 4  # Don't use alpha channel unless needed
    lab_out = np.zeros((src_height, src_width, out_channels_lab), dtype=np.float32)

    for i_lab_out, lab_name in enumerate(label_classes):
        points_labelled = [tuple((c[0], src_height-c[1])) for c in label[lab_name]]
        points_labelled.append(points_labelled[0])  # Add first point to end to make full loop
        points_labelled = np.array(points_labelled)

        n_points = len(points_labelled)

        points_labelled_i = np.arange(n_points)
        points_labelled_x = points_labelled[:, 0]
        points_labelled_y = points_labelled[:, 1]

        fx = interpolate.make_interp_spline(points_labelled_i, points_labelled_x, k=3, bc_type=([(2, 0)], [(2, 0)]))
        fy = interpolate.make_interp_spline(points_labelled_i, points_labelled_y, k=3, bc_type=([(2, 0)], [(2, 0)]))

        interp_t = np.linspace(0, n_points - 1, 200)
        interp_x = fx(interp_t)
        interp_y = fy(interp_t)

        multi_radius_stack = np.zeros((len(interp_x), src_height, src_width))

        for i_interp_point, (x, y) in enumerate(zip(interp_x, interp_y)):
            x = int(max(0, min(src_width - 1, x)))
            y = int(max(0, min(src_height - 1, y)))

            multi_radius_stack[i_interp_point, :, :] = radius_matrix[
                                                       src_height - int(y):2 * src_height - int(y),
                                                       src_width - int(x):2 * src_width - int(x)]
            distances_to_sausage = np.amin(multi_radius_stack, 0)
            lab_out[:, :, i_lab_out] += calc_gauss_on_a_scalar_or_matrix(distances_to_sausage, gaussian_sigma)

    # Scale label between 0 and 1 -> 255
    lab_out = lab_out - np.min(lab_out, axis=(0, 1))
    lab_out = lab_out / np.max(lab_out)
    lab_out = lab_out[:src_height, :src_width]
    lab_out = (lab_out * 255).astype(np.uint8)

    # SAVE
    # Dicom
    outpath_img = f"{date_dir}__{study_dir}__{npy_name}__img.png"
    skimage.io.imsave(os.path.join(output_dir, outpath_img), seq_out, check_contrast=False)

    # Label
    outpath_lab = f"{date_dir}__{study_dir}__{npy_name}__lab.png"
    skimage.io.imsave(os.path.join(output_dir, outpath_lab), lab_out, check_contrast=False)
