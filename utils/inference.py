import math
import torch
import numpy as np
import matplotlib.pyplot as plt

from utils.labeling import dicom_to_img


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


def pose_mask_to_coords(prediction_mask,  # a single channel, H * W
                        default_relative_step_size=1/50,
                        minimum_probability_to_trace=0.0001,
                        inertia=0.5,
                        minimum_bend_cosine=-0.2,
                        far_enough_fraction=0.5):  # How far away each ridge point step must be from the closest other points (to prevent bendback). Don't drop below 0.5

    def clamp_x(x):
        if type(x) == torch.Tensor:
            x = x.cpu().numpy()
        return int(np.clip(x, 0, img_width - 1))

    def clamp_y(y):
        if type(y) == torch.Tensor:
            y = y.cpu().numpy()
        return int(np.clip(y, 0, img_height - 1))

    def distance_from_point(proposed_x, proposed_y, point):
        return (proposed_x - point["x"]) ** 2 + (proposed_y - point["y"]) ** 2

    def distance_from_nearest_other_point(proposed_coord, two_lists_of_points):
        """Returns (in network pixels) the distance between the proposed coord and the closes existing ridge point"""

        proposed_x = clamp_x(proposed_coord['x'])
        proposed_y = clamp_y(proposed_coord['y'])

        unified_list_of_points = two_lists_of_points[0] + two_lists_of_points[1]

        # L2 distance between the proposed point and each of the existing ridge traced points in PNG units
        r2 = [distance_from_point(proposed_x, proposed_y, curve_point) for curve_point in unified_list_of_points]

        r_pixels = np.sqrt(np.array(r2))
        return min(r_pixels) if len(r_pixels) >= 1 else None

    img_height, img_width = prediction_mask.shape
    step_size = max(int(np.max([img_height, img_width]) * default_relative_step_size), 1)

    if np.max(prediction_mask) >= minimum_probability_to_trace:
        # Pick the brightest starting point and go in both directions
        start_cursor_y, start_cursor_x = np.unravel_index(np.argmax(prediction_mask), prediction_mask.shape)
        n_directions = 2

        last_theta_deg = None
        first_direction_first_step_theta_deg = None
        points_in_direction = [[], []]
        points_in_direction[0].append({
            "x": clamp_x(start_cursor_x),
            "y": clamp_y(start_cursor_y),
            "z": prediction_mask[clamp_y(start_cursor_y), clamp_x(start_cursor_x)],
        })

        for direction in range(n_directions):
            cursor_y, cursor_x = start_cursor_y, start_cursor_x

            if direction != 0:  # 2nd direction, so use the opposite of the angle used from the first step of the 1st direction
                if first_direction_first_step_theta_deg is not None:
                    last_theta_deg = 180 + first_direction_first_step_theta_deg

            # i_cursor 0 is the FIRST step AWAY FROM the central one
            for i_cursor in range(int(4 / default_relative_step_size)):  # arbitrarily set the maximum curve length to be 4 times the long side of the image
                best_theta_deg = None
                best_bend_cosine = None
                best_z = float("-inf")
                for theta_deg in range(360):
                    new_x = cursor_x + step_size * math.cos((math.pi / 180) * theta_deg)
                    new_y = cursor_y + step_size * math.sin((math.pi / 180) * theta_deg)
                    pure_z = prediction_mask[clamp_y(new_y), clamp_x(new_x)]
                    if direction == 0 and i_cursor == 0:
                        bend_cosine = 1
                    elif last_theta_deg is not None:
                        bend_cosine = math.cos(
                            (theta_deg - last_theta_deg) * math.pi / 180)
                    # penalise if going back on self
                    angle_weighting = np.maximum(0, 1 + inertia * bend_cosine)
                    z = pure_z * angle_weighting
                    if theta_deg == 0 or z > best_z:
                        best_theta_deg = theta_deg
                        best_z = z
                        best_new_x = new_x
                        best_new_y = new_y
                        best_bend_cosine = bend_cosine
                cursor_x, cursor_y = best_new_x, best_new_y
                last_theta_deg = best_theta_deg
                # save the first step of the first direction to allow reverse traversal later
                if direction == 0 and i_cursor == 0:
                    first_direction_first_step_theta_deg = best_theta_deg

                # The central one, and the first (i_cursor==0) in each direction, are the middle three
                probability_enough = prediction_mask[clamp_y(cursor_y), clamp_x(cursor_x)] > minimum_probability_to_trace
                straight_enough = best_bend_cosine > minimum_bend_cosine
                far_enough_away = distance_from_nearest_other_point({"x": cursor_x, "y": cursor_y},
                                                                    points_in_direction) > far_enough_fraction * step_size

                need_this_point_to_get_to_minimum_3 = i_cursor == 0 and \
                                                      ((direction == 0 and i_cursor == 0) or
                                                       (direction == 1 and len(points_in_direction[0]) <= 1))

                if need_this_point_to_get_to_minimum_3 or (probability_enough and straight_enough):
                    if far_enough_away:
                        points_in_direction[direction].append({
                            "x": clamp_x(cursor_x),
                            "y": clamp_y(cursor_y),
                            "z": prediction_mask[clamp_y(cursor_y), clamp_x(cursor_x)]
                        })
                    else:
                        # Found a point, but close to another one
                        # If direction 0 and we've got back on ourselves, we've managed to trace the surface in 1 run
                        if direction == 0 and distance_from_point(cursor_x, cursor_y, points_in_direction[0][0]):
                            points_in_direction[direction].append({
                                "x": points_in_direction[0][0]['x'],
                                "y": points_in_direction[0][0]['y'],
                                "z": prediction_mask[clamp_y(cursor_y), clamp_x(cursor_x)]
                            })
                            return points_in_direction[0]

                else:
                    print(f"Breaking {need_this_point_to_get_to_minimum_3} {probability_enough} {straight_enough}")
                    break

        # After done both directions, assemble a single long curve
        points_in_whole_curve = points_in_direction[1][::-1] + points_in_direction[0]
        return points_in_whole_curve