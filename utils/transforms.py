import cv2
import albumentations as A

def get_segmentation_transforms(cfg, normalise=True):
    img_height, img_width = cfg['data']['image_dim'][0], cfg['data']['image_dim'][1]
    assert img_height, img_width
    img_size = img_height

    mean, std = cfg['data']['mean'], cfg['data']['std']
    cfgtr = cfg['transforms']['train']

    transform_method = cfg['transforms']['method']

    if transform_method == 'longestresize_then_pad':

        interp = cfg['transforms'].get('interp', 'inter_linear')
        if interp == 'inter_linear':
            interp = cv2.INTER_LINEAR
        elif interp == 'inter_cubic':
            interp = cv2.INTER_CUBIC

        train_transforms = [A.LongestMaxSize(img_size, interpolation=interp), A.PadIfNeeded(min_height=img_height, min_width=img_width)]
        test_transforms = [A.LongestMaxSize(img_size, interpolation=interp), A.PadIfNeeded(min_height=img_height, min_width=img_width)]
    elif transform_method == 'pad':
        if cfg['data'].get('high_res', False):
            print(f"!!!WARNING: In high res mode, surely want to resize, not pad to size!!!")
        train_transforms = [A.PadIfNeeded(min_height=img_size, min_width=img_size)]
        test_transforms = [A.PadIfNeeded(min_height=img_size, min_width=img_size)]
    else:
        raise ValueError()

    if cfgtr.get('randomresizedcrop', False):
        train_transforms.append(A.RandomResizedCrop(img_height, img_width, scale=(0.3, 1.0)))
    if cfgtr.get('randomcrop', False):
        crop_height, crop_width = cfgtr['randomcrop']
        train_transforms.append(A.RandomCrop(crop_height, crop_width))
    if cfgtr.get('rotate', False):
        train_transforms.append(A.Rotate(90))
    if cfgtr.get('hflip', False):
        train_transforms.append(A.HorizontalFlip())
    if cfgtr.get('vflip', False):
        train_transforms.append(A.VerticalFlip())
    if cfgtr.get('brightnesscontrast', False):
        train_transforms.append(A.RandomBrightnessContrast())
    if cfgtr.get('griddropout', False):
        train_transforms.append(A.GridDropout(fill_value=255, mask_fill_value=0))
    if cfgtr.get('maskdropout', False):
        train_transforms.append(A.MaskDropout(max_objects=1, image_fill_value=255, mask_fill_value=0, p=0.25))

    if normalise:
        train_transforms.append(A.Normalize(mean=mean, std=std))
        test_transforms.append(A.Normalize(mean=mean, std=std))
    else:
        print(f"WARNING: NOT NORMALISING! Assume you are checking normalisation parameters?")

    train_transforms = A.Compose(train_transforms)
    test_transforms = A.Compose(test_transforms)

    return train_transforms, test_transforms
