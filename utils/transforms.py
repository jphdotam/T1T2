import albumentations as A

def get_segmentation_transforms(cfg, normalise=True):
    img_height, img_width = cfg['data']['image_dim'][0], cfg['data']['image_dim'][1]
    assert img_height, img_width
    img_size = img_height

    mean, std = cfg['data']['mean'], cfg['data']['std']
    cfgtr = cfg['transforms']['train']

    train_transforms = [A.LongestMaxSize(img_size), A.PadIfNeeded(min_height=img_height, min_width=img_width)]
    test_transforms = [A.LongestMaxSize(img_size), A.PadIfNeeded(min_height=img_height, min_width=img_width)]

    if 'randomresizedcrop' in cfgtr:
        train_transforms.append(A.RandomResizedCrop(img_height, img_width, scale=(0.3, 1.0)))
    if 'hflip' in cfgtr:
        train_transforms.append(A.HorizontalFlip())
    if 'vflip' in cfgtr:
        train_transforms.append(A.VerticalFlip())
    if 'brightnesscontrast' in cfgtr:
        train_transforms.append(A.RandomBrightnessContrast())
    if 'griddropout' in cfgtr:
        train_transforms.append(A.GridDropout(fill_value=255, mask_fill_value=0))
    if 'maskdropout' in cfgtr:
        train_transforms.append(A.MaskDropout(max_objects=1, image_fill_value=255, mask_fill_value=0, p=0.25))

    if normalise:
        train_transforms.append(A.Normalize(mean=mean, std=std))
        test_transforms.append(A.Normalize(mean=mean, std=std))
    else:
        print(f"WARNING: NOT NORMALISING! Assume you are checking normalisation parameters?")

    train_transforms = A.Compose(train_transforms)
    test_transforms = A.Compose(test_transforms)

    return train_transforms, test_transforms
