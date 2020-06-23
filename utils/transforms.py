import cv2
import albumentations as A

def get_segmentation_transforms(cfg, normalise=True):
    mean, std = cfg['data']['mean'], cfg['data']['std']

    def get_transforms(trans_cfg, normalise):
        transforms = []
        if trans_cfg.get('resize', False):
            resize_height, resize_width = trans_cfg['resize']
            transforms.append(A.Resize(resize_height, resize_width,
                                       interpolation=cv2.INTER_CUBIC,
                                       p=1))
        if p := trans_cfg.get('rotate', False):
            transforms.append(A.Rotate(90, p=p))
        if p := trans_cfg.get('shiftscalerotate', False):
            transforms.append(A.ShiftScaleRotate(p=p))
        if p := trans_cfg.get('elastictransform', False):
            transforms.append(A.ElasticTransform(p=p))
        if p := trans_cfg.get('hflip', False):
            transforms.append(A.HorizontalFlip(p=p))
        if p := trans_cfg.get('vflip', False):
            transforms.append(A.VerticalFlip(p=p))
        if p := trans_cfg.get('brightnesscontrast', False):
            transforms.append(A.RandomBrightnessContrast(p=p))
        if p := trans_cfg.get('griddropout', False):
            transforms.append(A.GridDropout(fill_value=0, mask_fill_value=0, p=p))
        if p := trans_cfg.get('blur', False):
            transforms.append(A.OneOf([A.MedianBlur(blur_limit=5, p=p),
                                       A.Blur(blur_limit=5, p=p)]))
        if p := trans_cfg.get('noise', False):
            transforms.append(A.OneOf([A.GaussNoise(p=p),
                                       A.MultiplicativeNoise(p=p)]))
        if p := trans_cfg.get('hsv', False):
            transforms.append(A.HueSaturationValue(p=p))

        # Do these last so are less likely to need to reflect etc. during skew/rotation
        if trans_cfg.get('centrecrop', False):
            crop_height, crop_width = trans_cfg['centrecrop']
            transforms.append(A.PadIfNeeded(crop_height, crop_width))
            transforms.append(A.CenterCrop(crop_height, crop_width))
        if trans_cfg.get('randomresizedcrop', False):
            resize_height, resize_width = trans_cfg['randomresizedcrop']
            transforms.append(A.RandomResizedCrop(resize_height, resize_width,
                                                  scale=(0.5, 1.0),
                                                  ratio=(0.9, 1.1),
                                                  interpolation=cv2.INTER_CUBIC,
                                                  p=1))
        if trans_cfg.get('randomcrop', False):
            crop_height, crop_width = trans_cfg['randomcrop']
            transforms.append(A.RandomCrop(crop_height, crop_width, p=1))
        if trans_cfg.get('final_resize', False):
            final_resize = trans_cfg['final_resize']
            transforms.append(A.Resize(height=final_resize[0], width=final_resize[1]))
        if normalise:
            transforms.append(A.Normalize(mean=mean, std=std))
        else:
            print(f"WARNING: NOT NORMALISING! Assume you are checking normalisation parameters?")
        return A.Compose(transforms)

    if cfg['pose_or_seg'] == 'seg':
        res = cfg['res']
        train_transforms = get_transforms(cfg['transforms'][res]['train'], normalise=normalise)
        test_transforms = get_transforms(cfg['transforms'][res]['test'], normalise=normalise)
    elif cfg['pose_or_seg'] == 'pose':
        train_transforms = get_transforms(cfg['transforms']['train'], normalise=normalise)
        test_transforms = get_transforms(cfg['transforms']['test'], normalise=normalise)
    else:
        raise ValueError()

    return train_transforms, test_transforms
