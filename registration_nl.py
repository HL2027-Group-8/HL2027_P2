#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


def smooth_and_resample(image, shrink_factor, smoothing_sigma):
    """
    Args:
        image: The image we want to resample.
        shrink_factor: A number greater than one, such that the new image's size is original_size/shrink_factor.
        smoothing_sigma: Sigma for Gaussian smoothing, this is in physical (image spacing) units, not pixels.
    Return:
        Image which is a result of smoothing the input and then resampling it using the given sigma and shrink factor.
    """
    smoothed_image = sitk.SmoothingRecursiveGaussian(image, smoothing_sigma)

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [int(sz / float(shrink_factor) + 0.5) for sz in original_size]
    new_spacing = [((original_sz - 1) * original_spc) / (new_sz - 1)
                   for original_sz, original_spc, new_sz in zip(original_size, original_spacing, new_size)]
    return sitk.Resample(smoothed_image, new_size, sitk.Transform(),
                         sitk.sitkLinear, image.GetOrigin(),
                         new_spacing, image.GetDirection(), 0.0,
                         image.GetPixelID())


def multiscale_demons(registration_algorithm,
                      fixed_image, moving_image, initial_transform=None,
                      shrink_factors=None, smoothing_sigmas=None):
    """
    Run the given registration algorithm in a multiscale fashion. The original scale should not be given as input as the
    original images are implicitly incorporated as the base of the pyramid.
    Args:
        registration_algorithm: Any registration algorithm that has an Execute(fixed_image, moving_image, displacement_field_image)
                                method.
        fixed_image: Resulting transformation maps points from this image's spatial domain to the moving image spatial domain.
        moving_image: Resulting transformation maps points from the fixed_image's spatial domain to this image's spatial domain.
        initial_transform: Any SimpleITK transform, used to initialize the displacement field.
        shrink_factors: Shrink factors relative to the original image's size.
        smoothing_sigmas: Amount of smoothing which is done prior to resmapling the image using the given shrink factor. These
                          are in physical (image spacing) units.
    Returns:
        SimpleITK.DisplacementFieldTransform
    """
    # Create image pyramid.
    fixed_images = [fixed_image]
    moving_images = [moving_image]
    if shrink_factors:
        for shrink_factor, smoothing_sigma in reversed(list(zip(shrink_factors, smoothing_sigmas))):
            fixed_images.append(smooth_and_resample(fixed_images[0], shrink_factor, smoothing_sigma))
            moving_images.append(smooth_and_resample(moving_images[0], shrink_factor, smoothing_sigma))

    # Create initial displacement field at lowest resolution.
    # Currently, the pixel type is required to be sitkVectorFloat64 because of a constraint imposed by the Demons filters.
    if initial_transform:
        initial_displacement_field = sitk.TransformToDisplacementField(initial_transform,
                                                                       sitk.sitkVectorFloat64,
                                                                       fixed_images[-1].GetSize(),
                                                                       fixed_images[-1].GetOrigin(),
                                                                       fixed_images[-1].GetSpacing(),
                                                                       fixed_images[-1].GetDirection())
    else:
        initial_displacement_field = sitk.Image(fixed_images[-1].GetWidth(),
                                                fixed_images[-1].GetHeight(),
                                                fixed_images[-1].GetDepth(),
                                                sitk.sitkVectorFloat64)
        initial_displacement_field.CopyInformation(fixed_images[-1])

    # Run the registration.
    initial_displacement_field = registration_algorithm.Execute(fixed_images[-1],
                                                                moving_images[-1],
                                                                initial_displacement_field)
    # Start at the top of the pyramid and work our way down.
    for f_image, m_image in reversed(list(zip(fixed_images[0:-1], moving_images[0:-1]))):
        initial_displacement_field = sitk.Resample(initial_displacement_field, f_image)
        initial_displacement_field = registration_algorithm.Execute(f_image, m_image, initial_displacement_field)
    return sitk.DisplacementFieldTransform(initial_displacement_field)

def est_nl_transf(fix_img, mov_img,fix_mask):
    """
    Estimate non-linear transform to align `im_mov` to `im_ref` and
    return the transform parameters.
    """
    demons_filter = sitk.DemonsRegistrationFilter()
    demons_filter.SetNumberOfIterations(20)
    # Regularization (update field - viscous, total field - elastic).
    demons_filter.SetSmoothDisplacementField(True)
    demons_filter.SetStandardDeviations(2.0)

    # Run the registration.
    tx = multiscale_demons(registration_algorithm=demons_filter,
                           fixed_image=fix_img,
                           moving_image=mov_img,
                           shrink_factors=[4, 2],
                           smoothing_sigmas=[8, 4])

    return tx

def apply_nl_transf(fix_img, mov_img, nl_transf):
    """
    Apply given non-linear transform `nl_xfm` to `im_mov` and return
    the transformed image."""
    return sitk.Resample(mov_img, fix_img, nl_transf, sitk.sitkNearestNeighbor, 0.0, mov_img.GetPixelID())

def seg_atlas(im, atlas_ct_list, atlas_seg_list):
    """
    Apply atlas-based segmentation of `im` using the list of CT
    images in `atlas_ct_list` and the corresponding segmentation masks
    in `atlas_seg_list`. Return the resulting segmentation mask after
    majority voting.
    """
    pass

def train_classifier(im_list, labels_list):
    """
    Receive a list of images `im_list` and a list of vectors (one
    per image) with the labels 0 or 1 depending on the sagittal 2D slice
    contains or not the pubic symphysis. Returns the trained classifier.
    """
    pass

def pubic_symphysis_selection(im, classifier):
    """
    Receive a CT image and the trained classifier. Returns the
    sagittal slice number with the maximum probability of containing the
    pubic symphysis.
    """
    pass

# load ref images and masks
fix_img_filepath = './data/Resized/COMMON/common_40_image.nii.gz'
fix_mask_filepath = './data/Resized/COMMON/common_40_mask_2c.nii.gz'
fix_img = sitk.ReadImage(fix_img_filepath, sitk.sitkFloat32)
fix_mask = sitk.ReadImage(fix_mask_filepath, sitk.sitkFloat32)

# load moving images
mov_img_filepath = './data/Resized/GROUP8/g8_74_image.nii.gz'
mov_img = sitk.ReadImage(mov_img_filepath, sitk.sitkFloat32)
nl_transf = est_nl_transf(fix_img,mov_img,fix_mask)

image_viewer = sitk.ImageViewer()
image_viewer.SetApplication('/usr/bin/itksnap')
# apply lin transf
mov_img_resampled = apply_nl_transf(fix_img, mov_img, nl_transf)
image_viewer.Execute(fix_img)
image_viewer.Execute(mov_img)
image_viewer.Execute(mov_img_resampled)