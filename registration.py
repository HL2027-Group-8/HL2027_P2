#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
from data_resample_crop import *

def est_lin_transf(fix_img, mov_img,fix_mask, print_log = False):
    """
    Estimate linear transform to align `mov_img` to `fix_img` and
    return the transform parameters.
    """

    # initial alignment of the two volumes
    initial_transform = sitk.CenteredTransformInitializer(fix_img,mov_img,sitk.AffineTransform(3),sitk.CenteredTransformInitializerFilter.GEOMETRY)

    # initialize the registration
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMeanSquares()
    # registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)

    # set the mask on which you are going to evaluate the similarity between the two images
    registration_method.SetMetricFixedMask(fix_mask)

    # interpolator
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=1, numberOfIterations=100, convergenceMinimumValue=1e-6,
                                                      convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Set the initial moving and optimized transforms.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # perform registration
    final_transform = registration_method.Execute(sitk.Cast(fix_img, sitk.sitkFloat32),sitk.Cast(mov_img, sitk.sitkFloat32))

    if print_log:
        print("--------")
        print("Affine registration:")
        print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
        print("Optimizer stop condition: {0}".format(registration_method.GetOptimizerStopConditionDescription()))
        print("Number of iterations: {0}".format(registration_method.GetOptimizerIteration()))
        print("--------")
    return final_transform

def apply_lin_transf(fix_img, mov_img, lin_transf, is_label=False):
    """
    Apply given linear transform `lin_transf` to `mov_img` and return
    the transformed image.
    """
    # only supports images with sitkFloat32 and sitkFloat64 pixel types
    fix_img = sitk.Cast(fix_img, sitk.sitkFloat32)
    mov_img = sitk.Cast(mov_img, sitk.sitkFloat32)

    # resample moving image
    resampler = sitk.ResampleImageFilter()

    # set the reference image
    resampler.SetReferenceImage(fix_img)

    # use a linear interpolator
    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)

    # set the desired transformation
    resampler.SetTransform(lin_transf)

    mov_img_resampled = resampler.Execute(mov_img)
    mov_img_resampled_data = sitk.GetArrayFromImage(mov_img_resampled)
    return mov_img_resampled

def est_nl_transf(fix_img, mov_img,fix_mask,print_log=False):
    """
    Estimate non-linear transform to align `im_mov` to `im_ref` and
    return the transform parameters.
    """

    # initialize the registration
    reg_method = sitk.ImageRegistrationMethod()

    # create initial identity transformation.
    transform_to_displacement_field_filter = sitk.TransformToDisplacementFieldFilter()
    transform_to_displacement_field_filter.SetReferenceImage(fix_img)
    initial_transform = sitk.DisplacementFieldTransform(transform_to_displacement_field_filter.Execute(sitk.Transform()))

    #  regularization. The update field refers to fluid regularization; the total field to elastic regularization.
    initial_transform.SetSmoothingGaussianOnUpdate(varianceForUpdateField=0.0, varianceForTotalField=2.0)

    # set the initial transformation
    reg_method.SetInitialTransform(initial_transform)

    # use the function 'SetMetricAsDemons' to be able to perform Demons registration.
    # Be aware that you will need to provide a parameter (the intensity difference threshold) as input:
    # during the registration, intensities are considered to be equal if their difference is less than the given threshold.
    reg_method.SetMetricAsDemons(0.01)

    # evaluate the metrics only in the mask, if provided as an input
    reg_method.SetMetricFixedMask(fix_mask)

    # Multi-resolution framework.
    reg_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    reg_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[8,4,0])

    # set a linear interpolator
    reg_method.SetInterpolator(sitk.sitkLinear)

    # set a gradient descent optimizer
    reg_method.SetOptimizerAsGradientDescent(learningRate=0.5, numberOfIterations=40, convergenceMinimumValue=1e-6,
                                             convergenceWindowSize=10)
    reg_method.SetOptimizerScalesFromPhysicalShift()

    if print_log:
        print("--------")
        print("Demons registration:")
        print('Final metric value: {0}'.format(reg_method.GetMetricValue()))
        print("Optimizer stop condition: {0}".format(reg_method.GetOptimizerStopConditionDescription()))
        print("Number of iterations: {0}".format(reg_method.GetOptimizerIteration()))
        print("--------")
    return reg_method.Execute(sitk.Cast(fix_img, sitk.sitkFloat32), sitk.Cast(mov_img, sitk.sitkFloat32))

def apply_nl_transf(fix_img, mov_img, nl_transf, is_label=False):
    """
    Apply given non-linear transform `nl_xfm` to `im_mov` and return
    the transformed image."""
    if is_label:
        output = sitk.Resample(mov_img, fix_img, nl_transf, sitk.sitkNearestNeighbor, 0.0, mov_img.GetPixelID())
    else:
        output = sitk.Resample(mov_img, fix_img, nl_transf, sitk.sitkLinear, 0.0, mov_img.GetPixelID())
    return output

def seg_atlas(atlas_seg_list,image_view=False):
    """
    Apply atlas-based segmentation of `im` using the list of CT
    images in `atlas_ct_list` and the corresponding segmentation masks
    in `atlas_seg_list`. Return the resulting segmentation mask after
    majority voting.
    """
    labelForUndecidedPixels = 10
    reference_segmentation= sitk.LabelVoting(atlas_seg_list, labelForUndecidedPixels)
    if image_view:
        image_viewer.Execute(reference_segmentation)
    return reference_segmentation

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

if __name__ == '__main__':
    image_view = False # if show image via itksnap

    # load data
    fix_img_indexes = [40,41,42] # common data
    for i in fix_img_indexes:
        # load fix images and masks
        fix_img_filepath = './data/COMMON_images_masks/common_{0}_image.nii.gz'.format(i)
        fix_mask_filepath = './data/COMMON_images_masks/common_{0}_mask_2c_gt.nii.gz'.format(i) # choose ground true as mask
        fix_img = sitk.ReadImage(fix_img_filepath, sitk.sitkFloat32)
        fix_mask = sitk.ReadImage(fix_mask_filepath)

        mov_img_indexes = [74,75,76]
        atlas_ct_list = [] # list of aligned (non-linear and linear) images in group
        atlas_seg_list = [] # list of aligned masks in group
        for j in mov_img_indexes:
            # load moving images and masks
            mov_img_filepath = './data/GROUP_images/g8_{0}_image.nii.gz'.format(j)
            mov_mask_filepath = './data/GROUP_images/g8_{0}_mask_2c.nii.gz'.format(j) # semi-auto labeled masks by itk-snap
            mov_img = sitk.ReadImage(mov_img_filepath, sitk.sitkFloat32)
            mov_mask = sitk.ReadImage(mov_mask_filepath)

            # data resample to speed up when testing
            fix_img = resample_image(fix_img, out_size = [128,128,205], is_label=False)
            mov_img = resample_image(mov_img, out_size = [128,128,205], is_label=False)
            mov_mask = resample_image(mov_mask, out_size = [128,128,205], is_label=True)
            fix_mask = resample_image(fix_mask, out_size = [128,128,205], is_label=True)

            # do affine registration
            lin_transf = est_lin_transf(fix_img, mov_img, fix_mask, print_log=True)
            lin_aligned_image = apply_lin_transf(fix_img, mov_img, lin_transf, is_label=False)
            lin_aligned_mask = apply_lin_transf(fix_img, mov_mask, lin_transf, is_label=True)

            # do demons registration
            nl_transf = est_nl_transf(fix_img, lin_aligned_image, fix_mask)
            nl_aligned_image = apply_nl_transf(fix_img, lin_aligned_image, nl_transf, is_label=False)
            nl_aligned_mask = apply_nl_transf(fix_img, lin_aligned_mask, nl_transf, is_label=True)
            atlas_ct_list.append(nl_aligned_image)
            atlas_seg_list.append(sitk.Cast(nl_aligned_mask,sitk.sitkUInt8))

            if image_view:
                image_viewer = sitk.ImageViewer()
                image_viewer.SetApplication('/usr/bin/itksnap')
                image_viewer.SetTitle('fix_img')
                image_viewer.Execute(fix_img)
                image_viewer.SetTitle('mov_img')
                image_viewer.Execute(mov_img)
                image_viewer.SetTitle('lin_aligned_image')
                image_viewer.Execute(lin_aligned_image)
                image_viewer.SetTitle('nl_aligned_image')
                image_viewer.Execute(nl_aligned_image)
                image_viewer.SetTitle('fix_mask')
                image_viewer.Execute(fix_mask)
                image_viewer.SetTitle('mov_mask')
                image_viewer.Execute(mov_mask)
                image_viewer.SetTitle('lin_aligned_mask')
                image_viewer.Execute(lin_aligned_mask)
                image_viewer.SetTitle('nl_aligned_mask')
                image_viewer.Execute(nl_aligned_mask)


        # do atlas_based seg
        est_fix_mask = seg_atlas(atlas_seg_list,image_view)

        # save image
        est_fix_mask_filepath = './data/COMMON_images_masks/common_{0}_est_mask_2c.nii.gz'.format(i)
        sitk.WriteImage(est_fix_mask, est_fix_mask_filepath)
