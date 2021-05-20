#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import SimpleITK as sitk
import numpy as np
import os

def resample_image(itk_image, out_size = [512,512,205], is_label=False):

    if is_label:
        interpolator_type = sitk.sitkNearestNeighbor
    else:
        interpolator_type = sitk.sitkLinear


    out_spacing = [origin_sz*origin_spc/out_sz  for origin_sz, origin_spc, out_sz in zip(itk_image.GetSize(), itk_image.GetSpacing(), out_size)]


    return sitk.Resample(itk_image, out_size, sitk.Transform(), interpolator_type, itk_image.GetOrigin(), out_spacing, itk_image.GetDirection(), 0.0, itk_image.GetPixelIDValue())
    print_itk_img_info(_new_vol)


def print_itk_img_info(itk_img):
    selected_image = itk_img
    print('--------------------')
    print('origin: ' + str(selected_image.GetOrigin()))
    print('size: ' + str(selected_image.GetSize()))
    print('spacing: ' + str(selected_image.GetSpacing()))
    print('direction: ' + str(selected_image.GetDirection()))
    print('pixel type: ' + str(selected_image.GetPixelIDTypeAsString()))
    print('number of pixel components: ' + str(selected_image.GetNumberOfComponentsPerPixel()))
    print('GetPixelIDValue: ' + str(selected_image.GetPixelIDValue()))

def resample_data_in_dir(input_dir, output_dir, image_view = False, print_info = False):
    data_names = os.listdir(input_dir)
    if image_view:
        image_viewer = sitk.ImageViewer()
        image_viewer.SetApplication('/usr/bin/itksnap')

    for data_name in data_names:
        # resampling and cropping
        input_path = input_dir+data_name
        output_path = output_dir+data_name

        itk_img = sitk.ReadImage(input_path)
        if 'mask' in data_name:
            is_label = True
        else:
            is_label = False

        new_itk_img = resample_image(itk_img, is_label=is_label)

        if image_view:
            image_viewer.Execute(itk_img)
            image_viewer.Execute(new_itk_img)

        if print_info:
            print('--- Info before resample: ---')
            print_itk_img_info(itk_img)
            print('\n --- Info after resample: --- ')
            print_itk_img_info(new_itk_img)

        # save image
        sitk.WriteImage(new_itk_img, output_path)

# # for common data
# input_dir = './data/COMMON_images_masks/'
# output_dir = './data/Resized/COMMON/'
# resample_data_in_dir(input_dir, output_dir)
#
# # for group data
# input_dir = './data/GROUP_images/'
# output_dir = './data/Resized/GROUP8/'
# resample_data_in_dir(input_dir, output_dir)

