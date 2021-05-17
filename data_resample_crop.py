#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import SimpleITK as sitk
import numpy as np
import os

def resample_image(itk_image, out_spacing=[1.0, 1.0, 1.0], out_size = [512,512,205], is_label=False):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    original_origin = itk_image.GetOrigin()

    resample = sitk.ResampleImageFilter()
    # resample.SetOutputSpacing(out_spacing)
    resample.SetOutputSpacing(original_spacing)
    resample.SetSize(out_size)
    # resample.SetSize(original_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)


def print_itk_img_info(itk_img):
    selected_image = itk_img
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

        new_itk_img = resample_image(itk_img, out_spacing=[1.0, 1.0, 1.0], is_label=is_label)

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

# for common data
# input_dir = './data/COMMON_images_masks/'
# output_dir = './data/Resized1/COMMON/'
# resample_data_in_dir(input_dir, output_dir)

# for group data
input_dir = './data/GROUP_images/'
output_dir = './data/Resized1/GROUP8/'
resample_data_in_dir(input_dir, output_dir)

