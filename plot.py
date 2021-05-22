#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from matplotlib import pyplot as plt
import SimpleITK as sitk

def show_image(image_list, nrow=1, ncol=3):
    for i in range(len(image_list)):
        plt.figure()
        plt.subplot(nrow, ncol, 1)
        plt.imshow(image_list[i][int((image_list[i].shape[0]) / 2), :, :], cmap='gray')
        plt.title('Axial slice')
        plt.subplot(nrow, ncol, 2)
        plt.imshow(image_list[i][:, int((image_list[i].shape[1]) / 2), :], cmap='gray')
        plt.title('Coronal slice')
        plt.subplot(nrow, ncol, 3)
        plt.imshow(image_list[i][:, :, int((image_list[i].shape[2]) / 2)], cmap='gray')
        plt.title('Sagittal slice')
        plt.show()

def show_overlaid_images(image1, image2, alpha = 0.3):
    # Visualization of the resampled image overlaid to the fixed image
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(image1[int(image1.shape[0]/ 2), :, :], cmap='Blues')
    plt.imshow(image2[int(image2.shape[0]/ 2), :, :], cmap='Reds', alpha = alpha)
    plt.title('Axial slices')

    plt.subplot(1, 3, 2)
    plt.imshow(image1[:, int(image1.shape[1]/ 2), :], cmap='Blues')
    plt.imshow(image2[:, int(image2.shape[1]/ 2), :], cmap='Reds', alpha = alpha)
    plt.title('Coronal slices')

    plt.subplot(1, 3, 3)
    plt.imshow(image1[:, :, int(image1.shape[2] / 2)], cmap='Blues')
    plt.imshow(image2[:, :, int(image2.shape[2] / 2)], cmap='Reds', alpha = alpha)
    plt.title('Sagittal slices')
    plt.show()

def show_obtained_masks_from_snake():
    # load data
    # common masks (obtained from SNAKE) and images
    common_masks_list_paths  = ["./data/COMMON_images_masks/common_40_mask_2c.nii.gz",
                                "./data/COMMON_images_masks/common_41_mask_2c.nii.gz",
                                "./data/COMMON_images_masks/common_42_mask_2c.nii.gz"]
    common_images_list_paths = ["./data/COMMON_images_masks/common_40_image.nii.gz",
                                "./data/COMMON_images_masks/common_41_image.nii.gz",
                                "./data/COMMON_images_masks/common_42_image.nii.gz"]

    # group8 masks (obtained from SNAKE) and images
    group_masks_list_paths =  ["./data/GROUP_images/g8_74_mask_2c.nii.gz",
                               "./data/GROUP_images/g8_75_mask_2c.nii.gz",
                               "./data/GROUP_images/g8_76_mask_2c.nii.gz"]
    group_images_list_paths = ["./data/GROUP_images/g8_74_image.nii.gz",
                               "./data/GROUP_images/g8_75_image.nii.gz",
                               "./data/GROUP_images/g8_76_image.nii.gz"]

    common_masks_list  = [sitk.GetArrayFromImage(sitk.ReadImage(file_name)) for file_name in common_masks_list_paths]
    common_images_list = [sitk.GetArrayFromImage(sitk.ReadImage(file_name, sitk.sitkFloat32)) for file_name in common_images_list_paths]
    group_masks_list   = [sitk.GetArrayFromImage(sitk.ReadImage(file_name)) for file_name in group_masks_list_paths]
    group_images_list  = [sitk.GetArrayFromImage(sitk.ReadImage(file_name, sitk.sitkFloat32)) for file_name in group_images_list_paths]

    # show common and group masks
    show_overlaid_images(common_images_list[0], common_masks_list[0], alpha=0.5) # show overlaid image and mask for common_40
    show_overlaid_images(group_images_list[0], group_masks_list[0], alpha=0.5) # show overlaid image and mask for group_74
    # show_image(common_masks_list)
    # show_image(group_images_list

def show_obtained_masks_from_atlas_based_seg():
    # load data
    # common masks (obtained from atlas_seg) and images
    common_masks_list_paths  = ["./data/outputs/common_40_est_mask_2c.nii.gz",
                                "./data/outputs/common_41_est_mask_2c.nii.gz",
                                "./data/outputs/common_42_est_mask_2c.nii.gz"]
    common_images_list_paths = ["./data/COMMON_images_masks/common_40_image.nii.gz",
                                "./data/COMMON_images_masks/common_41_image.nii.gz",
                                "./data/COMMON_images_masks/common_42_image.nii.gz"]
    common_masks_list = [sitk.GetArrayFromImage(sitk.ReadImage(file_name)) for file_name in common_masks_list_paths]
    common_images_list = [sitk.GetArrayFromImage(sitk.ReadImage(file_name, sitk.sitkFloat32)) for file_name in
                          common_images_list_paths]
    show_overlaid_images(common_images_list[0], common_masks_list[0], alpha=0.5)
    show_overlaid_images(common_images_list[1], common_masks_list[1], alpha=0.5)
    show_overlaid_images(common_images_list[2], common_masks_list[2], alpha=0.5)


#show_obtained_masks_from_snake()
#show_obtained_masks_from_atlas_based_seg()