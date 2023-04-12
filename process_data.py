import os
import numpy as np
import SimpleITK as sitk
import json
from bb import bounding_box


def resize(img, newSize, mask=False):
    resampler = sitk.ResampleImageFilter()
    originSize = img.GetSize()
    originSpacing = img.GetSpacing()
    newSize = np.array(newSize, float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int)

    resampler.SetReferenceImage(img)
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(
        sitk.sitkLinear if not mask else sitk.sitkNearestNeighbor)
    img = resampler.Execute(img)

    return img


def resample(img, mask=False):
    # Resample slice to isotropic
    original_spacing = img.GetSpacing()
    original_size = img.GetSize()

    new_spacing = [1, 1, 1]
    new_size = [
        int(round(original_size[0] * (original_spacing[0] / 1))),
        int(round(original_size[1] * (original_spacing[1] / 1))),
        int(round(original_size[2] * (original_spacing[2] / 1)))
    ]
    resampleSliceFilter = sitk.ResampleImageFilter()
    resampleSliceFilter.SetSize(new_size)
    resampleSliceFilter.SetTransform(sitk.Transform())
    resampleSliceFilter.SetInterpolator(
        sitk.sitkLinear if not mask else sitk.sitkNearestNeighbor)
    resampleSliceFilter.SetOutputOrigin(img.GetOrigin())
    resampleSliceFilter.SetOutputSpacing(new_spacing)
    resampleSliceFilter.SetOutputDirection(img.GetDirection())
    resampleSliceFilter.SetDefaultPixelValue(0)
    resampleSliceFilter.SetOutputPixelType(img.GetPixelID())

    img = resampleSliceFilter.Execute(img)
    return img


def gen(nii_name, save_img, output_shape, start, end):
    for i in range(start, end):
        sub_seg_dir = "./results/single_msk/" + nii_name[9:-7] + "/" + nii_name[9:-7] + "_" + str(i).zfill(2) + "_seg.nii.gz"

        img_sitk = sitk.ReadImage(nii_name)
        mask_sitk = sitk.ReadImage(sub_seg_dir)

        originOrigin = img_sitk.GetOrigin()
        originSize = img_sitk.GetSize()
        originSpace = img_sitk.GetSpacing()
        # mskSize = mask_sitk.GetSize()
        mskSpace = mask_sitk.GetSpacing()
        mskOrigin = mask_sitk.GetOrigin()

        img_np = sitk.GetArrayFromImage(img_sitk)
        mask_np = sitk.GetArrayFromImage(mask_sitk)

        del img_sitk, mask_sitk

        img_np = np.transpose(img_np, (2, 1, 0))
        mask_np = np.transpose(mask_np, (2, 1, 0))

        mask_np[mask_np != 1] = 0

        if not np.all(mask_np==0):
            # mask_np[mask_np>0] =1

            mask_down = bounding_box(mask_np)[0]
            mask_up = bounding_box(mask_np)[1]

            originSize = np.array(originSize)

            down = np.zeros(shape=(3))
            up = np.zeros(shape=(3))
            up[0] = originSize[0]
            up[1] = originSize[1]

            # 0.05*originSize[2] (deprecated) 0.05*originSize较大
            pad = mask_up[2]-mask_down[2]
            down[2] = np.floor(mask_down[2]-0.06*pad)
            up[2] = np.ceil((mask_up[2]+1)+0.06*pad)  # 这里是开区间

            down = down.astype(int)
            up = up.astype(int)

            down[down < 0] = 0
            up = np.where(up <= originSize, up, originSize)

            # 由于开区间，正好不用e+1，因为切片不包括e
            slices = tuple([slice(s, e) for s, e in zip(down, up)])

            img_cropped = sitk.GetImageFromArray(
                np.transpose(img_np[slices], (2, 1, 0)))
            mask_cropped = sitk.GetImageFromArray(
                np.transpose(mask_np[slices], (2, 1, 0)))

            del img_np, mask_np

            img_cropped.SetSpacing(originSpace)
            img_cropped.SetOrigin(originOrigin)
            mask_cropped.SetSpacing(mskSpace)
            mask_cropped.SetOrigin(mskOrigin)

            img_np = sitk.GetArrayFromImage(img_cropped)
            mask_np = sitk.GetArrayFromImage(mask_cropped)


            img_np = np.transpose(img_np, (2, 1, 0))
            mask_np = np.transpose(mask_np, (2, 1, 0))

            mask_down = bounding_box(mask_np)[0]
            mask_up = bounding_box(mask_np)[1]

            originSize = np.array(originSize)


            down = np.floor(mask_down - 0.02 * originSize).astype(int)
            up = np.ceil((mask_up + 1) + 0.02 * originSize).astype(int)  # 这里是开区间
            # down = np.floor(mask_down-0.1*originSize).astype(int)
            # up = np.ceil((mask_up+1)+0.1*originSize).astype(int)  # 这里是开区间

            down[down < 0.] = 0
            up = np.where(up <= originSize, up, originSize)

            # 由于开区间，正好不用e+1，因为切片不包括e
            slices = tuple([slice(s, e) for s, e in zip(down, up)])



            img_cropped = sitk.GetImageFromArray(
                np.transpose(img_np[slices], (2, 1, 0)))
            mask_cropped = sitk.GetImageFromArray(
                np.transpose(mask_np[slices], (2, 1, 0)))

            del img_np, mask_np

            img_cropped.SetSpacing(originSpace)
            mask_cropped.SetSpacing(mskSpace)

            img_sitk = resample(img_cropped)
            mask_sitk = resample(mask_cropped, mask=True)

            img_sitk = resize(img_sitk, newSize=output_shape)
            mask_sitk = resize(mask_sitk, newSize=output_shape, mask=True)

            # get vocel data, depth, height, width(z,y,x)(IAR)
            img_np = sitk.GetArrayFromImage(img_sitk)
            mask_np = sitk.GetArrayFromImage(mask_sitk)

            # transpose to x,y,z RAI
            img_np = np.transpose(img_np, (2, 1, 0))

            mask_np = np.transpose(mask_np, (2, 1, 0))

            del img_sitk, mask_sitk

            np.save(save_img + "/" + nii_name[9:-7] + "/" + nii_name[9:-7] + "_" + str(i).zfill(2) + ".npy", img_np)

            del img_np, mask_np