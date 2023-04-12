import os
import shutil
import time
import numpy as np

from new_classification import get_classification
from process_data import gen
from segment_spine import binary_segmentor
from segment_vertebra import per_location_refiner_segmentor
from yolov7.detect import detect_full_spine


def load_models(seg_spine_norm=False, seg_vert_norm=False):

    if seg_spine_norm:
        model_file_seg_binary = 'models/segmentor_spine_norm.pth'
    else:
        model_file_seg_binary = 'models/segmentor_spine.pth'

    if seg_vert_norm:
        model_file_seg_idv = 'models/segmentor_vertebra_norm.pth'
    else:
        model_file_seg_idv = 'models/segmentor_vertebra.pth'

    model_file_loc_sag = 'models/locator_sagittal.pth'
    model_file_loc_cor = 'models/locator_coronal.pth'

    id_group_model_file = 'models/classifier_group.pth'
    id_cer_model_file = 'models/classifier_cervical.pth'
    id_thor_model_file = 'models/classifier_thoracic.pth'
    id_lum_model_file = 'models/classifier_lumbar.pth'


    return {'seg_binary': model_file_seg_binary, 'seg_individual': model_file_seg_idv,
            'loc_sagittal': model_file_loc_sag, 'loc_coronal': model_file_loc_cor,
            'id_group': id_group_model_file, 'id_cervical': id_cer_model_file,
            'id_thoracic': id_thor_model_file, 'id_lumbar': id_lum_model_file}

def get_size_and_spacing_and_orientation_from_nifti_file(file):
    import nibabel as nib

    data = nib.load(file)

    size = data.shape

    # read orientation code
    a, b, c = nib.orientations.aff2axcodes(data.affine)
    orientation_code = a+b+c

    # read voxel spacing
    header = data.header
    pixdim = header['pixdim']
    spacing = pixdim[1:4]

    aff = data.affine

    return size, spacing, orientation_code, aff

def resampling(nifti_img, spacing, target_shape=None):
    from nilearn.image import resample_img
    import numpy as np

    new_affine = np.copy(nifti_img.affine)
    new_affine[:3, :3] *= 1.0/spacing

    if target_shape is None:
        target_shape = (nifti_img.shape * spacing).astype(int)

    resampled_nifti_img = resample_img(nifti_img, target_affine=new_affine,
                                                  target_shape=target_shape,
                                                  interpolation='nearest')
    # also return nifti image
    return resampled_nifti_img

def reorienting(img, start_orient_code, end_orient_code):
    import nibabel as nib

    start_orient = nib.orientations.axcodes2ornt(start_orient_code)
    end_orient = nib.orientations.axcodes2ornt(end_orient_code)

    trans = nib.orientations.ornt_transform(start_orient, end_orient)

    return nib.orientations.apply_orientation(img, trans)

def read_isotropic_pir_img_from_nifti_file(file, itm_orient='PIR'):
    import nibabel as nib

    _, spacing, orientation_code, _ = get_size_and_spacing_and_orientation_from_nifti_file(file)

    nifti_img = nib.load(file)

    resampled_nifti_img = resampling(nifti_img, spacing)
    resampled_img = resampled_nifti_img.get_fdata()

    transformed_img = reorienting(resampled_img, orientation_code, itm_orient)
    return transformed_img

def reorient_resample_back_to_original(img, ori_orient_code, spacing, ori_size, ori_aff, itm_orient='PIR'):
    import nibabel as nib
    import numpy as np

    transformed_img = reorienting(img, itm_orient, ori_orient_code)

    nifti_img = nib.Nifti1Image(transformed_img, ori_aff)

    resampled_nifti_img = resampling(nifti_img, 1.0/spacing, ori_size)

    return resampled_nifti_img.get_fdata()

def save_to_nifti_file(img, save_filename, aff=None):
    import nibabel as nib
    import os
    import numpy as np

    if aff is not None:
        img = nib.Nifti1Image(img, aff)
    else:
        img = nib.Nifti1Image(img, np.eye(4))

    nib.save(img, save_filename)
    print('saved to {}'.format(save_filename))

# 将单个椎体掩膜处理成识别网络需要的包围盒尺寸
def process_data(img_dir, seg_dir, save_img):
    gen(img_dir, seg_dir, save_img, output_shape=(64, 64, 32))

# 通过5节识别结果的到整个椎体的结果
def get_whole_res(ver_num, cla_res, start, end):
    res = np.full(ver_num, 100)

    res[start: end] = cla_res

    for i in range(0, start):
        add_num = cla_res.min() - (start - i)
        if add_num >= 0:
            res[i] = add_num

    for i in range(end, ver_num):
        add_num = cla_res.max() + i - end + 1
        if add_num <= 24:
            res[i] = add_num

    return res

# 将单个脊柱掩膜合并成整个掩膜
def merge_single_mask(pir_shape, vertebra_num, total_single_mask, ori_orient_code, ori_spacing, ori_size, ori_aff):
    # 将每一块单个椎骨掩膜合并为整体分割掩膜
    test = np.zeros(pir_shape)

    for i in range(vertebra_num):
        test = np.logical_or(test, total_single_mask[i])

    test = bool_to_01(test)

    # 保存合并后的分割结果
    write_result_to_file(test, ori_orient_code, ori_spacing, ori_size, ori_aff, "sample", "test")

# 将单个掩膜结果切除小翅膀
def cut_mask(pir_shape, mix, total_single_mask):
    # 切除小翅膀
    cut_mask = np.zeros(pir_shape)
    cut_mask[mix[i][0] : mix[i][1], :, :] = 1

    # 合并小翅膀掩膜
    total_single_mask[i] = np.logical_and(cut_mask, total_single_mask[i])


# 将bool掩膜转换为01掩膜
def bool_to_01(matrix):
    return np.where(matrix, 1.0, 0.0)

# 保存结果
def write_result_to_file(pir_mask, ori_orient_code, spacing, ori_size, ori_aff, save_dir, filename):
    mask = reorient_resample_back_to_original(pir_mask, ori_orient_code, spacing, ori_size, ori_aff)

    save_to_nifti_file(mask, os.path.join(save_dir, '{}_seg.nii.gz'.format(filename)), ori_aff)



if __name__ == '__main__':
    data_path = os.listdir("./sample")
    data_path.sort()
    for pt in data_path:
        total_start = time.time()

        res_list = {0: "C1", 1: "C2", 2: "C3", 3: "C4", 4: "C5", 5: "C6", 6: "C7", 7: "T1", 8: "T2", 9: "T3", 10: "T4", 11: "T5", 12: "T6", 13: "T7", 14: "T8", 15: "T9", 16:"T10",
                    17: "T11", 18: "T12", 19: "L1", 20: "L2", 21: "L3", 22: "L4", 23: "L5", 24: "S", 100: "NULL"}

        # 加载相关模型及参数
        models = load_models(False, False)

        nii_name = "./sample/" + pt

        # 创建结果文件夹
        os.makedirs("./results/npy/" + pt[:-7], exist_ok=True)
        os.makedirs("./results/single_msk/" + pt[:-7], exist_ok=True)
        os.makedirs("./results/final_mask/" + pt[:-7], exist_ok=True)

        # 记录最开始数据的参数
        ori_size, ori_spacing, ori_orient_code, ori_aff = get_size_and_spacing_and_orientation_from_nifti_file(nii_name)

        # 读取并预处理数据
        pir_img = read_isotropic_pir_img_from_nifti_file(nii_name)

        # 获得数据的二值分割结果
        start_seg = time.time()
        binary_mask = binary_segmentor(pir_img, models['seg_binary'], mode='overlap', norm=False)
        # write_result_to_file(binary_mask, ori_orient_code, ori_spacing, ori_size, ori_aff, "sample", "resres")
        end_seg = time.time()
        print("二值分割总共时间开销为：{}".format(end_seg - start_seg))

        # 获得冠状面方向的掩膜中心线
        index_start = 0
        index_end = 0

        for i in range(pir_img.shape[2]):
            if binary_mask[:,:,i].max() != 0:
                index_start = i
                break

        for i in range(pir_img.shape[2] - 1, -1, -1):
            if binary_mask[:, :, i].max() != 0:
                index_end = i
                break

        index_center = int((index_end + index_start) / 2)
        print("冠状面的中心线位置为：{}".format(index_center))

        # 这一步使用Yolo获得单个椎骨的定位坐标
        res = detect_full_spine.detect(nii_name[9:-7])[0][1]
        res = np.array(res)

        # 去除异常检测框
        meanhei = res.mean(axis=0)[-1] * 0.4
        drop_index = []
        for i in range(res.shape[0]):
            if (res[i][-1] < meanhei):
                drop_index.append(i)
        res = np.delete(res, drop_index, 0)

        print("丢弃框的数量为：{}".format(len(drop_index)))
        print(res)

        vertebra_num = res.shape[0]
        start = int((vertebra_num - 5) / 2)
        end = start + 5

        # 将冠状面中心位置和yolo结果一起加进location列表里
        # mix数组在需要切除小翅膀的时候需要用到
        # mix = np.zeros((vertebra_num, 2), dtype=int)
        location = np.zeros((vertebra_num, 3), dtype=int)

        for i in range(vertebra_num):
            # mix[i][0] = int(round((res[i][0] - res[i][2] / 2) * pir_img.shape[0]))
            # mix[i][1] = int(round((res[i][0] + res[i][2] / 2) * pir_img.shape[0]))
            location[i][0] = int(round(res[i][0] * pir_img.shape[0]))
            location[i][1] = int(round(res[i][1] * pir_img.shape[1]))
            location[i][2] = index_center

        print("计算之后得到的每个椎体质心坐标：")
        print(location)

        # 使用估计位置的方法分割单个锥体
        total_single_mask = np.zeros((vertebra_num, pir_img.shape[0], pir_img.shape[1], pir_img.shape[2]))

        for i in range(vertebra_num):
            # 通过位置信息获得单个椎体掩膜
            total_single_mask[i] = per_location_refiner_segmentor(location[i][0], location[i][1], location[i][2], pir_img, models["seg_individual"])[1]

            # 保存单个脊椎掩膜
            write_result_to_file(total_single_mask[i], ori_orient_code, ori_spacing, ori_size, ori_aff, "results/single_msk/" + nii_name[9:-7], nii_name[9:-7] + "_" + str(i).zfill(2))

        # 将单个掩膜结果处理成识别需要的包围盒尺寸
        save_dir = "./results/npy"
        gen(nii_name=nii_name, save_img=save_dir, output_shape=(64, 64, 32), start=start, end=end)

        # 通过五个单个椎体掩膜获得
        res = get_classification(nii_name=nii_name, start=start, end=end)
        res = np.array(res)
        res = get_whole_res(vertebra_num, res, start, end)

        print("最终的识别结果是：", end='')
        for i in range(vertebra_num):
            print(res_list[int(res[i])], end=' ')
        print()

        # 将结果赋值给单个掩膜
        final_mask = np.zeros(pir_img.shape)

        for i in range(vertebra_num):
            # 由于训练的时候脊柱的标签是从0开始，所以保存结果的时候要加1
            total_single_mask[i] *= int(res[i] + 1)
            tmp = total_single_mask[i] != 0
            final_mask = final_mask * ~tmp + total_single_mask[i] * tmp

        write_result_to_file(final_mask, ori_orient_code, ori_spacing, ori_size, ori_aff, "results/final_mask/" + nii_name[9:-7], nii_name[9:-7] + "_final")

        total_end = time.time()
        print("所有的步骤花费时间为：{}".format(total_end - total_start))


