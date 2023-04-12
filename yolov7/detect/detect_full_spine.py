from yolov7.detect.mix_box import *
from yolov7.detect.make_slice import *
from yolov7.detect2 import instance_detct
import os
import time

def detect(nii_name):
    start = time.time()

    label_dir='full_spine_label'

    # pth=f'/home/wjx/yolov7_pytorch/dataset/full_spine_nii_test' #读取路径 储存nii的文件夹
    pth=f'./sample' #读取路径 储存nii的文件夹
    # pth=f'./yolov7/detect/data/nii'

    root_path = './yolov7/detect'


    save_path=f'{root_path}/data/full_spine_nii_slice' #储存路径，储存输出的切片
    sample_path=f'{root_path}/data/full_spine_nii_img_sample' #生成sample图片的路径（切片完要有一个展示的图片）

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    if os.path.exists(sample_path):
        shutil.rmtree(sample_path)

    start1 = time.time()

    make_slice(nii_name=nii_name,pth=pth,save_path=save_path,sample_path=sample_path,step=2,pool_core_num=8,start_slice=-60,end_slice=60)
    end1 = time.time()
    print("Making slices takes {}s.".format(end1 - start1))

    if os.path.exists(f'{root_path}/{label_dir}/'):
        print('rm path', f'{root_path}/{label_dir}/')
        shutil.rmtree(f'{root_path}/{label_dir}/')

    start2 = time.time()
    # os.system(cmd)
    instance_detct()
    end2 = time.time()
    print("yolo____________________________________")
    print(end2-start2)

    classify(load_path=f'{root_path}/{label_dir}/labels',
             save_path=f'{root_path}/{label_dir}/classified_labels')
    combine(load_path=f'{root_path}/{label_dir}/classified_labels',
            save_path=f'{root_path}/{label_dir}/combined_labels')
    res = classify_box_DBSCAN(load_path=f'{root_path}/{label_dir}/combined_labels',
                              save_cluster_path=f'{root_path}/{label_dir}/clustered_labels',  # 聚类后的路径
                              save_fin_path=f'{root_path}/{label_dir}/fin_labels',
                              count_percent=6,
                              eps_center=0.012,
                              eps_wh=0.02)  # 合并后的路径 0.02 0.04
    end = time.time()
    print(end - start)
    return res
    # remove_top(load_path=f'{root_path}/{label_dir}/fin_labels',save_path=f'{root_path}/{label_dir}/fin_removed_labels',h_rate=0.6)

if __name__ == "__main__":
    detect()