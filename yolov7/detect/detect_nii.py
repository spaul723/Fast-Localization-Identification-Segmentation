from mix_box import *
from make_slice import *
import os

root_path='/home/wjx/yolov7_pytorch/detect'
label_dir='spine_label_tst'

pth=f'/home/wjx/yolov7_pytorch/dataset/all_nii/all' #读取路径 储存nii的文件夹
save_path=f'{root_path}/data/all_nii_slice' #储存路径，储存输出的切片
sample_path=f'{root_path}/data/all_nii_img_sample' #生成sample图片的路径（切片完要有一个展示的图片）

if os.path.exists(save_path):
    shutil.rmtree(save_path)
if os.path.exists(sample_path):
    shutil.rmtree(sample_path)

make_slice(pth=pth,save_path=save_path,sample_path=sample_path,step=10,pool_core_num=32)

if os.path.exists(f'{root_path}/{label_dir}/'):
    print('rm path', f'{root_path}/{label_dir}/')
    shutil.rmtree(f'{root_path}/{label_dir}/')

gpuids='3' #选用的显卡编号，他这个detect后续有点问题，要先设定一下CUDA_VISIBLE_DEVICES才行
weights_dir='spine2' #权重的目录

args=f'--weights /home/wjx/yolov7_pytorch/runs/train/{weights_dir}/weights/best.pt ' \
     f'--save-txt --device {gpuids} --conf-thres 0.7 --iou-thres 0.6 ' \
     f'--source {save_path}/all --project {root_path} --name {label_dir}' \
     ' --nosave'
#这个cmd也要改一下，把/home/wjx/.conda/envs/torch/bin/python3.9改成你自己的python路径，后面这个detect.py改为你自己的路径
# cmd = 'CUDA_VISIBLE_DEVICES={} /home/wjx/.conda/envs/torch/bin/python3.9 {} {}'.format(gpuids,'/home/wjx/yolov7_pytorch/detect.py',args)
cmd = '/home/wjx/.conda/envs/torch/bin/python3.9 {} {}'.format('/home/wjx/yolov7_pytorch/detect.py',args)
print(cmd)
os.system(cmd)

classify(load_path=f'{root_path}/{label_dir}/labels',
         save_path=f'{root_path}/{label_dir}/classified_labels')
combine(load_path=f'{root_path}/{label_dir}/classified_labels',
        save_path=f'{root_path}/{label_dir}/combined_labels')
classify_box_DBSCAN(load_path=f'{root_path}/{label_dir}/combined_labels',
                    save_cluster_path=f'{root_path}/{label_dir}/clustered_labels',  # 聚类后的路径
                    save_fin_path=f'{root_path}/{label_dir}/fin_labels')  # 合并后的路径
remove_top(load_path=f'{root_path}/{label_dir}/fin_labels',save_path=f'{root_path}/{label_dir}/fin_removed_labels',h_rate=0.7)
