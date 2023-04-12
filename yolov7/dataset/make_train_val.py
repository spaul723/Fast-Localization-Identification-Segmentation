import os
import shutil
import random

#多个数据共同训练
# all_org_paths=['/home/wjx/yolov7_pytorch/dataset/all','/home/wjx/yolov7_pytorch/dataset/full_spine_img_all']
# sub_dir='full_spine'


all_org_paths=['/home/wjx/yolov7_pytorch/dataset/full_spine_img_all']
sub_dir='only_full_spine'

tar_img_pth=f'/home/wjx/yolov7_pytorch/dataset/{sub_dir}/images'
tar_label_pth=f'/home/wjx/yolov7_pytorch/dataset/{sub_dir}/labels'
train_ratio=0.95 #训练集比例

if os.path.exists(tar_img_pth):
    shutil.rmtree(tar_img_pth)
if os.path.exists(tar_label_pth):
    shutil.rmtree(tar_label_pth)


for org_path in all_org_paths:
    img_pth=f'{org_path}/img'
    img_list=os.listdir(img_pth)
    img_list.sort()
    label_pth=f'{org_path}/label'
    label_list=os.listdir(label_pth)
    label_list.sort()
    label_list.remove('classes.txt')

    print(len(img_list))
    print(len(label_list))

    n=len(img_list)

    train_num=int(n*train_ratio)
    print('train_num:',train_num)

    train_list=[]
    val_list=[]
    # 随机数生成训练集和测试集
    for i in range(len(img_list)):
        p=random.random()
        print(p)
        if p>=train_ratio:
            val_list.append(i)
        else:
            train_list.append(i)

    print(len(train_list),len(val_list))


    #训练集
    if not os.path.exists(f'{tar_img_pth}/train'):
        os.makedirs(f'{tar_img_pth}/train')
    if not os.path.exists(f'{tar_label_pth}/train'):
        os.makedirs(f'{tar_label_pth}/train')
    shutil.copy(f'{label_pth}/classes.txt', f'{tar_label_pth}/train/classes.txt')
    for i in train_list:
        shutil.copy(f'{img_pth}/{img_list[i]}',f'{tar_img_pth}/train/{img_list[i]}')
        shutil.copy(f'{label_pth}/{label_list[i]}', f'{tar_label_pth}/train/{label_list[i]}')

    #测试集
    if not os.path.exists(f'{tar_img_pth}/val'):
        os.makedirs(f'{tar_img_pth}/val')
    if not os.path.exists(f'{tar_label_pth}/val'):
        os.makedirs(f'{tar_label_pth}/val')
    shutil.copy(f'{label_pth}/classes.txt', f'{tar_label_pth}/val/classes.txt')
    for i in val_list:
        shutil.copy(f'{img_pth}/{img_list[i]}',f'{tar_img_pth}/val/{img_list[i]}')
        shutil.copy(f'{label_pth}/{label_list[i]}', f'{tar_label_pth}/val/{label_list[i]}')




