import os
import shutil
root_path='/home/wjx/yolov7_pytorch/dataset/all_nii'
nii_dir=os.listdir(root_path)

save_path='/home/wjx/yolov7_pytorch/dataset/all_nii/all'
if not os.path.exists(save_path):
    os.makedirs(save_path)

tmp_pth=[root_path + '/' + i for i in nii_dir]
if '/home/wjx/yolov7_pytorch/dataset/all_nii/all' in tmp_pth:
    tmp_pth.remove('/home/wjx/yolov7_pytorch/dataset/all_nii/all')
print(tmp_pth)
nii_pth=[]
for i in tmp_pth:
    kk=os.listdir(i)
    if kk[0].find('.nii.gz')==-1:
        p=os.listdir(i)
        for j in p:
            nii_pth.append(f'{i}/{j}')
    else:
        nii_pth.append(i)
print(nii_pth)

for pth in nii_pth:
    nii=os.listdir(pth)
    for i in nii:
        source_pth=f'{pth}/{i}'
        print(source_pth)
        target_pth=f'{save_path}/{i}'
        print(target_pth)
        shutil.copy(source_pth,target_pth)