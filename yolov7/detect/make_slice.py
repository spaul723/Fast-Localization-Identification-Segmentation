import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import cv2
from multiprocessing import Pool

def pad_crop(img,size=(640, 640),padnum=0): #填补+裁剪
    x,y=img.shape
    if x>size[0]:
        img=img[:size[0],:]
    if y>size[1]:
        k=y-size[1]
        a=int(k/2)
        b=int((k+1)/2)
        img=img[:,a:-b]

    x_pad=max(0,size[0]-x)
    x_pad_l=int(x_pad/2)
    x_pad_r=int((x_pad+1)/2)

    y_pad = max(0, size[0] - y)
    y_pad_l = int(y_pad / 2)
    y_pad_r = int((y_pad + 1) / 2)

    img=np.concatenate([np.zeros([x_pad_l,img.shape[1]])+padnum,img,np.zeros([x_pad_r,img.shape[1]])+padnum],axis=0)
    img = np.concatenate([np.zeros([img.shape[0],y_pad_l])+padnum, img, np.zeros([img.shape[0],y_pad_r])+padnum], axis=1)
    return img


def mang(rd):
    ### 对切片继续处理
    slice, spth ,size,fin_size,pad_num,pad=rd  #这个size是xy轴反的，因为cv2.resize就是反的
    # print(size)
    slice = np.flipud(slice) #上下翻转 sitk读出来的numpy是上下颠倒的
    img = cv2.resize(slice, size) #插值为size高宽
    # img=slice
    # print(img.shape)
    # if pad:
    #     img=pad_crop(img,fin_size,pad_num)

    plt.imsave(spth, img, cmap=cm.gray) #储存

def make_slice(nii_name,
               pth='/home/wjx/yolov7_pytorch/detect/data/abnormal', #读取路径 储存nii的文件夹
               save_path='/home/wjx/yolov7_pytorch/detect/data/abnormal_slice', #储存路径，储存输出的切片
               sample_path='/home/wjx/yolov7_pytorch/detect/data/img_sample',
               start_slice=-100,end_slice=100,step=5,pool_core_num=32,out_shape=(640,640),pad=False):
    # nii_list=os.listdir(pth)

    if nii_name=='fk':
        nii_list = [x for x in os.listdir(os.path.join(pth))]
    else:
        nii_list=[x for x in os.listdir(os.path.join(pth)) if nii_name in x]
    print(nii_list)
    nii_list.sort()
    for p in nii_list:
        nii_pth=f'{pth}/{p}'
        print(nii_pth)
        nii = sitk.ReadImage(nii_pth) #读nii
        spacing = nii.GetSpacing()
        print(spacing)
        nv=spacing[2]/spacing[1]
        print(nv)

        nii = sitk.GetArrayFromImage(nii) #转为numpy array
        print(nii.shape)
        print(np.min(nii),np.max(nii))
        pad_num=np.min(nii)

        z_size = int(nv *nii.shape[0])
        print(z_size)

        if nii.shape[1]<5 or nii.shape[2]<5:
            print('error shape')
            continue
        a,b,c = nii.shape

        path_dir=f"{save_path}/all"
        os.makedirs(path_dir,exist_ok=True)

        slice=[]

        os.makedirs(sample_path,exist_ok=True)

        spth=f"{sample_path}/{p.split('.')[0]}.png"
        mang([nii[:, :, int(c/2)],spth,(nii.shape[1],z_size),out_shape,pad_num,pad])

        stt=np.max([0,int(c/2+start_slice)])
        edd=np.min([c,int(c/2+end_slice)])
        for i in range(stt,edd,step):  #取中间200片切片
            spth = f"{path_dir}/{p.split('.')[0]}_{i}.png"
            slice.append([nii[:, :, i],spth,(nii.shape[1],z_size),out_shape,pad_num,pad])

        ### 并行串行二选一
        ### 并行
        pool = Pool(pool_core_num)
        pool.map(mang, slice)
        pool.close()
        pool.join()

        ### 串行
        # for i in slice:
        #     mang(i)

if __name__=='__main__':
    make_slice(pad=False)