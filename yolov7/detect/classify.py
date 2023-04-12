import os
import shutil

def classify(load_path='/home/wjx/yolov7_pytorch/detect/spine_label/labels',
            save_path='/home/wjx/yolov7_pytorch/detect/spine_label/classified_labels'):
    txt_list=os.listdir(load_path)
    txt_list.sort()

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in txt_list:
        k=i.split('_')[0]
        if not os.path.exists(f'{save_path}/{k}'):
            os.makedirs(f'{save_path}/{k}')
        shutil.copy(f'{load_path}/{i}',f'{save_path}/{k}/{i}')

if __name__=='__main__':
    classify()
