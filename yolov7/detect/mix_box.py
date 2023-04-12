import os
import numpy as np
from sklearn.cluster import DBSCAN
import shutil

def xyxy2xywh(x, size=None):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    if size is None:
        size = [640, 640]
    w, h = size
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    y[:, 0] = y[:, 0] / w
    y[:, 1] = y[:, 1] / h
    y[:, 2] = y[:, 2] / w
    y[:, 3] = y[:, 3] / h
    return y

def xywh2xyxy(x, size=None):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    if size is None:
        size = [640, 640]
    w,h=size
    y = np.copy(x)
    y[:, 0] = np.round(w * (x[:, 0] - x[:, 2] / 2)) # top left x
    y[:, 1] = np.round(h * (x[:, 1] - x[:, 3] / 2))  # top left y
    y[:, 2] = np.round(w * (x[:, 0] + x[:, 2] / 2))  # bottom right x
    y[:, 3] = np.round(h * (x[:, 1] + x[:, 3] / 2))  # bottom right y
    return y

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

def combine(load_path='/home/wjx/yolov7_pytorch/detect/spine_label/classified_labels',
            save_path='/home/wjx/yolov7_pytorch/detect/spine_label/combined_labels'):
    # 把一个nii的不同切片的box合起来到一个txt中
    tar_list=os.listdir(load_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    out = open(f'{save_path}/classes.txt', mode='w')
    out.writelines(f'spine\n')
    out.close()

    for nii in tar_list:
        print(nii)
        out = open(f'{save_path}/{nii}.txt', mode='w')
        txt_list = os.listdir(f'{load_path}/{nii}')
        for txt in txt_list:
            print(txt)
            pth=f'{load_path}/{nii}/{txt}'
            file = open(pth, 'r')
            content = file.read()
            out.writelines(content)

            file.close()
        out.close()

#自定义box类，用于排序
class Box:
    def __init__(self,x, y, w, h):
        self.x,self.y,self.w,self.h=x, y, w, h
    # 重载print
    def __str__(self):
        return ("x={} y={} w={} h={}".format(self.x,self.y,self.w,self.h))
    # 重载 < 符号用于sort
    def __lt__(self, other):
        return self.y < other.y

def classify_box(load_path='/home/wjx/yolov7_pytorch/detect/spine_label/combined_labels',
                 save_path='/home/wjx/yolov7_pytorch/detect/spine_label/classified_labels'):
    ### 按坐标手分,没写完，暂时放置不用
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    txt_list = os.listdir(load_path)
    for txt in txt_list:
        print(txt)
        pth = f'{load_path}/{txt}'
        file = open(pth, 'r')
        content = file.read()

        content = content.split('\n')
        content.remove('')

        all_box=[]
        for coord in content:
            ax = coord.split(' ')
            box = []
            for i in range(1, len(ax)):
                box.append(float(ax[i]))
            x, y, w, h = box
            all_box.append(Box(x, y, w, h))
        all_box.sort()
        # for i in all_box:
        #     print(i)
        skip_dis=[]
        for i in range(1,len(all_box)):
            skip_dis.append(all_box[i].y - all_box[i - 1].y)

        flag_skip=0
        skip_dis.sort()
        print(skip_dis)
        for i in range(1,len(skip_dis)):
            if skip_dis[i]>skip_dis[i-1]*5 and skip_dis[i-1]!=0:
                flag_skip=skip_dis[i]
                print(skip_dis[i],len(skip_dis)-i)
                print()
                break

        classified_box=[[]]
        for i in range(1,len(all_box)):
            if all_box[i].y-all_box[i-1].y<flag_skip:
                classified_box[-1].append(all_box[i])
            else:
                classified_box.append([])

        print(classified_box)
        for i in classified_box:
            for j in i:
                print(j)
            print()

        file.close()


def classify_box_DBSCAN(load_path='/home/wjx/yolov7_pytorch/detect/spine_label/combined_labels',
                        save_cluster_path='/home/wjx/yolov7_pytorch/detect/spine_label/clustered_labels', #聚类后的路径
                        save_fin_path='/home/wjx/yolov7_pytorch/detect/spine_label/fin_labels', #合并后的路径
                        count_percent=3, #一个类最少的box数量占最多的几分之一
                        eps_center=0.03, # 中心坐标聚类参数
                        eps_wh=0.04,  #高宽聚类参数 0.03
                        DBSCAN_min_samples=2):   #聚类参数
    ### 用密度聚类分类
    if not os.path.exists(save_cluster_path):
        os.makedirs(save_cluster_path)
    if not os.path.exists(save_fin_path):
        os.makedirs(save_fin_path)

    out = open(f'{save_cluster_path}/classes.txt', mode='w')
    for i in range(50):
        out.writelines(f'{i}\n')
    out.writelines(f'noise\n')  # 50 噪音
    out.writelines(f'wh_problem\n')  # 51 wh筛掉的
    out.close()

    out = open(f'{save_fin_path}/classes.txt', mode='w')
    out.writelines(f'spine\n')
    out.close()

    txt_list = os.listdir(load_path)
    txt_list.sort()
    if 'classes.txt' in txt_list:
        txt_list.remove('classes.txt')
    print(txt_list)
    ans=[]
    for txt in txt_list:
        print(txt)
        pth = f'{load_path}/{txt}'
        file = open(pth, 'r')
        content = file.read()

        content = content.split('\n')
        content.remove('')

        all_box=[]
        all_center=[]
        wh=[]
        for coord in content:
            ax = coord.split(' ')
            box = []
            for i in range(1, len(ax)):
                box.append(float(ax[i]))
            x, y, w, h = box
            # print(np.asarray([x, y, w, h]))
            all_box.append(np.asarray([x, y, w, h]))
            all_center.append(np.asarray([x, y]))
            wh.append(np.asarray([w,h]))  #长宽聚类用来剔除问题分割框

        # 先使用中心聚类，然后在同一个类别中使用长宽聚类
        all_box=np.asarray(all_box)
        all_center=np.asarray(all_center)
        wh=np.asarray(wh)

        # 中心聚类
        cluster=DBSCAN(eps=eps_center, min_samples=DBSCAN_min_samples).fit(all_center) #0.02 0.03 0.02能完全分开，就是有点碎
        box_class=cluster.labels_
        # print('box class:',box_class)
        print(max(box_class))

        # 聚完后筛长宽
        boxwh_clustered_classified=[]
        rank=[] #记录下标，以可以直接在all_box内删改
        for i in range(np.max(box_class)+1):
            boxwh_clustered_classified.append([])
            rank.append([])

            #下标记录
            rkin=np.asarray(np.where(box_class == i))
            rkin=rkin.flatten()

            # 拿出分完的长宽
            kpt= wh[box_class == i, :]

            for j in range(kpt.shape[0]):
                boxwh_clustered_classified[-1].append(kpt[j])
                rank[-1].append(rkin[j])
            boxwh_clustered_classified[-1]=np.asarray(boxwh_clustered_classified[-1])

        # 再把聚好的里面，把长宽不行的扔掉
        print(len(boxwh_clustered_classified))
        for kt in range(len(boxwh_clustered_classified)):
            t=boxwh_clustered_classified[kt]
            tmp_cluster = DBSCAN(eps=eps_wh, min_samples=DBSCAN_min_samples).fit(t).labels_
            # kpk=np.unique(tmp_cluster)
            # print(kpk)
            # print('tol:',tmp_cluster.shape)
            # for i in kpk:
            #     print(sum(tmp_cluster==i))

            import copy
            tmp=copy.deepcopy(tmp_cluster)
            tmp[tmp==-1]=100
            main_elm=np.argmax(np.bincount(tmp)) #出现次数最多的，为主box
            if main_elm==100:
                main_elm=-1
            # print(main_elm)
            # print()

            for i in range(tmp_cluster.shape[0]):
                if tmp_cluster[i]!=main_elm:
                    box_class[rank[kt][i]]=51

        # 储存分好类的box
        pth=f'{save_cluster_path}/{txt}'
        out = open(pth, mode='w')
        for i in range(len(all_box)):
            if box_class[i]==-1:
                kk=50
            else:
                kk=box_class[i]
            out.writelines(f'{kk} {all_box[i,0]} {all_box[i,1]} {all_box[i,2]} {all_box[i,3]}\n')
        out.close()

        # 聚全部重新分类合并
        box_clustered_classified=[]
        rank=[] #记录下标，以可以直接在all_box内删改
        # print('max(box_class)',max(box_class))
        for i in range(np.max(box_class)+1):
            box_clustered_classified.append([])
            rank.append([])

            #下标记录
            rkin=np.asarray(np.where(box_class == i))
            rkin=rkin.flatten()

            # 拿出分完的box
            kpt= all_box[box_class == i, :]

            for j in range(kpt.shape[0]):
                box_clustered_classified[-1].append(kpt[j])
                rank[-1].append(rkin[j])
            box_clustered_classified[-1]=np.asarray(box_clustered_classified[-1])

        maxx=-1
        print(len(box_clustered_classified))
        for i in range(len(box_clustered_classified)):
            if i >=50:
                break
            maxx=max(maxx,len(box_clustered_classified[i]))
        print('maxx:',maxx)
        if maxx==-1:
            print('none class error')
            continue
        # if min_box_num==-1:
        min_box_num=int(maxx/count_percent+1)
        print('min_box_num:',min_box_num)

        fin_label=[]
        ### 合并+储存
        print('len:')
        for i in range(len(box_clustered_classified)):
            print((len(box_clustered_classified[i])),end=' ')
            if len(box_clustered_classified[i])<min_box_num: #小于min_box_num的类直接不要
                continue
            if i>=50: #噪点和wh筛除的不要
                continue
            xt=box_clustered_classified[i]
            xt=np.asarray(xt)
            # print(xt)
            xyxy=xywh2xyxy(xt,[640,640])
            # print(xyxy)
            xyxy_fin=[np.min(xyxy[:,0]),np.min(xyxy[:,1]),np.max(xyxy[:,2]),np.max(xyxy[:,3])]
            fin_label.append(xyxy_fin)
        # print(fin_label)
        fin_label=np.asarray(fin_label)
        fin_label=xyxy2xywh(fin_label,[640,640])
        # print(fin_label)
        pth = f'{save_fin_path}/{txt}'
        out = open(pth, mode='w')
        fin_res = []
        for i in range(fin_label.shape[0]):
            _res = []
            for j in range(4):
                _res.append(fin_label[i, j])
            fin_res.append(_res)
            out.writelines(f'0 {fin_label[i,0]} {fin_label[i,1]} {fin_label[i,2]} {fin_label[i,3]}\n')
        ans.append([txt,fin_res])
        file.close()

        global ans_all_box
        ans_all_box = []
        for fk in ans:
            for fkk in fk:
                if '.txt' in fkk:
                    continue
                for coord in fkk:
                    x, y, w, h = coord
                    ans_all_box.append(Box(x, y, w, h))
        ans_all_box.sort()
    print()

    fin_ans=[]
    for fkkk in ans:
        tmp=fkkk[0]
        ttemp = []
        for k in ans_all_box:
            ttemp.append([k.x,k.y,k.w,k.h])
        fin_ans.append(tmp)
        fin_ans.append(ttemp)
    return [fin_ans]

def remove_top(load_path='/home/wjx/yolov7_pytorch/detect/spine_label/fin_labels',
               save_path='/home/wjx/yolov7_pytorch/detect/spine_label/fin_removed_labels',
               h_rate=0.7, #最上面box筛除的最低百分比
               min_y=5 #最上面box移除时top y的最小像素
               ):
    #把最顶上的不全的box移除
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    out = open(f'{save_path}/classes.txt', mode='w')
    out.writelines(f'spine\n')
    out.close()
    txt_list = os.listdir(load_path)
    txt_list.sort()
    if 'classes.txt' in txt_list:
        txt_list.remove('classes.txt')
    for txt in txt_list:
        print(txt)
        pth = f'{load_path}/{txt}'
        file = open(pth, 'r')
        content = file.read()

        content = content.split('\n')
        content.remove('')

        all_box=[]
        for coord in content:
            ax = coord.split(' ')
            box = []
            for i in range(1, len(ax)):
                box.append(float(ax[i]))
            x, y, w, h = box
            all_box.append(Box(x, y, w, h))
        all_box.sort()
        box_c=[]
        for i in all_box:
            box_c.append([i.x,i.y,i.w,i.h])
            # print(i)
        box_c=np.asarray(box_c)
        removed_fin = box_c
        box_array=xywh2xyxy(box_c)
        # print(box_array.shape)
        # print(box_array)

        pth = f'{save_path}/{txt}'
        out = open(pth, mode='w')

        if box_array.shape[0]<=1:
            for i in range(removed_fin.shape[0]):
                out.writelines(f'0 {removed_fin[i, 0]} {removed_fin[i, 1]} {removed_fin[i, 2]} {removed_fin[i, 3]}\n')
            continue

        if box_array[0,1]<=min_y:  #筛选最上面的box
            avg_h=0
            cnt=0
            for i in range(1,max(2,int(box_array.shape[0]/2))):
                # print(box_array[i,:])
                # print(box_array[i,3]-box_array[i,1])
                avg_h+=box_array[i,3]-box_array[i,1]
                cnt+=1
            avg_h/=cnt
            # print(box_array[0,3]-box_array[0,1])
            # print(avg_h)
            if box_array[0,3]-box_array[0,1]<avg_h*h_rate:
                print('delete box')
                removed_fin=removed_fin[1:,:]

        for i in range(removed_fin.shape[0]):
            out.writelines(f'0 {removed_fin[i, 0]} {removed_fin[i, 1]} {removed_fin[i, 2]} {removed_fin[i, 3]}\n')
        out.close()

        file.close()



if __name__=='__main__':
    ## 在之前先运行make_abnormal_slice 生成nii切片，然后运行detect.py，结果储存在/home/wjx/yolov7_pytorch/detect/内的spine_label内
    ## 即detect.py参数增加  --project /home/wjx/yolov7_pytorch/detect --name spine_label
    ## 然后改参数里的路径就行了

    label_dir='spine_label'

    classify(load_path=f'/home/wjx/yolov7_pytorch/detect/{label_dir}/labels',
             save_path=f'/home/wjx/yolov7_pytorch/detect/{label_dir}/classified_labels')

    combine(load_path=f'/home/wjx/yolov7_pytorch/detect/{label_dir}/classified_labels',
            save_path=f'/home/wjx/yolov7_pytorch/detect/{label_dir}/combined_labels')

    classify_box_DBSCAN(load_path=f'/home/wjx/yolov7_pytorch/detect/{label_dir}/combined_labels',
                        save_cluster_path=f'/home/wjx/yolov7_pytorch/detect/{label_dir}/clustered_labels',  # 聚类后的路径
                        save_fin_path=f'/home/wjx/yolov7_pytorch/detect/{label_dir}/fin_labels')  # 合并后的路径

    remove_top(load_path=f'/home/wjx/yolov7_pytorch/detect/{label_dir}/fin_labels',
                save_path=f'/home/wjx/yolov7_pytorch/detect/{label_dir}/fin_removed_labels')
