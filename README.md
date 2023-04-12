# 全脊椎分割、定位、识别

此代码融合了脊椎的分割、定位、识别三个步骤，首先将脊椎二值分割得到整个脊椎掩膜， 
用整个椎体掩膜通过yolov7模型得到单节椎体的定位信息（估计的质心坐标），然后通过定位信息来预测单节椎体的分割掩膜，
最终使用一个两阶段识别网络识别出单个脊椎的分类，将识别结果赋值给单节椎体掩膜并合成整个椎体掩膜。

## 环境准备

环境所需库可以通过下面的命令获取，注意：需要Simpleitk小于2.1版本，可以安装2.0.2。

```bash
pip install -r requirements.txt

# 删除原来版本simpleitk，安装2.02版本
pip uninstall SimpleITK
pip install SimpleITK==2.02
```


## 运行程序

```bash
python test_identify.py
```

结果将被存到“results”文件夹下的一个_final_seg.nii.gz文件，该文件为带label的整体分割掩膜。在代码执行过程中会得到每个椎体的中心坐标、每个椎骨对应的分类结果以及单节椎骨掩膜。

## 更换数据

待识别的数据放置在 "sample" 文件夹下，修改运行代码中"nii_name"。
