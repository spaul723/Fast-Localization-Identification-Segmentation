# Fast spine localization, identification and segmentation

This code integrates the three steps of segmentation, localization and identification of the spine. Firstly, binary segmentation of the spine is performed to get the whole vertebral mask.

The positioning information of the single vertebral body (estimate dcenter-of-mass coordinate) was obtained by the yolov7 model with the mask of the whole vertebral body, and then the segmentation mask of the single vertebral body was predicted by the positioning information.

Finally, a fast recognition network was used to identify the classification of individual vertebrae, assign the recognition results to the single vertebral mask and synthesize the entire vertebral mask.

## Environmental preparation

The libraries required by the environment can be obtained by using the following command, noting that SimpleITK is less than 2.1, so 2.0.2 can be installed.

```bash
pip install -r requirements.txt

# Uninstall the original version of SimpleITK and install version 2.0.2
pip uninstall SimpleITK
pip install SimpleITK==2.0.2
```

## Model download

We have trained all the relevant models, and saved the model parameters, stored in the following Baidu cloud disk link, please download and store according to the given path. 

```txt
https://pan.baidu.com/s/1gFwo8osB6w5N0f97t9LVjg?pwd=qeb2 
```

Of course, you can also retrain the model yourself, but the training code is not given here, you can contact me directly through `spaul0723@163.com`.

## Data preparation

All the data to be identified will be placed in the folder `sample`.

## Run program

```bash
python test_identify.py
```

The results will be stored in `results`, where `single_mask` and `npy` store the mask of the single vertebra and the identification network input data, and the `final_mask` stores the final labeled entire spine mask. In the process of code execution, the central coordinates of each vertebra, the corresponding classification results of each vertebra and the single vertebra mask will be obtained. 

