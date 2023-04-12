__copyright__ = \
"""
Copyright &copyright © (c) 2021 Inria Grenoble Rhône-Alpes.
All rights reserved.

This source code is to be used for academic research purposes only.
For commercial uses of the code, please send an email to edmond.boyer@inria.fr and sergi.pujades@inria.fr

"""
__license__ = "CC BY-NC-SA 4.0"
__authors__ = "Di Meng"



def vol_padding(vol, cube_size, front_after_pad_size=0):
    import numpy as np 

    pad_value = -1000

    h,w,c = vol.shape

    padX = [front_after_pad_size, front_after_pad_size]
    padY = [front_after_pad_size, front_after_pad_size]
    padZ = [front_after_pad_size, front_after_pad_size]


    if h < cube_size:
        padX[1] += cube_size - h 
    elif h % cube_size != 0:
        padX[1] += cube_size - (h % cube_size)
    else:
        pass

    if w < cube_size:
        padY[1] += cube_size - w
    elif w % cube_size != 0:
        padY[1] += cube_size - (w % cube_size)
    else:
        pass

    if c < cube_size:
        padZ[1] += cube_size - c
    elif c % cube_size != 0:
        padZ[1] += cube_size - (c % cube_size)
    else:
        pass

    vol = np.pad(vol, [padX, padY, padZ], mode='constant', constant_values=pad_value)

    assert min(vol.shape) >= cube_size

    return vol 



def load_spine_segmentor(model_file):
    import torch 
    from segmentor import Unet3D_attention
    model = Unet3D_attention(in_channels=1, out_channels=1, 
                             activation2='sigmoid', feature_maps=[16, 32, 64, 128, 256])

    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    if torch.cuda.is_available():
        model.to(torch.device("cuda"))
    model.eval()

    return model 



def toTenser(x):
    import torch

    assert len(x.shape) == 3

    out = torch.from_numpy(x).view(1, 1, x.shape[-3], x.shape[-2], x.shape[-1])
    out = out.to(dtype=torch.float32)

    if torch.cuda.is_available(): out = out.cuda() 

    return out


def toNumpy(x):
    import numpy as np 

    out = x.view(x.size(-3), x.size(-2), x.size(-1))
    out = out.cpu()
    out = out.detach().numpy().astype(np.float32)

    return out 


def model_apply(input_data, model, multi_channel):
    import torch

    input_tensor = toTenser(input_data)

    output_tensor = model(input_tensor)

    if multi_channel:
        output_tensor = torch.argmax(output_tensor, dim=1)

    output_array = toNumpy(output_tensor)

    # output_array[output_array>0.5] = 1
    # output_array[output_array<=0.5] = 0

    return output_array


def vol2binary_overlap(vol, model, norm):
    import numpy as np 

    cube_size = 96
    stride = 48

    front_after_pad_size = 24

    ori_h, ori_w, ori_c = vol.shape

    vol = vol_padding(vol, cube_size, front_after_pad_size)

    if norm:
        from utils import globalNormalization
        vol = globalNormalization(vol)

    h, w, c = vol.shape

    vol_out = np.zeros(vol.shape)
    idx_vol = np.zeros(vol.shape)

    for i in range(0, h, stride):
        for j in range(0, w, stride):
            for k in range(0, c, stride):

                if i+cube_size > h or j+cube_size > w or k+cube_size > c:
                    continue
                    
                cube = vol[i:i+cube_size, j:j+cube_size, k:k+cube_size]
                cube_out = model_apply(cube, model, multi_channel=False)
                vol_out[i:i+cube_size, j:j+cube_size, k:k+cube_size] += cube_out
                idx_vol[i:i+cube_size, j:j+cube_size, k:k+cube_size] += 1

    assert (idx_vol > 0).all()

    vol_out = vol_out / idx_vol

    vol_output = vol_out[front_after_pad_size:front_after_pad_size+ori_h, 
                         front_after_pad_size:front_after_pad_size+ori_w, 
                         front_after_pad_size:front_after_pad_size+ori_c]

    vol_output[vol_output>0.5] = 1
    vol_output[vol_output<=0.5] = 0
    return vol_output


def vol2binary_overlap_central_votes(vol, model, norm):
    import numpy as np 

    cube_size = 96
    stride = 24

    front_after_pad_size = 24

    ori_h, ori_w, ori_c = vol.shape

    vol = vol_padding(vol, cube_size, front_after_pad_size)

    if norm:
        from utils import globalNormalization
        vol = globalNormalization(vol)

    h, w, c = vol.shape

    vol_out = np.zeros(vol.shape)
    idx_vol = np.zeros(vol.shape)

    for i in range(0, h, stride):
        for j in range(0, w, stride):
            for k in range(0, c, stride):

                if i+cube_size > h or j+cube_size > w or k+cube_size > c:
                    continue
                    
                cube = vol[i:i+cube_size, j:j+cube_size, k:k+cube_size]
                cube_out = model_apply(cube, model, multi_channel=False)

                # print('cube out max: {}, min{}'.format(cube_out.max(), cube_out.min()))

                vol_out[i+24:i+72, j+24:j+72, k:k+96] += cube_out[24:72, 24:72, :]
                vol_out[i:i+24, j+24:j+72, k+24:k+72] += cube_out[0:24, 24:72, 24:72]
                vol_out[i+72:i+96, j+24:j+72, k+24:k+72] +=  cube_out[72:96, 24:72, 24:72]
                vol_out[i+24:i+72, j:j+24, k+24:k+72] += cube_out[24:72, 0:24, 24:72]
                vol_out[i+24:i+72, j+72:j+96, k+24:k+72] += cube_out[24:72, 72:96, 24:72]

                idx_vol[i+24:i+72, j+24:j+72, k:k+96] += 1
                idx_vol[i:i+24, j+24:j+72, k+24:k+72] += 1
                idx_vol[i+72:i+96, j+24:j+72, k+24:k+72] += 1
                idx_vol[i+24:i+72, j:j+24, k+24:k+72] += 1
                idx_vol[i+24:i+72, j+72:j+96, k+24:k+72] += 1

    print('raw out max: {}, min: {}'.format(vol_out.max(), vol_out.min()))

    vol_out = divide_with_zeros(vol_out, idx_vol)

    print('after division max: {}, min: {}'.format(vol_out.max(), vol_out.min()))

    vol_out = np.nan_to_num(vol_out)
    
    vol_output = vol_out[front_after_pad_size:front_after_pad_size+ori_h, 
                         front_after_pad_size:front_after_pad_size+ori_w, 
                         front_after_pad_size:front_after_pad_size+ori_c]

    vol_output[vol_output>0.5] = 1
    vol_output[vol_output<=0.5] = 0

    print(np.unique(vol_output))

    return vol_output


def divide_with_zeros(a, b):
    import numpy as np 

    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a,b)
        c[c == np.inf] = 0
        c = np.nan_to_num(c)
    return c 


def vol2binary(vol, model, norm):

    import numpy as np 

    cube_size = 96

    vol_pad = vol_padding(vol, cube_size)

    h, w, c = vol_pad.shape
    assert h % cube_size == 0 and w % cube_size == 0 and c % cube_size == 0

    if norm: 
        from utils import globalNormalization
        vol_pad = globalNormalization(vol_pad)

    stride = cube_size
    vol_out = np.empty(vol_pad.shape)
    idx_vol = np.zeros(vol_pad.shape)

    for i in range(0, h-cube_size+1, stride):
        for j in range(0, w-cube_size+1, stride):
            for k in range(0, c-cube_size+1, stride):

                cube = vol_pad[i:i+cube_size, j:j+cube_size, k:k+cube_size]
                cube_out = model_apply(cube, model, multi_channel=False)
                vol_out[i:i+cube_size, j:j+cube_size, k:k+cube_size] = cube_out
                idx_vol[i:i+cube_size, j:j+cube_size, k:k+cube_size] += 1

    assert np.equal(idx_vol, np.ones(vol_pad.shape)).all()

    vol_output = vol_out[0:vol.shape[0], 0:vol.shape[1], 0:vol.shape[2]]

    vol_output[vol_output>0.5] = 1
    vol_output[vol_output<=0.5] = 0

    assert vol_output.shape == vol.shape

    return vol_output



def binary_segmentor(img, model_file, mode='overlap', norm=False):

    # mode : slide | overlap | central_votes

    model = load_spine_segmentor(model_file)

    if mode == 'slide':
        pred_msk = vol2binary(img, model, norm)
        return pred_msk
    elif mode == 'overlap':
        pred_msk = vol2binary_overlap(img, model, norm)
        return pred_msk
    elif mode == 'central_votes':
        pred_msk = vol2binary_overlap_central_votes(img, model, norm)
        return pred_msk
    else:
        raise NotImplementedError('Unknown mode {}.'.format(mode))


    