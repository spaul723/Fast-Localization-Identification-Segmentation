__copyright__ = \
"""
Copyright &copyright © (c) 2021 Inria Grenoble Rhône-Alpes.
All rights reserved.

This source code is to be used for academic research purposes only.
For commercial uses of the code, please send an email to edmond.boyer@inria.fr and sergi.pujades@inria.fr

"""
__license__ = "CC BY-NC-SA 4.0"
__authors__ = "Di Meng"

import matplotlib.pyplot as plt


def load_vertebra_segmentor(model_file):
    import torch 
    from segmentor import Unet3D_attention
    model = Unet3D_attention(in_channels=2, out_channels=1, activation2='sigmoid', 
                                feature_maps=[16, 32, 64, 128, 256])

    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    if torch.cuda.is_available():
        model.to(torch.device("cuda"))
    model.eval()

    return model 

def centroid_translate(img, cube_size, x_shift=20):
    import numpy as np 

    img_copy = np.copy(img)

    cube_size_x, cube_size_y, cube_size_z = cube_size[:]

    x_trans = cube_size_x//2
    y_trans = cube_size_y//2
    z_trans = cube_size_z//2

    img_copy = np.pad(img_copy, [(cube_size_x//2, cube_size_x//2+x_shift), (cube_size_y//2, cube_size_y//2), (cube_size_z//2, cube_size_z//2)], mode='constant', constant_values=img.min())

    return img_copy, x_trans, y_trans, z_trans


def create_centermap(cube_size, sigma, x_shift=20):

    import numpy as np 

    cube_size_x, cube_size_y, cube_size_z = cube_size[:]

    grid_x, grid_y, grid_z = np.mgrid[0:cube_size_x, 0:cube_size_y, 0:cube_size_z]
    D2 = (grid_x- cube_size_x//2 + x_shift)**2 + (grid_y - cube_size_y//2)**2 + (grid_z - cube_size_z//2)**2

    return np.exp(-D2/2.0/sigma/sigma)


def toTensor(x):
    import torch

    assert len(x.shape) == 3

    x = torch.from_numpy(x).to(dtype=torch.float32)
    out = x.view(1, 1, x.size(0), x.size(1), x.size(2))
    if torch.cuda.is_available(): out = out.cuda()

    return out


def toNumpy(x):
    import numpy as np

    out = x.view(x.size(-3), x.size(-2), x.size(-1))
    out = out.cpu()
    out = out.detach().numpy().astype(np.float32)

    return out 


def model_apply(x, model):
    import torch

    cube_size = (128, 128, 128)
    sigma = 20
    centermap = create_centermap(cube_size, sigma)

    centermap = toTensor(centermap)
    x = toTensor(x)

    input_ = torch.cat((x, centermap), 1)

    pred = model(input_)
    out = toNumpy(pred)

    # print('out min:{}, max: {}'.format(out.min(), out.max()))

    out[out>0.5] = 1
    out[out<=0.5] = 0

    return out 


def centroid_from_labeled_volume(vol, label=1):
    import numpy as np 
    from scipy.ndimage.measurements import center_of_mass

    vol_copy = (vol==label)

    segmented_size = np.sum(vol_copy)
    print('computing center of mass, segmented size: ',segmented_size)

    if segmented_size == 0:
        import warnings
        warnings.warn('empty segmented volume. ')
        return None 

    ctd = center_of_mass(vol_copy)

    return np.round(ctd).astype(np.int)


def per_location_refiner_segmentor(x, y, z, pir_img, model_file, norm=False):
    import numpy as np 

    cube_size = (128, 128, 128)
    x_shift = 20

    cube_size_x, cube_size_y, cube_size_z = cube_size[:]

    img_pad, x_trans, y_trans, z_trans = centroid_translate(pir_img, cube_size, x_shift)

    vol_out = np.zeros(img_pad.shape)

    model = load_vertebra_segmentor(model_file)

    x,y,z = int(round(x)+x_trans), int(round(y)+y_trans), int(round(z)+z_trans)

    img_cube = img_pad[x-cube_size_x//2+x_shift:x+cube_size_x//2+x_shift, y-cube_size_y//2:y+cube_size_y//2, \
                        z-cube_size_z//2:z+cube_size_z//2]


    if norm:    
        from utils import globalNormalization
        img_cube = globalNormalization(img_cube)


    cube_out = model_apply(img_cube, model)

    vol_out[x-cube_size_x//2+x_shift:x+cube_size_x//2+x_shift, y-cube_size_y//2:y+cube_size_y//2, \
                        z-cube_size_z//2:z+cube_size_z//2] = cube_out


    vol_output = vol_out[x_trans:x_trans+pir_img.shape[0], y_trans:y_trans+pir_img.shape[1], \
                            z_trans:z_trans+pir_img.shape[2]]


    ctd = centroid_from_labeled_volume(vol_output)
    return ctd, vol_output
    
