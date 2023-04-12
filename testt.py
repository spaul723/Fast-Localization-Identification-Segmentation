import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from new_classification import get_classification
from process_data import gen
from test_identify import get_whole_res
from yolov7.detect.detect_full_spine import detect
#
# matplotlib.use('TKAgg')
# test1 = np.load("./results/npy/CTJ233211_05.npy")
# test2 = np.load("./results/npy/verse613_14.npy")
# print(test1.mean())
# print(test2.mean())
#
# test1 = test1[48]
# test2 = test2[48]
# print(test1.shape)
# print(test2.shape)
# plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(test1)
# plt.subplot(1,2,2)
# plt.imshow(test2)
# plt.show()


vertebra_num = 15
start = 5
end = 10

# 通过五个单个椎体掩膜获得
res = get_classification()
res = np.array(res)
res = get_whole_res(vertebra_num, res, start, end)
