
import sys
sys.path.append('..')
import numpy as np
from radon_utils import (create_sinogram, bp, filter_op,
                         fbp, reade_ima, write_img, sinogram_2c_to_img,
                         padding_img, unpadding_img, indicate)

img_List = np.load('./batch_result/123.npy')
#img_List = np.load('./Test_CT/batch_img(1).npy')
print(img_List.shape)

picNum = img_List.shape[0]

for picNO in range(picNum):
    print(f"picNO is : {picNO}")

    img = img_List[picNO,...]
    print(img.shape)
    write_img(img, 'tph'+str(picNO)+'.png')
