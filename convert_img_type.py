import numpy as np
import os
from scipy import misc


def convert():
    img_dir = 'train_masks/'
    output_dir = 'train_masks_png/'

    for i in os.listdir(img_dir):
        if i.endswith('.gif'):
            img_path = img_dir + i
            img = misc.imread(img_path)

            out_path = output_dir + i.split('.gif')[0] + '.png'
            misc.imsave(out_path, img)
            print('processed img {}'.format(img_path))



if __name__ == '__main__':
    convert()

