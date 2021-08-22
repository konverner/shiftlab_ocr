import io, cv2, os, torch
import numpy as np
from os.path import join
import random

# RESIZE AND NORMALIZE IMAGE
def process_image(img):
    '''
    params:
    ---
    img : np.array
    returns
    ---
    img : np.array
    '''
    w, h, _ = img.shape
    new_w = 64
    new_h = int(h * (new_w / w))
    img = cv2.resize(img, (new_h, new_w))
    w, h, _ = img.shape

    img = img.astype('float32')

    new_h = 256
    if h < new_h:
        add_zeros = np.full((w, new_h - h, 3), 255)
        img = np.concatenate((img, add_zeros), axis=1)

    if h > new_h:
        img = cv2.resize(img, (new_h, new_w))

    return img

# TRANSLATE INDICIES TO TEXT
def labels_to_text(s, idx2char):
    '''
    paramters
    ---
    idx2char : dict
        keys : int
            indicies of characters
        values : str
            characters
    returns
    ---
    S : str
    '''
    S = "".join([idx2char[i] for i in s])
    if S.find('EOS') == -1:
        return S
    else:
        return S[:S.find('EOS')]
