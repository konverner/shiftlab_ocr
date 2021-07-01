import cv2
import editdistance
import torch
import numpy as np

# RESIZE AND NORMALIZE A IMAGE
def process_image(img):
    """
    params:
    ---
    img : np.array

    returns
    ---
    img : np.array
    """
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

# LOAD A STATE OF MODEL FROM CHK_PATH
# IF CHK_PATH IS EMPTY THEN IT INITIALIZES THE STATE TO ZERO
def load_from_checkpoint(model,chk_path):
    """
    params
    ---
    model : nn.Module

    chk_path : string
        path to the .pt file

    returns
    ---
    model : nn.Module

    """
    if chk_path:
        if torch.cuda.is_available():
          ckpt = torch.load(chk_path)
        else:
          ckpt = torch.load(chk_path,  map_location=torch.device('cpu'))
        if 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
        else:
            model.load_state_dict(ckpt)
        print('weights have been loaded')
    return model

# TRANSLATE INDICIES TO TEXT
def labels_to_text(s, idx2char):
    """
    params
    ---
    idx2char : dict
        keys : int
            indicies of characters
        values : str
            characters
    returns
    ---
    S : str
    """
    S = "".join([idx2char[i] for i in s])
    if S.find('EOS') == -1:
        return S
    else:
        return S[:S.find('EOS')]


# COMPUTE CHARACTER ERROR RATE
def char_error_rate(p_seq1, p_seq2):
    """
    params
    ---
    p_seq1 : str
    p_seq2 : str

    returns
    ---
    cer : float
    """
    p_vocab = set(p_seq1 + p_seq2)
    p2c = dict(zip(p_vocab, range(len(p_vocab))))
    c_seq1 = [chr(p2c[p]) for p in p_seq1]
    c_seq2 = [chr(p2c[p]) for p in p_seq2]
    return editdistance.eval(''.join(c_seq1),
                             ''.join(c_seq2)) / max(len(c_seq1), len(c_seq2))
