import os
dirname = os.path.dirname(__file__)
DIR = os.path.join(dirname, 'weights')
PATH_TO_SEGM_MODEL = os.path.join(DIR,'unet.pth')
PATH_TO_MASK = os.path.join(DIR,'last_mask.png')