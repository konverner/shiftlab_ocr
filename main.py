import os
from .crop import *
from .ocr_model import *
from .paths import *
import easyocr
from PIL import Image
import urllib.request

class Scanner():
    def __init__(self, model_name, path_to_model=None):
        """
        params
        ---
        model_name : str
        path_to_model : str
            path to weights in .pt extension
        
        """
        if model_name == 'hw-cyr':
            path_to_model = os.path.join(dirname, 'ocr_transformer_rn50_64x256_52wer_jit.pt')
            if not os.path.isfile(path_to_model):
                print('downloding a model...')
                urllib.request.urlretrieve('https://storage.googleapis.com/handwritten_rus/ocr_transformer_rn50_64x256_52str_jit.pt',\
                    path_to_model)
            self.ocr_model = get_model(path_to_model)
            self.segm_model = easyocr.Reader(['ru', 'ru'])
        elif path_to_model != None:
            self.ocr_model = get_model(path_to_model)
            self.segm_model = easyocr.Reader(['ru', 'ru'])
        else:
            raise ValueError('Wrong model_name. The list of models: \n 1) "hw-cyr" - Handwriting Cyrillic recognition ')


    def doc2text(self,IMAGE_PATH):
        """
        params
        ---
        IMAGE_PATH : str
          path to .png or .jpg file with image to read

        returns
        ---
        text : str
        crops : list of PIL.image objects
        crops are sorted
        """
        text = ''
        bounds = self.segm_model.readtext(IMAGE_PATH)
        image = Image.open(IMAGE_PATH).convert('RGB')
        crops = []
        pad = 5 # padding
        for bound in bounds:
            p0, p1, p2, p3 = bound[0]
            cropped = image.crop((p0[0] - pad, p0[1] - pad, p2[0] + pad, p2[1] + pad))
            crops.append(Crop([p0, p2], img=cropped))
        crops = sorted(crops)
        for crop in crops:
            text += self.ocr_model.scan(crop.img)+' '
        return text, crops
