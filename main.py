from .crop import *
from .ocr_model import *
from .segmentation import *
from .paths import *
import urllib

class Scanner():
    def __init__(self, ocr_model=None):
        if ocr_model == 'hw-cyr':
            path_to_model = os.path.join(dirname, 'ocr_transformer_rn50_64x256_52wer_jit.pt')
            if not os.path.isfile(path_to_model):
                print('downloding OCR model...')
                urllib.request.urlretrieve('https://storage.googleapis.com/handwritten_rus/ocr_transformer_rn50_64x256_52str_jit.pt',\
                    path_to_model)
            self.ocr_model = get_model(path_to_model)
            self.segm_model = UNet(n_filters=hyperparametrs['n_filters'])
            self.segm_model.load_state_dict(torch.load(PATH_TO_SEGM_MODEL, map_location=torch.device('cpu')))
        else:
            raise Exception('choose a ocr model: ocr model="hw-cyr" for Cyrillic handwriting')


    def doc2text(self,IMAGE_PATH):
        """
        params
        ---
        IMAGE_PATH : str
          path to .png or .jpg file with image to read
        PATH_TO_WEIGHTS : str
          path to .pt file with weights of pytorch ocr model

        returns
        ---
        text : str
        crops : list of PIL.image objects
        crops are sorted
        """
        text = ''
        image = Image.open(IMAGE_PATH)
        image = image.resize((512, 512), Image.ANTIALIAS)
        boxes, _ = run_segmentation(self.segm_model, IMAGE_PATH)
        crops = []
        pad = 12 # padding
        for box in boxes:
            y1, y2, x1, x2 = box
            cropped = image.crop((x1 - pad, y1 - pad, x2 + pad, y2 + pad))

            crops.append(Crop([[x1,y1], [x2,y2]], img=cropped))
        crops = sorted(crops)
        for crop in crops:
            text += self.ocr_model.scan(crop.img) + ' '
        return text, crops