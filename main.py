import easyocr
import matplotlib.pyplot as plt
from PIL import Image
from crop import *
from ocr_model import *

def doc2text(IMAGE_PATH, PATH_TO_WEIGHTS):
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
    model = get_model(PATH_TO_WEIGHTS)
    text = ''
    reader = easyocr.Reader(['ru', 'ru'])
    bounds = reader.readtext(IMAGE_PATH)
    image = Image.open(IMAGE_PATH)
    crops = []
    pad = 5 # padding
    for bound in bounds:
        p0, p1, p2, p3 = bound[0]
        cropped = image.crop((p0[0] - pad, p0[1] - pad, p2[0] + pad, p2[1] + pad))
        crops.append(Crop([p0, p2], img=cropped))
    crops = sorted(crops)
    # show crops
    n = len(crops)
    fig = plt.figure(figsize=(8, 8))
    rows = int(n / 4) + 2
    columns = int(n / 8) + 2
    for j, crop in enumerate(crops):
        fig.add_subplot(rows, columns, j + 1)
        plt.imshow(np.asarray(crop.img))
        text += model.scan(crop.img) + ' '
    return text, crops