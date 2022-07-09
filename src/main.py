import torch
from shiftlab_ocr.src.crop import *
from shiftlab_ocr.src.recognition import Recognizer
from shiftlab_ocr.src.segmentation import *
from shiftlab_ocr.src.paths import *
import urllib


class Scanner:
    def __init__(self, detector_weights, recognizer_weights):
      self.recognizer = Recognizer()
      self.recognizer.load_model(recognizer_weights)
      self.detector = Detector()
      self.detector.load_model(detector_weights)

    def doc2text(self, IMAGE_PATH):
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
      image = Image.open(IMAGE_PATH).convert("RGB")
      original_image_width, original_image_height = image.size
      resized_image = image.resize((512, 512), Image.ANTIALIAS)
      boxes, _ = self.detector.run(resized_image, IMAGE_PATH)
      crops = []
      pad = 16  # padding
      for box in boxes:
          y1, y2, x1, x2 = box
          cropped = image.crop((x1*original_image_width/512 - pad, y1*original_image_height/512 - pad, \
                                x2*original_image_width/512 + pad, y2*original_image_height/512 + pad))

          crops.append(Crop([[x1, y1], [x2, y2]], img=cropped))
      crops = sorted(crops)
      for crop in crops:
          text += self.recognizer.run(crop.img) + ' '

      return text, crops
