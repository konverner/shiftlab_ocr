import os
from PIL import Image

from shiftlab_ocr.doc2text.crop import Crop
from shiftlab_ocr.doc2text.recognition import Recognizer
from shiftlab_ocr.doc2text.segmentation import Detector


class Reader:
    def __init__(
      self,
      detector_weights = os.path.join(os.path.dirname(__file__), "unet.pth"),
      recognizer_weights = os.path.join(os.path.dirname(__file__), "ocr_transformer_4h2l_simple_conv_64x256.pth"),
      ):
      self.recognizer = Recognizer()
      self.recognizer.load_model(recognizer_weights)
      self.detector = Detector()
      self.detector.load_model(detector_weights)

    def doc2text(self, image_path):
      """
      params
      ---
      image_path : str
      path to .png or .jpg file with image to read

      returns
      ---
      text : str
      crops : list of PIL.image objects
      crops are sorted
      """
      text = ''
      image = Image.open(image_path).convert("RGB")
      original_image_width, original_image_height = image.size
      resized_image = image.resize((512, 512), Image.ANTIALIAS)
      boxes, _ = self.detector.run(resized_image, image_path)
      crops = []
      pad = 16  # padding
      for box in boxes:
          y1, y2, x1, x2 = box
          cropped = image.crop((x1 * original_image_width / 512 - pad, y1 * original_image_height / 512 - pad,
                                x2 * original_image_width / 512 + pad, y2 * original_image_height / 512 + pad))

          crops.append(Crop([[x1, y1], [x2, y2]], img=cropped))
      crops = sorted(crops)
      for crop in crops:
          text += self.recognizer.run(crop.img) + ' '

      return text, crops
