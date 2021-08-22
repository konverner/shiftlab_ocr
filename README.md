# shiftlab_ocr

SHIFT OCR is a library fo handwriting text segmentation and character recognition.
 
# Get Started

``` 
pip install shiftlab_ocr

```

```
import shiftlab_ocr

PATH_TO_IMAGE = 'test.jpg'
scanner = shift_ocr.Scanner(model_name='hw-cyr')
result = scanner.doc2text(PATH_TO_IMAGE)

('Директору Заявление 10 январе 2019г. Ирл Иванов А.П. ',
 [<shift_ocr.crop.Crop at 0x7f158cffd1d0>,
  <shift_ocr.crop.Crop at 0x7f158cffd610>,
  <shift_ocr.crop.Crop at 0x7f158cffd790>,
  <shift_ocr.crop.Crop at 0x7f158cffd8d0>,
  <shift_ocr.crop.Crop at 0x7f158cffdd50>,
  <shift_ocr.crop.Crop at 0x7f158cffd910>])

```

![](https://github.com/constantin50/shiftlab_ocr/blob/main/image.png)
  

Also, see [Google Colab Demo](https://colab.research.google.com/drive/1FPfQY9HvjEPEdzfFEZsgSCk5P1TBUAse?usp=sharing)
