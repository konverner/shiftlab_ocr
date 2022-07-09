# shiftlab_ocr

SHIFT OCR is a library fo handwriting text segmentation and character recognition.
 
# Get Started

## Doc2Text

it transcribes an image into text 

``` 
pip install shiftlab_ocr

```

```

from shiftlab_ocr.doc2text import Reader

reader = Reader()
result = reader.doc2text("/content/test.png")

('Директору Заявление 10 январе 2019г. Ирл Иванов А.П. ',
 [<shift_ocr.crop.Crop at 0x7f158cffd1d0>,
  <shift_ocr.crop.Crop at 0x7f158cffd610>,
  <shift_ocr.crop.Crop at 0x7f158cffd790>,
  <shift_ocr.crop.Crop at 0x7f158cffd8d0>,
  <shift_ocr.crop.Crop at 0x7f158cffdd50>,
  <shift_ocr.crop.Crop at 0x7f158cffd910>])

```

![](https://github.com/constantin50/shiftlab_ocr/blob/main/image.png)

## Generator of handwriting

It generates handwriting script with random backgrounds and handwriting fonts with given list of strings

```

from shiftlab_ocr import Generator

g = Generator(lang='ru')
g.upload_source('/content/source.txt')

s = g.generate_from_string('Москва',min_length=4,max_length=24) # get from a string
s

```

![](https://sun9-51.userapi.com/impg/CSeyZPb4rDmP4aCYIDoMDx5VQMXcWO6CwtpGUA/vH_cghX1JtA.jpg?size=344x88&quality=96&sign=c61344d4c7f5576ffe03e750ca31f94c&type=album)

```

b = g.generate_batch(12,4,13) # get batch of random samples from source.txt
fig=plt.figure(figsize=(10, 10))
rows = int(len(b)/4) + 2
columns = int(len(b)/8) + 2
for i in range(len(b)):
  fig.add_subplot(rows, columns, i+1)
  plt.imshow(np.asarray(b[i][0])) 

```

![](https://sun9-80.userapi.com/impg/ay9o11D8ItN65kDqYnZBahiZFk1zZ2wo5BYoMA/I_nNhdMQeLs.jpg?size=600x409&quality=96&sign=9d6a3ee935fcdc7112aec557eeed74f1&type=album)

Also, see [Google Colab Demo](https://colab.research.google.com/drive/1FPfQY9HvjEPEdzfFEZsgSCk5P1TBUAse?usp=sharing)
