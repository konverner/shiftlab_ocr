# SHIFTLAB OCR

SHIFT OCR is a library for handwriting text segmentation and character recognition.
 
# Get Started

``` 
pip install shiftlab_ocr
```
## Doc2Text
`Reader` from `doc2text` performs text detection and the following recognition.

![](https://github.com/constantin50/shiftlab_ocr/blob/main/demo_image.png)

```
import urllib

from shiftlab_ocr.doc2text.reader import Reader


urllib.request.urlretrieve(
  'https://raw.githubusercontent.com/konverner/shiftlab_ocr/main/demo_image.png',
   'test.png')
   
reader = Reader()
result = reader.doc2text("test.png")

```

Display recognized text:

```
print(result[0])

Действительно ли добро сильнее зла?
Именно над этим вопросом аставля заставляет
читателей задуматься В. Тендряков.
Автор рассматривает данную пробле-
му на конкретном примере, рассказывая
историю 00 заблудившемся немце русских
солдатах, которые пожалели врала и
позволи ему остаться землянке. 

```

Display segmented crops:

```
import matplotlib.pyplot as plt

def show_img_grid(images, N):
    n = int(N**(0.5))
    k = 0
    f, axarr = plt.subplots(n,n,figsize=(10,10))
    for i in range(n):
        for j in range(n):
            axarr[i,j].imshow(images[k].img)
            k += 1
    f.show()

show_img_grid(result[1], 48)
```

![](https://github.com/konverner/shiftlab_ocr/blob/main/crops_image.png?raw=true)

## Generator of handwriting

It generates handwriting script with random backgrounds and handwriting fonts with a given string or a list of strings saved in `source.txt`.

Generating a random sample from a string:

```
from shiftlab_ocr.generator.generator import Generator

g = Generator(lang='ru')
s = g.generate_from_string('Москва',min_length=4,max_length=24) # get from a string
s
```

![](https://sun9-51.userapi.com/impg/CSeyZPb4rDmP4aCYIDoMDx5VQMXcWO6CwtpGUA/vH_cghX1JtA.jpg?size=344x88&quality=96&sign=c61344d4c7f5576ffe03e750ca31f94c&type=album)

Generating batch of random samples from `source.txt`:

```
import numpy as np

# upload source.txt with one word per line
g.upload_source('source.txt')
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
