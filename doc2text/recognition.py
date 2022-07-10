import math
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms as t
from torch.nn import Conv2d, MaxPool2d, BatchNorm2d, LeakyReLU


ALPHABET = ['PAD', 'SOS', ' ', '!', '"', '%', '(', ')', ',', '-', '.', '/',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?',
            '[', ']', '«', '»', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И',
            'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х',
            'Ц', 'Ч', 'Ш', 'Щ', 'Э', 'Ю', 'Я', 'а', 'б', 'в', 'г', 'д', 'е',
            'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т',
            'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я',
            'ё', 'EOS']


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.scale = torch.nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x) 


class TransformerModel(nn.Module):
  def __init__(
    self,
    outtoken=len(ALPHABET),
    hidden=512,
    enc_layers=2,
    dec_layers=2,
    nhead=4,
    dropout=0.1,
    device='cpu'):
    super(TransformerModel, self).__init__()

    self.enc_layers = enc_layers
    self.dec_layers = dec_layers
    self.backbone_name = 'conv(64)->conv(64)->conv(128)->conv(256)->conv(256)->conv(512)->conv(512)'

    self.conv0 = Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.conv1 = Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.conv2 = Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))
    self.conv3 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.conv4 = Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))
    self.conv5 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.conv6 = Conv2d(512, 512, kernel_size=(2, 1), stride=(1, 1))
    
    self.pool1 = MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    self.pool3 = MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    self.pool5 = MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1), dilation=1, ceil_mode=False)

    self.bn0 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.bn1 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.bn2 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.bn3 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.bn4 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.bn5 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.bn6 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    self.activ = LeakyReLU()

    self.pos_encoder = PositionalEncoding(hidden, dropout)
    self.decoder = nn.Embedding(outtoken, hidden)
    self.pos_decoder = PositionalEncoding(hidden, dropout)
    self.transformer = nn.Transformer(d_model=hidden, nhead=nhead, num_encoder_layers=enc_layers,
                                      num_decoder_layers=dec_layers, dim_feedforward=hidden * 4, dropout=dropout)

    self.fc_out = nn.Linear(hidden, outtoken)
    self.src_mask = None
    self.trg_mask = None
    self.memory_mask = None
    self._device = device
    self.to(self._device)

  def generate_square_subsequent_mask(self, sz):
    mask = torch.triu(torch.ones(sz, sz, device=self._device), 1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask

  def make_len_mask(self, inp):
    return (inp == 0).transpose(0, 1)
  
  def _get_features(self, src):
    """
    params
    ---
    src : Tensor [64, 3, 64, 256] : [B,C,H,W]
        B - batch, C - channel, H - height, W - width
    returns
    ---
    x : Tensor : [W,B,CH]
    """
    x = self.activ(self.bn0(self.conv0(src)))
    x = self.pool1(self.activ(self.bn1(self.conv1(x))))
    x = self.activ(self.bn2(self.conv2(x)))
    x = self.pool3(self.activ(self.bn3(self.conv3(x))))
    x = self.activ(self.bn4(self.conv4(x)))
    x = self.pool5(self.activ(self.bn5(self.conv5(x))))
    x = self.activ(self.bn6(self.conv6(x)))
    x = x.permute(0, 3, 1, 2).flatten(2).permute(1, 0, 2)
    return x


  def predict(self, batch):
    """
    params
    ---
    batch : Tensor [64, 3, 64, 256] : [B,C,H,W]
        B - batch, C - channel, H - height, W - width
    
    returns
    ---
    tokens : List [64, -1] : [B, -1]
        preticted sequences of tokens
    """
    indexes = []
    for item in batch:
      x = self._get_features(item.unsqueeze(0))
      memory = self.transformer.encoder(self.pos_encoder(x))
      out_indexes = [ALPHABET.index('SOS'), ]
      for i in range(100):
          trg_tensor = torch.LongTensor(out_indexes).unsqueeze(1).to(self._device)
          output = self.fc_out(self.transformer.decoder(self.pos_decoder(self.decoder(trg_tensor)), memory))

          out_token = output.argmax(2)[-1].item()
          out_indexes.append(out_token)
          if out_token == ALPHABET.index('EOS'):
              break
      indexes.append(out_indexes)
    return indexes


  def forward(self, src, trg):
    """
    params
    ---
    src : Tensor [64, 3, 64, 256] : [B,C,H,W]
        B - batch, C - channel, H - height, W - width
    trg : Tensor [13, 64] : [L,B]
        L - max length of label
    """
    if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
        self.trg_mask = self.generate_square_subsequent_mask(len(trg)).to(trg.device) 

    x = self._get_features(src)
    src_pad_mask = self.make_len_mask(x[:, :, 0])
    src = self.pos_encoder(x)
    trg_pad_mask = self.make_len_mask(trg)
    trg = self.decoder(trg)
    trg = self.pos_decoder(trg)

    output = self.transformer(src, trg, src_mask=self.src_mask, tgt_mask=self.trg_mask,
                              memory_mask=self.memory_mask,
                              src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=trg_pad_mask,
                              memory_key_padding_mask=src_pad_mask)
    output = self.fc_out(output)

    return output


class Recognizer:
  def __init__(
    self,
    model=TransformerModel,
    alphabet=ALPHABET,
    image_config={'width': 256, 'height': 64, 'channels': 1},
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    ):
    self._device = device
    self.model = model(device=self._device)
    self.char2idx = {char: idx for idx, char in enumerate(alphabet)}
    self.idx2char = {idx: char for idx, char in enumerate(alphabet)}
    self.image_config = image_config

  def load_model(self, weights):
    """
    params
    ---
    weights : str
      path to weigths
    """
    self.model.load_state_dict(torch.load(weights, map_location=self._device))
    self.model.eval()
    print(f'recognizer weights has loaded from {weights}')

  def _indexes_to_text(self, indexes):
    text = "".join([self.idx2char[i] for i in indexes])
    text = text.replace('EOS', '').replace('PAD', '').replace('SOS', '')
    return text
  
  # RESIZE AND NORMALIZE IMAGE
  def _process_image(self, img):
    """
    params:
    ---
    img : np.array
    returns
    ---
    img : np.array
    """
    w, h, _ = img.shape
    new_w = self.image_config['height']
    new_h = int(h * (new_w / w))
    img = cv2.resize(img, (new_h, new_w))
    w, h, _ = img.shape

    img = img.astype('float32')

    new_h = self.image_config['width']
    if h < new_h:
        add_zeros = np.full((w, new_h - h, 3), 255)
        img = np.concatenate((img, add_zeros), axis=1)

    if h > new_h:
        img = cv2.resize(img, (new_h, new_w))
    return img

  def run(self, img):
    """
    params
    ---
    img : PIL.Image

    returns
    ---
    chars : List[str]
    """
    img = np.asarray(img)
    img = self._process_image(img).astype('uint8')
    img = img / img.max()
    img = np.transpose(img, (2, 0, 1))
    src = t.Compose([t.ToPILImage(), t.Grayscale(1), t.ToTensor()])(torch.FloatTensor(img)).unsqueeze(0).to(self._device)
    indexes = self.model.predict(src)[0]
    chars = self._indexes_to_text(indexes)
    return chars
