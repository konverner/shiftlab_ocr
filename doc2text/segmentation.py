import torch


class Detector:
  def __init__(self, yolo_path, model_path):
    self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model = torch.hub.load(yolo_path, 'custom', path=model_path, source='local')
    self.device = self._device

  def run(self, path_to_image):
    results = self.model([path_to_image])
    boxes = []
    data = results.pandas().xyxy[0]
    for i in range(data.shape[0]):
      boxes.append([data['xmin'][i], data['ymin'][i], data['xmax'][i], data['ymax'][i]])
    return boxes
