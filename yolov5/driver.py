import torch
import pprint

def main():
  yolo_medium = torch.hub.load("ultralytics/yolov5", "yolov5m")

  # Images
  dir = 'https://github.com/ultralytics/yolov5/raw/master/data/images/'
  imgs = [dir + f for f in ('zidane.jpg', 'bus.jpg')]  # batch of images

  # Inference
  yolo_medium.eval()

if __name__ == "__main__":
  main()