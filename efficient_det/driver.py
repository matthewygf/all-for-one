import tensorflow as tf
import requests
from PIL import Image
import numpy as np
from pathlib import Path

tf.compat.v1.disable_eager_execution()

saved_models_dir = 'efficient_det/ckpts/efficientdet-d1/saved_models'

from efficientdet import inference

def export_saved_model():
  #step 1. savedmodels
  # batchsize none for dynamic inference :/
  driver = inference.ServingDriver('efficientdet-d1', 
            'efficient_det/ckpts/efficientdet-d1', batch_size=None)
  driver.build()
  driver.export(saved_models_dir)

def main():
  # if not Path(saved_models_dir).exists():
  #   export_saved_model()

  dir = 'https://github.com/ultralytics/yolov5/raw/master/data/images/'
  imgs = [dir + f for f in ('zidane.jpg', 'bus.jpg')]  # batch of images
  with tf.compat.v1.Session() as sess:
    tf.compat.v1.saved_model.load(sess,['serve'],  saved_models_dir)
    raw_images = []

    for im in imgs:
      img = Image.open(requests.get(im, stream=True).raw)
      img = np.array(img.resize((640,640)))
      raw_images.append(img)

    detections = sess.run('detections:0', {'image_arrays:0': raw_images})
    driver = inference.ServingDriver('efficientdet-d1', 'efficient_det/ckpts/efficientdet-d1')
    for i, d in enumerate(detections):
      img = driver.visualize(raw_images[i], detections[i], min_score_thresh=0.45)
      Image.fromarray(img).save(f"x{str(i)}.jpg")

if __name__ == "__main__":
  main()