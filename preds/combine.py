import json
from os import path
import pickle
import pandas as pd
import numpy as np
from box_utils import *
import os
from absl import app
from absl import flags

flags.DEFINE_string("prediction_jsons","","list of prediction coco compatible jsons separated by comma.")

FLAGS = flags.FLAGS

"""
making a few changes from
https://github.com/ZFTurbo/Weighted-Boxes-Fusion/blob/4efef777d5982d212138d3da1a6bcfdb99003476/examples/example_oid.py#L191
"""



def json_to_df(detections_json):
  df = pd.DataFrame(detections_json)
  return df

def compute_weighted_box_avg(bboxes):
  new_boxes = []
  weighted_boxes = []
  for i in range(0, len(bboxes)):
    best_idx, best_iou = find_matching_box(weighted_boxes, bboxes[i], 0.45)
    if best_idx == -1:
      # there might be many boxes that predict for the same category
      # but on different location
      # so -1 indicates that this is a "different" location
      new_boxes.append([bboxes[i].copy()])
      weighted_boxes.append(bboxes[i].copy())
    else:
      # this indicate that there are boxes that match the current box
      # meaning we are having a clustered of boxes
      new_boxes[best_idx].append(bboxes[i])
      weighted_boxes[best_idx] = get_weighted_avg_box(new_boxes[best_idx])
  
  for i in range(len(new_boxes)):
      clustered_boxes = np.array(new_boxes[i])
      weighted_boxes[i][-1] = weighted_boxes[i][-1] * len(clustered_boxes)
  

  return weighted_boxes

def combine_predictions(pred_dfs=None, path="image_predictions.pkl"):

  if os.path.exists(path):
    with open(path, "rb") as f:
      images = pickle.load(f)
  else:
    images = {}
    for df in pred_dfs:
      unique_ids = np.array(df.image_id.unique())
      for uid in unique_ids:
        subdf = df[df.image_id == uid]
        predictions_category = images.get(uid, {})
        for _, row in subdf.iterrows():
          bboxes = predictions_category.get(row.category_id, [])
          bboxes.append(row.bbox + [row.score])
          predictions_category[row.category_id] = bboxes
        images[uid] = predictions_category

  # weighted ?
  overall_results = []
  for imageID, categories in images.items():
    for categoryId, bboxes in categories.items():
      # bboxes should container models prediction for that category.
      weighted_boxes = compute_weighted_box_avg(bboxes)
      for b in weighted_boxes:
        # higher confidence ?
        if b[-1] > 0.005:
          overall_results.append({"image_id": int(imageID), "category_id": int(categoryId),
                                "bbox": [round(float(x),3) for x in b[:4]], "score": round(float(b[-1]), 5)})
  print(len(overall_results))
  with open("detections_test-dev2017_ensembled_results.json", "w+") as f:
    json.dump(overall_results, f)

def main(argv):
  del argv

  dfs = []
  for j in FLAGS.prediction_jsons.split(","):
    with open(j, "r") as f:
      df = json_to_df(json.load(f))
      dfs.append(df)

  combine_predictions(pred_dfs=dfs)

if __name__ == "__main__":
  app.run(main)