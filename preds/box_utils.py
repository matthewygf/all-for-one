import numpy as np

def bb_iou(a, b):
  x1 = max(a[0], b[0])
  y1 = max(a[1], b[1])
  x2 = min(a[2], b[2])
  y2 = min(a[3], b[3])

  interArea = max(0, x2-x1) * max(0, y2-y1)

  if interArea == 0: return 0.0

  boxAArea = (a[2]-a[0]) * (a[3]-a[1])
  boxBArea = (b[2]-b[0]) * (b[3]-b[1])

  return interArea / float(boxAArea+boxBArea - interArea)


def find_matching_box(box_list, new_box, match_iou):
  best_iou = match_iou
  best_idx = -1
  for i in range(len(box_list)):
    box = box_list[i]
    iou = bb_iou(box[:4], new_box[:4])
    if iou > best_iou:
      best_idx = i
      best_iou = iou
  return best_idx, best_iou

def get_weighted_avg_box(clustered_boxes):
  confidence = 0
  box = np.zeros(5, dtype=np.float32)
  for b in clustered_boxes:
    # scale by the confidence
    box[:4] += np.array(b[:4]) * b[-1]
    confidence += b[-1]

  # avg confidence
  box[-1] = confidence / len(clustered_boxes)
  box[:4] /= confidence
  return box
