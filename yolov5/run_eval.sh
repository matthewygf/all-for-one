python /root/codes/yolov5/test.py --data /root/codes/yolov5/data/coco.yaml \
  --img 640 --conf-thres 0.001 --iou-thres 0.65 --weights /root/codes/yolov5/yolov5m.pt \
  --batch-size 8 --project /root/codes/yolov5/runs/coco/test --name v5m/infer --save-txt --save-hybrid 