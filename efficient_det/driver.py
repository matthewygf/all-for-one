import tensorflow as tf

tf.compat.v1.disable_eager_execution()
from efficientdet import inference

driver = inference.ServingDriver('efficientdet-d1', 'efficientdet-d1')
driver.build()
driver.export('efficientdet-d1/saved_models')