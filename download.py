import tensorflow as tf
import tensorflow_hub as hub


model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
model_dir = "ssd_mobilenet_v2"


model = hub.load(model_url)


tf.saved_model.save(model, model_dir)
