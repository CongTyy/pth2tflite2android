import tensorflow as tf

# Load the frozen model
with tf.io.gfile.GFile('yolov5s_saved_model/saved_model.pb', 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

# List node names
for node in graph_def.node:
    print(node.name)
