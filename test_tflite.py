import numpy as np
import tensorflow as tf

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="yolov5s-fp16.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()



# Test model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
# interpreter.set_tensor(input_details[0]['index'], input_data)

# interpreter.invoke()

my_signature = interpreter.get_signature_runner("my_method")
results = my_signature(input_1=input_data)
print(results["my_output"])

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
for i in range(4):
    print(interpreter.get_tensor(output_details[i]['index']).shape)
'''
1-25    1-25-4
1-25    1-25
1-25-4  1-25
1       1,
'''