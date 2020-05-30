import tflite_runtime.interpreter as tflite
import numpy as np
import time
import sys

from timeit import default_timer as timer

# Load TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path=sys.argv[1])
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

#Inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

#0=cls, 1=reg, 2=base_layers
tflite_results = interpreter.get_tensor(output_details[0]['index'])


# In[6]:

"""
from imageio import imread
from tensorflow.keras.preprocessing import image

orig_images = [] # Store the images here.
input_images = [] # Store resized versions of the images here.

img_height=300
img_width=300

# We'll only load one image in this example.
img_path = 'benchmarking_tests/python/silicon_valley.jpg'

orig_images.append(imread(img_path))
img = image.load_img(img_path, target_size=(img_height, img_width))
img = image.img_to_array(img) 
input_images.append(img)
input_images = np.array(input_images)


# In[8]:

#Inference
interpreter.set_tensor(input_details[0]['index'], input_images)
"""
runs = 25

start = timer()

for i in range(runs):
    interpreter.invoke()

end = timer() 

ave_latency = ((end - start)/runs)*1000
ave_FPS = 1000/ave_latency
print('Average detection latency is ', ave_latency, 'ms.')
print('Average FPS is ', ave_FPS)

#0=cls, 1=reg, 2=base_layers
y_pred = interpreter.get_tensor(output_details[0]['index'])
