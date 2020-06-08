import numpy as np
import onnx
import onnxruntime as ort

# Load the ONNX model
sess = ort.InferenceSession("../output/models/003/t1t2.onnx")

input_name = sess.get_inputs()[0].name
print("Input name  :", input_name)
input_shape = sess.get_inputs()[0].shape
print("Input shape :", input_shape)
input_type = sess.get_inputs()[0].type
print("Input type  :", input_type)

output_name = sess.get_outputs()[0].name
print("Output name  :", output_name)
output_shape = sess.get_outputs()[0].shape
print("Output shape :", output_shape)
output_type = sess.get_outputs()[0].type
print("Output type  :", output_type)

x = np.random.random((1,2,256,256)).astype(np.float32)

result = sess.run([output_name], {input_name: x})