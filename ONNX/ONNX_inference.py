################################################################################################################################# 
# ONNX runtime 
# Calling the Instance Segmentation model made with AutoML for Images
# 
# Date: 06-Jan-2021
# 
# Documentation:
# https://docs.microsoft.com/en-us/azure/machine-learning/how-to-inference-onnx-automl-image-models?tabs=instance-segmentation
#################################################################################################################################   
# 
# Syntax: 
# python ONNX_inference.py ONNXMODEL LABELS IMAGEFILE MINCONFIDENCE
# 
# ONNX_inference.py: this python file
# ONNXMODEL : the ONNX model made by AutoML for Images (you need to download it from Azure ML Studio)
# LABELS : labels file made by AutoML for Images (you need to download it from Azure ML Studio)
# IMAGEFILE : name of the image you want to analyze
# MINCONFIDENCE: minimum confidence
#
# Example:
# python ONNX_inference.py model.onnx labels.json image.png 0.7

import datetime
import json
import numpy as np
import onnxruntime
import os
import sys


# Arguments to retrieve from the command line
onnx_model_path = str(sys.argv[1]) # This is the ONNX model file (model.onnx for example)
labels_file = str(sys.argv[2]) # This is the labels files in a Json file format (labels.json)
test_image_path = str(sys.argv[3]) # This is the image you want to analyze
minconfidence = float(sys.argv[4]) # This is the minimum confidence of the prediction (0.7 for example)

onnxcsv = 'ONNXpredictions.csv' # Name of the csv results file where results will be stored


# Calling the ONNX model
t1 = datetime.datetime.now()
print()
print(t1, "Starting...")
print()
print("ONNX model:", onnx_model_path)
print("Labels file:", labels_file)
print("Image to test:", test_image_path)
print("Minimum Confidence value =", minconfidence)
print()
print("Python version:", sys.version)
print("ONNX runtime version:", onnxruntime.__version__)
print()


# Labels from the CV model
print("AutoML for Images Model labels to predict:")
with open(labels_file) as f:
    classes = json.load(f)
print(classes)
print()


# Opening the ONNX session
try:
    session = onnxruntime.InferenceSession(onnx_model_path)
    print("\nONNX model loaded...")
except Exception as e: 
    print("Error loading ONNX file: ",str(e))

sess_input = session.get_inputs()
sess_output = session.get_outputs()
print()
print(f"No. of inputs: {len(sess_input)}, No. of outputs: {len(sess_output)}")

for idx, input_ in enumerate(range(len(sess_input))):
    input_name = sess_input[input_].name
    input_shape = sess_input[input_].shape
    input_type = sess_input[input_].type
    print(f"{idx} Input name: { input_name }, Input shape: {input_shape}, \
    Input type: {input_type}")  


for idx, output in enumerate(range(len(sess_output))):
    output_name = sess_output[output].name
    output_shape = sess_output[output].shape
    output_type = sess_output[output].type
    print(f" {idx} Output name: {output_name}, Output shape: {output_shape}, \
    Output type: {output_type}") 


# Image preprocessing
def preprocess(image, resize_height, resize_width):
    """perform pre-processing on raw input image
        
    :param image: raw input image
    :type image: PIL image
    :param resize_height: resize height of an input image
    :type resize_height: Int
    :param resize_width: resize width of an input image
    :type resize_width: Int
    :return: pre-processed image in numpy format
    :rtype: ndarray of shape 1xCxHxW
    """

    image = image.convert('RGB')
    image = image.resize((resize_width,resize_height))
    np_image = np.array(image)

    # HWC -> CHW
    np_image = np_image.transpose(2, 0, 1)# CxHxW

    # normalize the image
    mean_vec = np.array([0.485, 0.456, 0.406])
    std_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(np_image.shape).astype('float32')

    for i in range(np_image.shape[0]):
        norm_img_data[i,:,:] = (np_image[i,:,:]/255 - mean_vec[i]) / std_vec[i]
    np_image = np.expand_dims(norm_img_data, axis=0)# 1xCxHxW
    
    return np_image


# use height and width based on the trained model
from PIL import Image
resize_height, resize_width = 600, 800 

img = Image.open(test_image_path)
print("\nInput image dimensions:", img.size)
#display(img)

img_data = preprocess(img, resize_height, resize_width)


# Get Predictions from ONNX session
def get_predictions_from_ONNX(onnx_session,img_data):
    """perform predictions with ONNX Runtime
    
    :param onnx_session: onnx model session
    :type onnx_session: class InferenceSession
    :param img_data: pre-processed numpy image
    :type img_data: ndarray with shape 1xCxHxW
    :return: boxes, labels , scores , masks with shapes
            (No. of instances, 4) (No. of instances,) (No. of instances,)
            (No. of instances, 1, HEIGHT, WIDTH))  
    :rtype: tuple
    """
    t2 = datetime.datetime.now()
    print()
    print(t2, "Get predictions from ONNX...")
    
    sess_input = onnx_session.get_inputs()
    sess_output = onnx_session.get_outputs()
    # predict with ONNX Runtime
    output_names = [ output.name for output in sess_output]
    boxes, labels, scores, masks = onnx_session.run(output_names=output_names,\
                                               input_feed={sess_input[0].name: img_data})
    print(datetime.datetime.now(), "Done! Time =", datetime.datetime.now() - t2)

    return boxes, labels, scores, masks

    
# Printing results
boxes, labels, scores, masks = get_predictions_from_ONNX(session, img_data)

print("\n=============== AutoML for Images Model Results using ONNX local inference ===============\n")
print("Analysing image:", test_image_path, "\n")

nbdefects = 0
totaldefects = 0

for nbdefects in range(0, len(scores)):
    if scores[nbdefects] > minconfidence:
        totaldefects +=1
        print(totaldefects, ": Defect has been detected with a confidence =", scores[nbdefects])
        print("Region of interest:", boxes[nbdefects], "\n")

print("=> Total of detected defects =", totaldefects, "on image:", test_image_path, "\n")
print("============================================ End ===========================================\n")


# Saving results into a csv file
print("Saving results to", onnxcsv)

if os.path.exists(onnxcsv):
    file_object = open(onnxcsv, 'a')
    file_object.write(str(datetime.datetime.now()) + ',' + str(test_image_path) + ',' + str(totaldefects) + '\n')
    file_object.close()

else:
    print("Creating the file headers...")
    file_object = open(onnxcsv, 'a')
    file_object.write("DateTime" + ',' + 'Image_File' + ',' + 'Number_of_Defects' + '\n')
    file_object.write(str(datetime.datetime.now()) + ',' + str(test_image_path) + ',' + str(totaldefects) + '\n')
    file_object.close()

print("Done!")


# End
print("\nPowered by Azure AutoML for Images and ONNX version", onnxruntime.__version__, '\n')

