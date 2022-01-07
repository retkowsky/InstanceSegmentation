# Local Inference with ONNX

1. Download the ONNX file from your Azure ML session
2. Download the labels.json file from your Azure ML session
3. Run this command from your CLI :  python ONNX_inference.py model.onnx labels.json test1.png 0.7
4. This command will detect defects based on your AutoML for Images Instance segmentation computer vision you made using a local inference (ONNX).

<img src = "https://github.com/retkowsky/InstanceSegmentation/blob/main/captures/image.png?raw=true">

ONNX: https://docs.microsoft.com/en-us/azure/machine-learning/how-to-inference-onnx-automl-image-models?tabs=multi-class

07-Jan-2022<br>
Serge Retkowsky | serge.retkowsky@microsoft.com | https://www.linkedin.com/in/serger/
