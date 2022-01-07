Local Inference with ONNX

1. Download the ONNX file from your Azure ML session
2. Downlaod the labels.json file from your Azure ML session
3. Run this command from your CLI :  python ONNX_inference.py model.onnx labels.json test1.png 0.7
4. This command will detect defects based on your AutoML for Images Instance segmentation computer vision you made using a local inference (ONNX).
