import onnxruntime as ort

# Load the model
enc_session = ort.InferenceSession("Bert_VITS2/onnx/BertVits/BertVits_enc_p.onnx")

# Get the input names
input_names = [input.name for input in enc_session.get_inputs()]
print("Expected inputs:", input_names)