from style_bert_vits2.tts_model import TTSModel
import torch
from typing import Union, Optional, Any, List, Dict, Tuple
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
import time
import os
import onnxruntime as ort
from style_bert_vits2.constants import (
    DEFAULT_ASSIST_TEXT_WEIGHT,
    DEFAULT_LENGTH,
    DEFAULT_LINE_SPLIT,
    DEFAULT_NOISE,
    DEFAULT_NOISEW,
    DEFAULT_SDP_RATIO,
    DEFAULT_SPLIT_INTERVAL,
    DEFAULT_STYLE,
    DEFAULT_STYLE_WEIGHT,
    Languages,
)
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.models.models import SynthesizerTrn
from style_bert_vits2.models.models_jp_extra import (
    SynthesizerTrn as SynthesizerTrnJPExtra,
)
from style_bert_vits2.models.infer import get_text, cast, infer
from style_bert_vits2.nlp import bert_models
from huggingface_hub import hf_hub_download
from convert_onnx import export_onnx

# Initialize the model and required components
bert_models.load_model(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
bert_models.load_tokenizer(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
model_file = "jvnv-F1-jp/jvnv-F1-jp_e160_s14000.safetensors"
config_file = "jvnv-F1-jp/config.json"
style_file = "jvnv-F1-jp/style_vectors.npy"

for file in [model_file, config_file, style_file]:
    print(f"Downloading {file}...")
    hf_hub_download("litagin/style_bert_vits2_jvnv", file, local_dir="model_assets")

assets_root = Path("model_assets")
print("Model files downloaded. Initializing model...")

class InnerInferModel(torch.nn.Module):
    """Module for torch.compile and ONNX export"""
    def __init__(self, net_g, device, hps):
        super().__init__()
        self.net_g = net_g
        self.device = device
        self.hps = hps
        
    def preprocess_text(
        self,
        text: str,
        language: Languages,
        assist_text: Optional[str] = None,
        assist_text_weight: float = 0.7,
        given_phone: Optional[list[str]] = None,
        given_tone: Optional[list[int]] = None,
        skip_start: bool = False,
        skip_end: bool = False,
    ):
        bert, ja_bert, en_bert, phones, tones, lang_ids = get_text(
            text,
            language,
            self.hps,
            self.device,
            assist_text=assist_text,
            assist_text_weight=assist_text_weight,
            given_phone=given_phone,
            given_tone=given_tone,
        )
        
        if skip_start:
            phones = phones[3:]
            tones = tones[3:]
            lang_ids = lang_ids[3:]
            bert = bert[:, 3:]
            ja_bert = ja_bert[:, 3:]
            en_bert = en_bert[:, 3:]
            
        if skip_end:
            phones = phones[:-2]
            tones = tones[:-2]
            lang_ids = lang_ids[:-2]
            bert = bert[:, :-2]
            ja_bert = ja_bert[:, :-2]
            en_bert = en_bert[:, :-2]
            
        return bert, ja_bert, en_bert, phones, tones, lang_ids
    
    def forward(
        self,
        bert, ja_bert, en_bert, phones, tones, lang_ids,
        style_vec: torch.Tensor,
        sdp_ratio: float,
        noise_scale: float,
        noise_scale_w: float,
        length_scale: float,
        sid: torch.Tensor,
    ):        
        with torch.no_grad():
            x_tst = phones.unsqueeze(0)
            tones = tones.unsqueeze(0)
            lang_ids = lang_ids.unsqueeze(0)
            bert = bert.unsqueeze(0)
            ja_bert = ja_bert.unsqueeze(0)
            en_bert = en_bert.unsqueeze(0)
            x_tst_lengths = torch.LongTensor([phones.size(0)]).to(self.device)
            
            net_g = self.net_g
            output = cast(SynthesizerTrnJPExtra, net_g).infer(
                x_tst,
                x_tst_lengths,
                sid,
                tones,
                lang_ids,
                ja_bert,
                style_vec=style_vec,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )
            audio = output[0][0, 0]
            return audio

class CustomTTSModel(TTSModel):
    def __init__(self, model_path: Path, config_path: Union[Path, HyperParameters], style_vec_path: Union[Path, NDArray[Any]], device: str) -> None:
        super().__init__(model_path, config_path, style_vec_path, device)
        self.load()
        assert self._TTSModel__net_g is not None, "Model not loaded correctly, net_g is None"
        
        # Setup torch.compile inference
        self.inner_infer = torch.compile(self._TTSModel__net_g, fullgraph=True, backend="onnxrt")
        self.compiled_inner_infer = InnerInferModel(self.inner_infer, self.device, self.hyper_parameters)
        self.use_compile = True
        self.fixed_seq_len = 128  # Fixed sequence length for ONNX export and inference
        
    def load_onnx_model(self, path: Path):
        """Load an ONNX model for inference"""
        if os.path.exists(path):
            # Create ONNX Runtime session
            self.onnx_path = str(path)
            # Configure session options for better performance
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Create providers with appropriate execution configuration
            providers = ['CPUExecutionProvider']
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers = ['CUDAExecutionProvider'] + providers
                
            self.ort_session = ort.InferenceSession(self.onnx_path, sess_options=sess_options, providers=providers)
            
            # Print model input details
            print(f"Loaded ONNX model from: {path}")
            print("ONNX Model Inputs:")
            for i, input_info in enumerate(self.ort_session.get_inputs()):
                print(f"  {i}: {input_info.name} - {input_info.shape} ({input_info.type})")
            return True
        else:
            print(f"ONNX model not found at: {path}")
            return False
            
    def pad_to_fixed_length(self, tensor, fixed_len=None, pad_dim=0, pad_value=0):
        """Pad or truncate tensor to fixed length for ONNX inference"""
        if fixed_len is None:
            fixed_len = self.fixed_seq_len
            
        if isinstance(tensor, np.ndarray):
            curr_len = tensor.shape[pad_dim]
            if curr_len < fixed_len:
                # Need padding
                pad_shape = [(0, 0)] * tensor.ndim
                pad_shape[pad_dim] = (0, fixed_len - curr_len)
                return np.pad(tensor, pad_shape, mode='constant', constant_values=pad_value)
            elif curr_len > fixed_len:
                # Need truncation
                slices = [slice(None)] * tensor.ndim
                slices[pad_dim] = slice(0, fixed_len)
                return tensor[tuple(slices)]
            return tensor
        else:  # PyTorch tensor
            curr_len = tensor.size(pad_dim)
            if curr_len < fixed_len:
                # Need padding
                padding_size = [0] * (2 * tensor.dim())
                padding_size[2 * pad_dim + 1] = fixed_len - curr_len
                return torch.nn.functional.pad(tensor, padding_size, 'constant', pad_value)
            elif curr_len > fixed_len:
                # Need truncation
                return torch.narrow(tensor, pad_dim, 0, fixed_len)
            return tensor
    
    def _compiled_infer_implementation(
        self,
        text: str,
        style_vector: NDArray[Any],
        sdp_ratio: float,
        noise: float,
        noisew: float,
        length: float,
        speaker_id: int,
        language: Languages,
        assist_text: Optional[str] = None,
        assist_text_weight: float = DEFAULT_ASSIST_TEXT_WEIGHT,
        given_phone: Optional[list[str]] = None,
        given_tone: Optional[list[int]] = None,
    ) -> NDArray[Any]:
        """The compiled infer implementation using torch.compile"""
        bert, ja_bert, en_bert, phones, tones, lang_ids = self.compiled_inner_infer.preprocess_text(
            text, language, assist_text, assist_text_weight, given_phone, given_tone
        )
        
        style_vec_tensor = torch.from_numpy(style_vector).to(self.device).unsqueeze(0)
        sid_tensor = torch.LongTensor([speaker_id]).to(self.device)
        
        audio = self.compiled_inner_infer(
            bert, ja_bert, en_bert, phones, tones, lang_ids,
            style_vec_tensor, sdp_ratio, noise, noisew, 
            length, sid_tensor
        )
        return audio.cpu().numpy()

    def infer(
        self,
        text: str,
        language: Languages = Languages.JP,
        speaker_id: int = 0,
        reference_audio_path: Optional[str] = None,
        sdp_ratio: float = DEFAULT_SDP_RATIO,
        noise: float = DEFAULT_NOISE,
        noise_w: float = DEFAULT_NOISEW,
        length: float = DEFAULT_LENGTH,
        line_split: bool = DEFAULT_LINE_SPLIT,
        split_interval: float = DEFAULT_SPLIT_INTERVAL,
        assist_text: Optional[str] = None,
        assist_text_weight: float = DEFAULT_ASSIST_TEXT_WEIGHT,
        use_assist_text: bool = False,
        style: str = DEFAULT_STYLE,
        style_weight: float = DEFAULT_STYLE_WEIGHT,
        given_phone: Optional[list[str]] = None,
        given_tone: Optional[list[int]] = None,
        pitch_scale: float = 1.0,
        intonation_scale: float = 1.0,
        compare_methods: bool = True,
        num_iterations: int = 3,
    ) -> tuple[int, NDArray[Any]]:
        use_compiled: bool = self.use_compile
        
        if language != "JP" and self.hyper_parameters.version.endswith("JP-Extra"):
            raise ValueError(
                "The model is trained with JP-Extra, but the language is not JP"
            )
        if reference_audio_path == "":
            reference_audio_path = None
        if assist_text == "" or not use_assist_text:
            assist_text = None

        if self._TTSModel__net_g is None:
            self.load()
        assert self._TTSModel__net_g is not None
        
        if reference_audio_path is None:
            style_id = self.style2id[style]
            style_vector = self._TTSModel__get_style_vector(style_id, style_weight)
        else:
            style_vector = self._TTSModel__get_style_vector_from_audio(
                reference_audio_path, style_weight
            )
            
        if compare_methods:
            compiled_times = []
            onnx_times = []
            print("\n--- Starting performance comparison ---")
            print(f"Running {num_iterations} iterations for each method...")
            test_texts = text if isinstance(text, list) else [text]
            
            for text_item in test_texts:
                print(f"  Text: '{text_item[:30]}{'...' if len(text_item) > 30 else ''}'")
                
                # First run torch.compile method
                if use_compiled:
                    compiled_results = []
                    for i in range(num_iterations):
                        start_time = time.time()
                        try:
                            _ = self._compiled_infer_implementation(
                                text=text_item, 
                                style_vector=style_vector,
                                sdp_ratio=sdp_ratio,
                                noise=noise,
                                noisew=noise_w,
                                length=length,
                                speaker_id=speaker_id,
                                language=language,
                                assist_text=assist_text,
                                assist_text_weight=assist_text_weight,
                                given_phone=given_phone,
                                given_tone=given_tone
                            )
                            compiled_time = time.time() - start_time
                            compiled_results.append(compiled_time)
                            if i == 0:
                                print(f"  Torch Compile method: {compiled_time:.4f} seconds")
                        except Exception as e:
                            print(f"  Torch Compile method failed: {e}")
                            break
                    
                    if compiled_results:
                        avg_time = sum(compiled_results) / len(compiled_results)
                        compiled_times.append(avg_time)
                        if len(compiled_results) > 1:
                            print(f"  Torch Compile average ({len(compiled_results)} runs): {avg_time:.4f} seconds")
                
                print("  " + "-" * 40)
                
            print("\n--- Performance Summary ---")
            if compiled_times:
                avg_compiled = sum(compiled_times) / len(compiled_times)                    
                print(f"Torch Compile method average time: {avg_compiled:.4f} seconds")
            print("--------------------------------------")
            
            return True, np.array([])

def export_to_onnx_with_padding(model, output_path, device="cpu", fixed_seq_len=128):
    """Export model to ONNX with fixed sequence length padding"""
    # Get the net_g component
    net_g = model._TTSModel__net_g
    net_g.eval()
    
    # Create wrapper for the net_g component
    class PaddedNetGWrapper(torch.nn.Module):
        def __init__(self, net_g, device, fixed_seq_len):
            super().__init__()
            self.net_g = net_g
            self.device = device
            self.fixed_seq_len = fixed_seq_len
            self.is_jp_extra = True  # For the specific model you're using
        
        def forward(self, bert, ja_bert, phones, tones, lang_ids,
                   style_vec, sdp_ratio, noise_scale, noise_scale_w, length_scale, sid):
            # Process inputs with fixed length
            seq_len = phones.size(1)  # Get the actual sequence length
            x_tst_lengths = torch.LongTensor([seq_len]).to(self.device)
            
            # Call the model's inference function
            if self.is_jp_extra:
                output = self.net_g.infer(
                    phones,
                    x_tst_lengths,
                    sid,
                    tones,
                    lang_ids,
                    ja_bert,
                    style_vec=style_vec,
                    sdp_ratio=sdp_ratio,
                    noise_scale=noise_scale,
                    noise_scale_w=noise_scale_w,
                    length_scale=length_scale,
                )
            else:
                output = self.net_g.infer(
                    phones,
                    x_tst_lengths,
                    sid,
                    tones,
                    lang_ids,
                    bert,
                    ja_bert,
                    None,  # en_bert
                    style_vec=style_vec,
                    sdp_ratio=sdp_ratio,
                    noise_scale=noise_scale,
                    noise_scale_w=noise_scale_w,
                    length_scale=length_scale,
                )
            
            # Return audio with batch dimension
            return output[0]  # shape [b, 1, audio_len]
    
    # Create sample input data
    # Create sample input with the right dimensions
    inner_model = InnerInferModel(net_g, device, model.hyper_parameters)
    sample_text = "こんにちは、これはテストです。"
    bert, ja_bert, en_bert, phones, tones, lang_ids = inner_model.preprocess_text(
        sample_text, Languages.JP
    )
    
    # Print original dimensions for debugging
    print(f"Raw dimensions - bert: {bert.shape}, ja_bert: {ja_bert.shape}, phones: {phones.shape}")
    
    # Pad tensors to fixed sequence length
    print(f"Original sequence lengths - phones: {phones.size(0)}, bert: {bert.size(1)}")
    
    # For BERT embeddings, we need to make sure the correct dimension is 1024
    # In the error, we need the second dimension (index 1) to be 1024
    # Let's examine the tensor shapes
    print(f"BERT dimensions: {bert.shape}, expected middle dim to be 1024")
    
    # Create fresh tensors with proper dimensions
    bert_dim = 1024
    bert_features = bert.size(-1)  # Last dimension (features)
    
    # Create properly sized tensors
    new_bert = torch.zeros((bert_dim, bert_features), device=bert.device, dtype=bert.dtype)
    new_ja_bert = torch.zeros((bert_dim, bert_features), device=ja_bert.device, dtype=ja_bert.dtype)
    
    # Copy the data we have up to the minimum size
    copy_size = min(bert.size(1), bert_dim)
    new_bert[:copy_size, :] = bert[:, :copy_size, :].squeeze(0)
    new_ja_bert[:copy_size, :] = ja_bert[:, :copy_size, :].squeeze(0)
    
    # Replace original tensors
    bert = new_bert
    ja_bert = new_ja_bert
    
    print(f"Resized BERT tensors to shape: {bert.shape}")
    
    # For sequence dimensions, pad to fixed length
    phones = model.pad_to_fixed_length(phones, fixed_seq_len)
    tones = model.pad_to_fixed_length(tones, fixed_seq_len)
    lang_ids = model.pad_to_fixed_length(lang_ids, fixed_seq_len)
    
    # Add batch dimension
    bert = bert.unsqueeze(0)
    ja_bert = ja_bert.unsqueeze(0)
    phones = phones.unsqueeze(0)
    tones = tones.unsqueeze(0)
    lang_ids = lang_ids.unsqueeze(0)
    
    print(f"Padded shapes - bert: {bert.shape}, ja_bert: {ja_bert.shape}, phones: {phones.shape}")
    
    # Create other required inputs
    style_vector = np.zeros((1, 256), dtype=np.float32)  # Add batch dimension
    style_vec_tensor = torch.from_numpy(style_vector).float().to(device)
    sdp_ratio_tensor = torch.tensor(DEFAULT_SDP_RATIO, dtype=torch.float32).to(device)
    noise_scale_tensor = torch.tensor(DEFAULT_NOISE, dtype=torch.float32).to(device)
    noise_scale_w_tensor = torch.tensor(DEFAULT_NOISEW, dtype=torch.float32).to(device)
    length_scale_tensor = torch.tensor(DEFAULT_LENGTH, dtype=torch.float32).to(device)
    sid = torch.LongTensor([0]).to(device)
    
    # Create the wrapper
    wrapper = PaddedNetGWrapper(net_g, device, fixed_seq_len)
    
    # Define input names
    input_names = [
        "bert", "ja_bert", "phones", "tones", "lang_ids",
        "style_vec", "sdp_ratio", "noise_scale", "noise_scale_w", "length_scale", "sid"
    ]
    
    # Define output name
    output_names = ["audio"]
    
    # Export to ONNX with fixed shapes
    torch.onnx.export(
        wrapper,
        (bert, ja_bert, phones, tones, lang_ids,
        style_vec_tensor, sdp_ratio_tensor, noise_scale_tensor, noise_scale_w_tensor, 
        length_scale_tensor, sid),
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=False,  # Disable constant folding which is causing warnings
        input_names=input_names,
        output_names=output_names,
        # No dynamic axes - using fixed sequence length
    )
    
    # Verify the ONNX model
    try:
        import onnx
        model = onnx.load(output_path)
        onnx.checker.check_model(model)
        print(f"ONNX model verification passed")
    except Exception as e:
        print(f"ONNX model verification failed: {e}")
    
    print(f"Model successfully exported to {output_path}")
    return output_path

def infer_with_padded_onnx(model, onnx_path, text, language=Languages.JP, speaker_id=0, 
                          style_vector=None, sdp_ratio=DEFAULT_SDP_RATIO, noise=DEFAULT_NOISE,
                          noise_w=DEFAULT_NOISEW, length=DEFAULT_LENGTH):
    """Run inference using ONNX model with padded fixed sequence inputs"""
    # Load the ONNX model if not already loaded
    if not hasattr(model, "ort_session") or model.ort_session is None:
        if not model.load_onnx_model(onnx_path):
            return None
    
    # Process text input
    inner_model = InnerInferModel(model._TTSModel__net_g, model.device, model.hyper_parameters)
    bert, ja_bert, en_bert, phones, tones, lang_ids = inner_model.preprocess_text(text, language)
    
    # Original dimensions for debugging
    orig_seq_len = phones.size(0)
    orig_bert_dim = bert.size(1)
    print(f"Original dimensions - sequence: {orig_seq_len}, bert channels: {orig_bert_dim}")
    
    # First handle BERT dimensions - must be exactly 1024 in the middle dimension
    print(f"BERT dimensions before resize: {bert.shape}")
    
    # Create fresh tensors with proper dimensions
    bert_dim = 1024
    bert_features = bert.size(-1)  # Last dimension (features)
    
    # Create properly sized tensors
    new_bert = torch.zeros((bert_dim, bert_features), device=bert.device, dtype=bert.dtype)
    new_ja_bert = torch.zeros((bert_dim, bert_features), device=ja_bert.device, dtype=ja_bert.dtype)
    
    # Copy the data we have up to the minimum size
    copy_size = min(bert.size(1), bert_dim)
    new_bert[:copy_size, :] = bert[:, :copy_size, :].squeeze(0)
    new_ja_bert[:copy_size, :] = ja_bert[:, :copy_size, :].squeeze(0)
    
    # Replace original tensors
    bert = new_bert
    ja_bert = new_ja_bert
    
    print(f"Resized BERT tensors to shape: {bert.shape}")
    
    # Then handle sequence dimensions
    phones = model.pad_to_fixed_length(phones)
    tones = model.pad_to_fixed_length(tones)
    lang_ids = model.pad_to_fixed_length(lang_ids)
    
    # Convert to numpy and handle dimensions correctly
    # For BERT embeddings, they need to be in format [batch, 1024, features]
    bert_np = bert.cpu().numpy().astype(np.float32)
    ja_bert_np = ja_bert.cpu().numpy().astype(np.float32)
    phones_np = phones.cpu().numpy().astype(np.int64)
    tones_np = tones.cpu().numpy().astype(np.int64)
    lang_ids_np = lang_ids.cpu().numpy().astype(np.int64)
    
    # Add batch dimension if not present
    if bert_np.ndim == 2:
        bert_np = np.expand_dims(bert_np, 0)  # Shape should become [1, 1024, features]
    if ja_bert_np.ndim == 2:
        ja_bert_np = np.expand_dims(ja_bert_np, 0)
    if phones_np.ndim == 1:
        phones_np = np.expand_dims(phones_np, 0)
    if tones_np.ndim == 1:
        tones_np = np.expand_dims(tones_np, 0)
    if lang_ids_np.ndim == 1:
        lang_ids_np = np.expand_dims(lang_ids_np, 0)
    
    # Double check BERT dimensions
    if bert_np.shape[1] != 1024:
        print(f"Warning: BERT still has wrong dimension! {bert_np.shape}")
        
    # Print shapes after padding for debugging
    print(f"Padded shapes - bert: {bert_np.shape}, ja_bert: {ja_bert_np.shape}, phones: {phones_np.shape}")
    
    # Other inputs
    if style_vector is None:
        style_vector = np.zeros((256), dtype=np.float32)
    style_vec_np = style_vector.astype(np.float32)
    if style_vec_np.ndim == 1:
        style_vec_np = np.expand_dims(style_vec_np, 0)  # Add batch dimension
    
    # Scalar inputs
    sdp_ratio_np = np.array(sdp_ratio, dtype=np.float32)
    noise_scale_np = np.array(noise, dtype=np.float32)
    noise_scale_w_np = np.array(noise_w, dtype=np.float32)
    length_scale_np = np.array(length, dtype=np.float32)
    sid_np = np.array([speaker_id], dtype=np.int64)
    
    # Create input dictionary for ONNX Runtime
    ort_inputs = {
        "bert": bert_np,
        "ja_bert": ja_bert_np,
        "phones": phones_np,
        "tones": tones_np,
        "lang_ids": lang_ids_np,
        "style_vec": style_vec_np,
        "sdp_ratio": sdp_ratio_np,
        "noise_scale": noise_scale_np,
        "noise_scale_w": noise_scale_w_np,
        "length_scale": length_scale_np,
        "sid": sid_np,
    }
    
    # Run inference
    try:
        start_time = time.time()
        outputs = model.ort_session.run(None, ort_inputs)
        inference_time = time.time() - start_time
        print(f"ONNX inference time: {inference_time:.4f} seconds")
        audio = outputs[0]  # First output is the audio
        return audio
    except Exception as e:
        print(f"ONNX inference error: {e}")
        # Print input shapes for debugging
        for name, value in ort_inputs.items():
            print(f"Input '{name}': shape {value.shape}, dtype {value.dtype}")
        return None

def main(text, compare_methods=True, num_iterations=3, export_only=False):
    device = "cpu"
    model = CustomTTSModel(
        model_path=assets_root / model_file,
        config_path=assets_root / config_file,
        style_vec_path=assets_root / style_file,
        device=device,
    )
    
    # Setup ONNX folders
    if not os.path.exists("onnx"):
        os.makedirs("onnx")
    
    # Check if we should just export the model
    if export_only:
        print("Exporting model to ONNX with fixed padding...")
        output_path = "style_bert_vits2_padded.onnx"
        export_to_onnx_with_padding(model, output_path, device, fixed_seq_len=  28)
        print("Export completed.")
        return
    
    if not compare_methods:
        print("Running single inference with ONNX model...")
        # Export if needed
        output_path = "style_bert_vits2_padded.onnx"
        if not os.path.exists(output_path):
            print("ONNX model not found. Exporting first...")
            export_to_onnx_with_padding(model, output_path, device, fixed_seq_len=128)
        
        # Load the ONNX model
        model.fixed_seq_len = 128  # Ensure same padding as export
        
        # Run inference with ONNX model
        print("\nRunning ONNX inference...")
        single_text = text if not isinstance(text, list) else text[0]
        audio = infer_with_padded_onnx(model, output_path, single_text)
        
        if audio is not None:
            print(f"ONNX Inference completed successfully. Audio shape: {audio.shape}")
            
            # Compare with torch.compile version
            print("\nRunning torch.compile inference for comparison...")
            start_time = time.time()
            style_vector = np.zeros((256), dtype=np.float32)
            audio_torch = model._compiled_infer_implementation(
                text=single_text,
                style_vector=style_vector,
                sdp_ratio=DEFAULT_SDP_RATIO,
                noise=DEFAULT_NOISE,
                noisew=DEFAULT_NOISEW,
                length=DEFAULT_LENGTH,
                speaker_id=0,
                language=Languages.JP
            )
            torch_time = time.time() - start_time
            print(f"torch.compile inference time: {torch_time:.4f} seconds")
            print(f"torch.compile audio shape: {audio_torch.shape}")
        else:
            print("ONNX inference failed")
        return
    
    print(f"Model initialized. Running inference performance comparison...")
    test_texts = [
        text,
        "元気ですか？",
        "hello how are you"
    ]
    
    # Export ONNX model for comparison
    output_path = "style_bert_vits2_padded.onnx"
    if not os.path.exists(output_path):
        print("ONNX model not found. Exporting first...")
        export_to_onnx_with_padding(model, output_path, device, fixed_seq_len=128)
    
    # Compare performance between torch.compile and ONNX
    print("\nComparing performance between torch.compile and ONNX...")
    model.fixed_seq_len = 128  # Ensure same padding as export
    
    torch_times = []
    onnx_times = []
    
    for text_item in test_texts:
        print(f"\nText: '{text_item[:30]}{'...' if len(text_item) > 30 else ''}'")
        
        # First measure torch.compile
        torch_results = []
        for i in range(num_iterations):
            style_vector = np.zeros((256), dtype=np.float32)
            start_time = time.time()
            try:
                audio_torch = model._compiled_infer_implementation(
                    text=text_item,
                    style_vector=style_vector,
                    sdp_ratio=DEFAULT_SDP_RATIO,
                    noise=DEFAULT_NOISE,
                    noisew=DEFAULT_NOISEW,
                    length=DEFAULT_LENGTH,
                    speaker_id=0,
                    language=Languages.JP
                )
                elapsed = time.time() - start_time
                torch_results.append(elapsed)
                if i == 0:
                    print(f"torch.compile: {elapsed:.4f} seconds, audio shape: {audio_torch.shape}")
            except Exception as e:
                print(f"torch.compile error: {e}")
                break
        
        if torch_results:
            avg_torch = sum(torch_results) / len(torch_results)
            torch_times.append(avg_torch)
            if len(torch_results) > 1:
                print(f"torch.compile average: {avg_torch:.4f} seconds")
        
        # Now measure ONNX performance
        onnx_results = []
        for i in range(num_iterations):
            start_time = time.time()
            try:
                audio_onnx = infer_with_padded_onnx(model, output_path, text_item)
                if audio_onnx is not None:
                    elapsed = time.time() - start_time
                    onnx_results.append(elapsed)
                    if i == 0:
                        print(f"ONNX: {elapsed:.4f} seconds, audio shape: {audio_onnx.shape}")
            except Exception as e:
                print(f"ONNX error: {e}")
                break
        
        if onnx_results:
            avg_onnx = sum(onnx_results) / len(onnx_results)
            onnx_times.append(avg_onnx)
            if len(onnx_results) > 1:
                print(f"ONNX average: {avg_onnx:.4f} seconds")
    
    # Print summary
    print("\nPerformance Summary:")
    if torch_times:
        avg_torch_all = sum(torch_times) / len(torch_times)
        print(f"torch.compile average time: {avg_torch_all:.4f} seconds")
    
    if onnx_times:
        avg_onnx_all = sum(onnx_times) / len(onnx_times)
        print(f"ONNX average time: {avg_onnx_all:.4f} seconds")
        
    if torch_times and onnx_times:
        avg_torch_all = sum(torch_times) / len(torch_times)
        avg_onnx_all = sum(onnx_times) / len(onnx_times)
        if avg_torch_all > avg_onnx_all:
            speedup = avg_torch_all / avg_onnx_all
            print(f"ONNX is {speedup:.2f}x faster than torch.compile")
        else:
            speedup = avg_onnx_all / avg_torch_all
            print(f"torch.compile is {speedup:.2f}x faster than ONNX")
    
    # Performance comparison results are already printed above
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="Text to be converted to speech", default="こんにちは、元気ですか？")
    parser.add_argument("--compare", action="store_true", help="Compare inference method performance")
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations for performance comparison")
    parser.add_argument("--export-only", action="store_true", help="Only export model to ONNX without running inference")
    args = parser.parse_args()
    
    main(
        args.text, 
        args.compare, 
        args.iterations,
        args.export_only
    )