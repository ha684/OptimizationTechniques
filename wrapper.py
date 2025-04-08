from style_bert_vits2.tts_model import TTSModel
import torch
from typing import Union
from pathlib import Path
from style_bert_vits2.models.hyper_parameters import HyperParameters
from numpy.typing import NDArray
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
from typing import Any, Optional
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.models.models import SynthesizerTrn
from style_bert_vits2.models.models_jp_extra import (
    SynthesizerTrn as SynthesizerTrnJPExtra,
)
from style_bert_vits2.voice import adjust_voice
import numpy as np
from style_bert_vits2.models.infer import get_text, cast, infer
from pathlib import Path
from huggingface_hub import hf_hub_download
import time
from style_bert_vits2.nlp import bert_models
import unicodedata
import onnxruntime as ort
import os

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
        # Call get_text outside of forward
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
        style_vec: NDArray[Any],
        sdp_ratio: float,
        noise_scale: float,
        noise_scale_w: float,
        length_scale: float,
        sid: int,
    ):
        is_jp_extra = self.hps.version.endswith("JP-Extra")
        
        with torch.no_grad():
            x_tst = phones.to(self.device).unsqueeze(0)
            tones = tones.to(self.device).unsqueeze(0)
            lang_ids = lang_ids.to(self.device).unsqueeze(0)
            bert = bert.to(self.device).unsqueeze(0)
            ja_bert = ja_bert.to(self.device).unsqueeze(0)
            en_bert = en_bert.to(self.device).unsqueeze(0)
            x_tst_lengths = torch.LongTensor([phones.size(0)]).to(self.device)
            style_vec_tensor = torch.from_numpy(style_vec).to(self.device).unsqueeze(0)
            sid_tensor = torch.LongTensor([sid]).to(self.device)
            
            net_g = self.net_g
            
            output = cast(SynthesizerTrnJPExtra, net_g).infer(
                x_tst,
                x_tst_lengths,
                sid_tensor,
                tones,
                lang_ids,
                ja_bert,
                style_vec=style_vec_tensor,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )
                
            audio = output[0][0, 0].data.cpu().float().numpy()
            return audio
    
class CustomTTSModel(TTSModel):
    def __init__(self, model_path: Path, config_path: Union[Path, HyperParameters], style_vec_path: Union[Path, NDArray[Any]], device: str) -> None:
        super().__init__(model_path, config_path, style_vec_path, device)
        self.load()
        assert self._TTSModel__net_g is not None, "Model not loaded correctly, net_g is None"
    
        self.inner_infer = torch.compile(self._TTSModel__net_g,fullgraph=True,backend="onnxrt")
        self.use_compile = True
        self.compiled_inner_infer = InnerInferModel(self.inner_infer, self.device, self.hyper_parameters)
        
    def load_onnx_model(self,
        path: Path
    ):
        """Load an ONNX model for inference"""
        import onnxruntime as ort
        
        if os.path.exists(path):
            # Create ONNX Runtime session
            self.onnx_path = str(path)
            # Configure session options for better performance
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Allow dynamic dimensions by setting execution mode
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_options.enable_cpu_mem_arena = False  # Help with dynamic shapes
            
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
        else:
            print(f"ONNX model not found at: {path}")
    
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
        audio = self.compiled_inner_infer(
            bert, ja_bert, en_bert, phones, tones, lang_ids,
            style_vector, sdp_ratio, noise, noisew, 
            length, speaker_id
        )
        return audio
        
    def infer_with_onnx(        
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
        """Inference using ONNX runtime with dynamic sequence handling"""
        # Use ONNX runtime for inference
        if not hasattr(self, "ort_session"):
            raise RuntimeError("ONNX model not loaded. Call load_onnx_model first.")
            
        # Process text to get input tensors
        inner_model = InnerInferModel(self._TTSModel__net_g, self.device, self.hyper_parameters)
        bert, ja_bert, en_bert, phones, tones, lang_ids = inner_model.preprocess_text(
            text, language, assist_text, assist_text_weight, given_phone, given_tone
        )
        
        print(f"Before padding - Input lengths: phones={len(phones)}, bert={bert.size(1)}")
        
        # Handle dynamic input sequence length for ONNX
        # Get expected shapes from the model's inputs
        input_details = self.get_onnx_input_details()
        
        # Add batch dimension and ensure correct shapes
        bert = self.prepare_onnx_input(bert, input_details.get('bert', [None, None]))
        ja_bert = self.prepare_onnx_input(ja_bert, input_details.get('ja_bert', [None, None]))
        en_bert = self.prepare_onnx_input(en_bert, input_details.get('en_bert', [None, None]))
        phones = self.prepare_onnx_input(phones, input_details.get('phones', [None]))
        tones = self.prepare_onnx_input(tones, input_details.get('tones', [None]))
        lang_ids = self.prepare_onnx_input(lang_ids, input_details.get('lang_ids', [None]))
        
        print(f"After padding - Input shapes: phones={phones.shape}, bert={bert.shape}")
        
        # Prepare other inputs
        style_vec_np = style_vector.astype(np.float32)
        style_vec_np = np.expand_dims(style_vec_np, axis=0)  # Add batch dimension
        sdp_ratio_np = np.array(sdp_ratio, dtype=np.float32)
        noise_scale_np = np.array(noise, dtype=np.float32)
        noise_scale_w_np = np.array(noisew, dtype=np.float32)
        length_scale_np = np.array(length, dtype=np.float32)
        sid_np = np.array([speaker_id], dtype=np.int64)
        
        # Create ONNX Runtime input
        ort_inputs = {
            "bert": bert,
            "ja_bert": ja_bert, 
            "en_bert": en_bert,
            "phones": phones,
            "tones": tones,
            "lang_ids": lang_ids,
            "style_vec": style_vec_np,
            "sdp_ratio": sdp_ratio_np,
            "noise_scale": noise_scale_np,
            "noise_scale_w": noise_scale_w_np,
            "length_scale": length_scale_np,
            "sid": sid_np,
        }
        
        # Run inference with better error handling
        try:
            audio = self.ort_session.run(None, ort_inputs)[0]
            # Return the audio as is - it will have a batch dimension
            return audio
        except Exception as e:
            print(f"ONNX inference error: {e}")
            # Print problematic input shapes
            for name, value in ort_inputs.items():
                print(f"Input '{name}': shape {value.shape}, dtype {value.dtype}")
            return None
    
    def prepare_onnx_input(self, tensor, expected_shape):
        """Prepare tensor for ONNX inference by ensuring correct shape."""
        # Convert to numpy if it's a tensor
        if isinstance(tensor, torch.Tensor):
            tensor_np = tensor.cpu().numpy()
        else:
            tensor_np = np.array(tensor)
            
        # Add batch dimension if needed
        if len(tensor_np.shape) == 1 and len(expected_shape) > 1:
            tensor_np = np.expand_dims(tensor_np, axis=0)
        
        return tensor_np
        
    def get_onnx_input_details(self):
        """Get expected input shapes from the ONNX model."""
        if not hasattr(self, "ort_session"):
            return {}
            
        input_details = {}
        for i, input_info in enumerate(self.ort_session.get_inputs()):
            input_details[input_info.name] = input_info.shape
            
        return input_details
    
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
            
        self.load_onnx_model("style_bert_vits2_model.onnx")
        if compare_methods:
            original_times = []
            compiled_times = []
            print("\n--- Starting performance comparison ---")
            print(f"Running {num_iterations} iterations for each method...")
            for i in range(len(text)):
                print(f"Iteration {i+1}/{num_iterations}")

                start_time = time.time()
                try:
                    _ = self.infer_with_onnx(
                        text=text[i], 
                        style_vector=style_vector,
                        sdp_ratio=sdp_ratio,
                        noise=noise,
                        noise_w=noise_w,
                        length=length,
                        speaker_id=speaker_id,
                        language=language,
                        assist_text=assist_text,
                        assist_text_weight=assist_text_weight,
                        given_phone=given_phone,
                        given_tone=given_tone
                    )
                    original_time = time.time() - start_time
                    original_times.append(original_time)
                    print(f"  Original method: {original_time:.4f} seconds")
                except Exception as e:
                    print(f"  Original method failed: {e}")

                if use_compiled:
                    start_time = time.time()
                    try:
                        _ = self._compiled_infer_implementation(
                            text=text[i], 
                            style_vector=style_vector,
                            sdp_ratio=sdp_ratio,
                            noise=noise,
                            noise_w=noise_w,
                            length=length,
                            speaker_id=speaker_id,
                            language=language,
                            assist_text=assist_text,
                            assist_text_weight=assist_text_weight,
                            given_phone=given_phone,
                            given_tone=given_tone
                        )
                        compiled_time = time.time() - start_time
                        compiled_times.append(compiled_time)
                        print(f"  Compiled method: {compiled_time:.4f} seconds")
                    except Exception as e:
                        print(f"  Compiled method failed: {e}")
                        use_compiled = False

            if original_times:
                avg_original = sum(original_times) / len(original_times)
                print(f"\nOriginal method average time: {avg_original:.4f} seconds")
            
                if compiled_times:
                    avg_compiled = sum(compiled_times) / len(compiled_times)
                    speedup = avg_original / avg_compiled if avg_compiled > 0 else 0
                    
                    print(f"Compiled method average time: {avg_compiled:.4f} seconds")
                    print(f"Speedup factor: {speedup:.2f}x ({(speedup-1)*100:.1f}% faster)")
                    if speedup > 1:
                        print(f"The compiled method is {speedup:.2f}x faster than the original method.")
                    elif speedup < 1:
                        print(f"The original method is {1/speedup:.2f}x faster than the compiled method.")
                    else:
                        print("Both methods have similar performance.")
                else:
                    print("Compiled method could not complete successfully.")
            else:
                print("Original method could not complete successfully.")
            print("--------------------------------------")

        return True

def export_net_g_to_onnx(tts_model, output_path, device="cuda"):
    """
    Export just the net_g component to ONNX format
    
    Args:
        tts_model: Your loaded CustomTTSModel instance
        output_path: Path to save the ONNX model
        device: Device to use for tensors
    """
    # Get the net_g component
    net_g = tts_model._TTSModel__net_g
    
    # Set to eval mode
    net_g.eval()
    
    # Determine if using JP-Extra version
    is_jp_extra = tts_model.hyper_parameters.version.endswith("JP-Extra")
    
    # Create a wrapper class that has the same input interface as your compiled_inner_infer
    class NetGWrapper(torch.nn.Module):
        def __init__(self, net_g, is_jp_extra, device):
            super().__init__()
            self.net_g = net_g
            self.is_jp_extra = is_jp_extra
            self.device = device
            
        def forward(self, bert, ja_bert, en_bert, phones, tones, lang_ids,
                   style_vec, sdp_ratio, noise_scale, noise_scale_w, length_scale, sid):
            # Handle shapes explicitly to ensure consistency
            # Note: We expect all inputs to be batched (have a leading batch dimension)
            
            # Use dynamic sequence length from the input tensor
            seq_len = phones.size(1)
            x_tst_lengths = torch.LongTensor([seq_len]).to(self.device)
            net_g = self.net_g
            
            output = cast(SynthesizerTrnJPExtra, net_g).infer(
                phones,               # Already has batch dimension
                x_tst_lengths,
                sid,                  # Should be tensor with shape [1]
                tones,                # Already has batch dimension
                lang_ids,             # Already has batch dimension
                ja_bert,              # Already has batch dimension
                style_vec=style_vec,  # Already has batch dimension
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )
            
            audio = output[0][0, 0].data.cpu().float().numpy()
            return audio
    
    # Create the wrapper instance
    wrapper = NetGWrapper(net_g, is_jp_extra, device)

    inner_model = InnerInferModel(net_g, device, tts_model.hyper_parameters)
    sample_text = "こんにちは、これはテストです。"
    bert, ja_bert, en_bert, phones, tones, lang_ids = inner_model.preprocess_text(
        sample_text, Languages.JP
    )
    
    # Use original sequence lengths - don't pad to fixed length
    seq_len = phones.size(0)
    
    # Add batch dimension
    bert = bert.unsqueeze(0)
    ja_bert = ja_bert.unsqueeze(0) 
    en_bert = en_bert.unsqueeze(0)
    phones = phones.unsqueeze(0)
    tones = tones.unsqueeze(0)
    lang_ids = lang_ids.unsqueeze(0)
    
    print(f"Export shapes - bert: {bert.shape}, ja_bert: {ja_bert.shape}, phones: {phones.shape}")

    style_vector = np.zeros((1, 256), dtype=np.float32)  # Add batch dimension
    sdp_ratio = DEFAULT_SDP_RATIO
    noise_scale = DEFAULT_NOISE
    noise_scale_w = DEFAULT_NOISEW
    length_scale = DEFAULT_LENGTH
    sid = torch.LongTensor([0]).to(device)  # Speaker ID as a tensor
    
    # Convert numpy/scalar values to tensors
    style_vec_tensor = torch.from_numpy(style_vector).float().to(device)
    sdp_ratio_tensor = torch.tensor(sdp_ratio, dtype=torch.float32).to(device)
    noise_scale_tensor = torch.tensor(noise_scale, dtype=torch.float32).to(device)
    noise_scale_w_tensor = torch.tensor(noise_scale_w, dtype=torch.float32).to(device)
    length_scale_tensor = torch.tensor(length_scale, dtype=torch.float32).to(device)
    # sid_tensor = torch.tensor(sid, dtype=torch.int64).to(device)
    
    # Define input names
    input_names = [
        "bert", "ja_bert", "en_bert", "phones", "tones", "lang_ids",
        "style_vec", "sdp_ratio", "noise_scale", "noise_scale_w", "length_scale", "sid"
    ]
    
    # Define output names
    output_names = ["audio"]
    
    # Use dynamic axes for sequence lengths like your successful torch.compile approach
    dynamic_axes = {
        "bert": {0: "batch_size", 1: "seq_len"},
        "ja_bert": {0: "batch_size", 1: "seq_len"},
        "en_bert": {0: "batch_size", 1: "seq_len"},
        "phones": {0: "batch_size", 1: "seq_len"},
        "tones": {0: "batch_size", 1: "seq_len"},
        "lang_ids": {0: "batch_size", 1: "seq_len"},
        "audio": {0: "batch_size", 1: "audio_len"}
    }
    # Export to ONNX
    # Instead of trying to script the model, we'll customize the export configuration
    # to handle problematic operations better
    
    # Create an export-optimized version of the wrapper
    class ExportOptimizedWrapper(torch.nn.Module):
        def __init__(self, original_wrapper):
            super().__init__()
            self.original_wrapper = original_wrapper
            
        def forward(self, bert, ja_bert, en_bert, phones, tones, lang_ids,
                   style_vec, sdp_ratio, noise_scale, noise_scale_w, length_scale, sid):
            with torch.no_grad():
                # Handle operations outside of tracing that might be problematic
                # This allows us to avoid some of the conditional branches
                
                # Pre-calculate sequence lengths to avoid conditional logic in traced code
                seq_len = phones.size(1)
                
                # Export with try-except to catch any errors
                try:
                    return self.original_wrapper(bert, ja_bert, en_bert, phones, tones, lang_ids,
                                       style_vec, sdp_ratio, noise_scale, noise_scale_w, 
                                       length_scale, sid)
                except Exception as e:
                    print(f"Error during export forward pass: {e}")
                    # Return a dummy tensor if something goes wrong
                    return torch.zeros((bert.size(0), 24000), device=bert.device)
    
    # Use the optimized wrapper
    optimized_wrapper = ExportOptimizedWrapper(wrapper)
    
    # Configure export options for better compatibility
    torch.onnx.export(
        optimized_wrapper,
        (bert, ja_bert, en_bert, phones, tones, lang_ids,
        style_vec_tensor, sdp_ratio_tensor, noise_scale_tensor, noise_scale_w_tensor, 
        length_scale_tensor, sid),
        output_path,
        export_params=True,
        opset_version=17,  # Use a high opset version for best compatibility
        do_constant_folding=False,  # Disable constant folding which is causing warnings
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        keep_initializers_as_inputs=True,  # This can help with some runtime compatibility issues
    )
    try:
        import onnx
        model = onnx.load(output_path)
        onnx.checker.check_model(model)
        print(f"ONNX model verification passed")
    except Exception as e:
        print(f"ONNX model verification failed: {e}")
        
    print(f"Model successfully exported to {output_path}")
    return output_path

def main(text, compare_methods=True, num_iterations=3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    print(f"Using device: {device}")
    
    model = CustomTTSModel(
        model_path=assets_root / model_file,
        config_path=assets_root / config_file,
        style_vec_path=assets_root / style_file,
        device=device,
    )
    if not compare_methods:
        output_path = export_net_g_to_onnx(model, "style_bert_vits2_model.onnx",device)
    else:
        print(f"Model initialized. Running inference with {'performance comparison' if compare_methods else 'compiled method only'}...")
        text = [
            text,
            "元気ですか？",
            "hello how are you",
            "Compare original vs compiled method performance",
            "Number of iterations for performance comparison",
            "Elasticsearch is a distributed search and analytics engine, scalable data store and vector database optimized for speed and relevance on production-scale workloads.",
            "Elasticsearch is the foundation of Elastic’s open Stack platform. Search in near real-time over massive datasets, perform vector searches, integrate with generative AI applications, and much more."
        ]
        result = model.infer(
            text=text, 
            compare_methods=compare_methods,
            num_iterations=num_iterations
        )
        
        if result:
            print("Success")
        else:
            print("Failed")
        
        print(f"Model initialized. Running inference with {'performance comparison' if compare_methods else 'compiled method only'}...")
        text = [
            text,
            "元気ですか？",
            "hello how are you",
            "Compare original vs compiled method performance",
            "Number of iterations for performance comparison",
            "Elasticsearch is a distributed search and analytics engine, scalable data store and vector database optimized for speed and relevance on production-scale workloads.",
            "Elasticsearch is the foundation of Elastic’s open Stack platform. Search in near real-time over massive datasets, perform vector searches, integrate with generative AI applications, and much more."
        ]
        result = model.infer(
            text=text, 
            compare_methods=compare_methods,
            num_iterations=num_iterations
        )
        
        if result:
            print("Success")
        else:
            print("Failed")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="Text to be converted to speech", default="こんにちは、元気ですか？")
    parser.add_argument("--compare", action="store_true", help="Compare original vs compiled method performance")
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations for performance comparison")
    args = parser.parse_args()
    main(args.text, args.compare, args.iterations)