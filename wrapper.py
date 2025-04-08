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

class ONNXExportWrapper(torch.nn.Module):
    """Simplified wrapper model for ONNX export to avoid control flow issues"""
    def __init__(self, net_g):
        super().__init__()
        self.net_g = net_g
        
    def forward(self, 
                x_tst, x_tst_lengths, sid, tones, lang_ids, ja_bert, 
                style_vec, sdp_ratio, noise_scale, noise_scale_w, length_scale):
        """Forward pass with flattened inputs to avoid control flow in ONNX export"""
        # Call net_g's infer method directly with all inputs
        output = self.net_g.infer(
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
        
        # Create ONNX inference setup
        self.onnx_path = assets_root / "model.onnx"
        self.onnx_session = None
        self.inner_model_for_export = InnerInferModel(self._TTSModel__net_g, self.device, self.hyper_parameters)
        
        # Create export wrapper for simplified ONNX export
        self.export_wrapper = ONNXExportWrapper(self._TTSModel__net_g)
    
    def export_to_onnx(self, force_export=False):
        """Export the model to ONNX format"""
        if self.onnx_path.exists() and not force_export:
            print(f"ONNX model already exists at {self.onnx_path}, skipping export")
            return self.onnx_path
        
        print(f"Exporting model to ONNX at {self.onnx_path}...")
        
        # Prepare dummy inputs for tracing
        dummy_text = "こんにちは"
        bert, ja_bert, en_bert, phones, tones, lang_ids = self.inner_model_for_export.preprocess_text(
            dummy_text, Languages.JP
        )
        
        # Set float32 data type for all tensors to ensure compatibility
        bert = bert.float()
        ja_bert = ja_bert.float()
        en_bert = en_bert.float()
        phones = phones.float()
        tones = tones.float()
        lang_ids = lang_ids.float()
        
        style_id = self.style2id[DEFAULT_STYLE]
        style_vector = self._TTSModel__get_style_vector(style_id, DEFAULT_STYLE_WEIGHT)
        style_vec_tensor = torch.from_numpy(style_vector).to(self.device).float()
        sid_tensor = torch.LongTensor([0]).to(self.device)
        
        # Prepare direct inputs for the simplified wrapper model
        x_tst = phones.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(self.device)
        tones_input = tones.unsqueeze(0)
        lang_ids_input = lang_ids.unsqueeze(0)
        ja_bert_input = ja_bert.unsqueeze(0)
        style_vec_input = style_vec_tensor.unsqueeze(0)
        
        # Scalar inputs as torch tensors with consistent dtype
        sdp_ratio_tensor = torch.tensor(DEFAULT_SDP_RATIO, dtype=torch.float32, device=self.device)
        noise_scale_tensor = torch.tensor(DEFAULT_NOISE, dtype=torch.float32, device=self.device)
        noise_scale_w_tensor = torch.tensor(DEFAULT_NOISEW, dtype=torch.float32, device=self.device)
        length_scale_tensor = torch.tensor(DEFAULT_LENGTH, dtype=torch.float32, device=self.device)
        
        # Set inputs for ONNX export - using the simplified wrapper
        dummy_inputs = (
            x_tst,
            x_tst_lengths,
            sid_tensor,
            tones_input, 
            lang_ids_input,
            ja_bert_input,
            style_vec_input,
            sdp_ratio_tensor,
            noise_scale_tensor,
            noise_scale_w_tensor,
            length_scale_tensor
        )
        
        # Export model to ONNX with simplified wrapper
        try:
            # Set export to avoid control flow issues
            torch.onnx.export(
                self.export_wrapper,
                dummy_inputs,
                self.onnx_path,
                export_params=True,
                opset_version=14,  # Try a lower opset version
                do_constant_folding=True,
                input_names=[
                    'x_tst', 'x_tst_lengths', 'sid', 'tones', 'lang_ids', 'ja_bert',
                    'style_vec', 'sdp_ratio', 'noise_scale', 'noise_scale_w', 'length_scale'
                ],
                output_names=['audio'],
                dynamic_axes={
                    'x_tst': {1: 'seq_len'},
                    'tones': {1: 'seq_len'},
                    'lang_ids': {1: 'seq_len'},
                    'ja_bert': {1: 'seq_len'},
                    'audio': {0: 'audio_len'}
                }
            )
            print(f"ONNX export successful: {self.onnx_path}")
            return self.onnx_path
        except Exception as e:
            print(f"ONNX export failed: {e}")
            torch.onnx.export(
                self.export_wrapper,
                dummy_inputs,
                self.onnx_path,
                export_params=True,
                opset_version=11,  # Try an even lower opset version
                do_constant_folding=False,  # Disable constant folding
                input_names=[
                    'x_tst', 'x_tst_lengths', 'sid', 'tones', 'lang_ids', 'ja_bert',
                    'style_vec', 'sdp_ratio', 'noise_scale', 'noise_scale_w', 'length_scale'
                ],
                output_names=['audio'],
                dynamic_axes={
                    'x_tst': {1: 'seq_len'},
                    'tones': {1: 'seq_len'},
                    'lang_ids': {1: 'seq_len'},
                    'ja_bert': {1: 'seq_len'},
                    'audio': {0: 'audio_len'}
                }
            )
            print(f"ONNX export with fallback settings successful: {self.onnx_path}")
            return self.onnx_path
    
    def load_onnx_session(self):
        """Load the ONNX model for inference"""
        if not self.onnx_path.exists() or self.onnx_session is None:
            path = self.export_to_onnx()
            if path is None:
                print("Failed to export ONNX model")
                return None
            
            try:
                # Create ONNX Runtime session
                providers = ['CPUExecutionProvider']
                if self.device.startswith('cuda'):
                    providers.insert(0, 'CUDAExecutionProvider')
                
                # Initialize session with options to fix the data type issues
                session_options = ort.SessionOptions()
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                
                self.onnx_session = ort.InferenceSession(
                    str(self.onnx_path), 
                    providers=providers,
                    sess_options=session_options
                )
                print("ONNX session initialized successfully")
            except Exception as e:
                print(f"Failed to load ONNX session: {e}")
                return None
        return self.onnx_session
    
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

    def _onnx_infer_implementation(
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
        """ONNX-based inference implementation using the simplified model"""
        session = self.load_onnx_session()
        if session is None:
            raise RuntimeError("Failed to load ONNX session")
        
        # Preprocess text (reuse from compiled model)
        bert, ja_bert, en_bert, phones, tones, lang_ids = self.inner_model_for_export.preprocess_text(
            text, language, assist_text, assist_text_weight, given_phone, given_tone
        )
        
        # Convert all tensors to float32 to prevent type mismatches
        x_tst = phones.float().unsqueeze(0).cpu().numpy()
        x_tst_lengths = torch.LongTensor([phones.size(0)]).cpu().numpy()
        tones_input = tones.float().unsqueeze(0).cpu().numpy()
        lang_ids_input = lang_ids.float().unsqueeze(0).cpu().numpy()
        ja_bert_input = ja_bert.float().unsqueeze(0).cpu().numpy()
        
        # Ensure style vector is float32
        style_vec_input = torch.from_numpy(style_vector).float().unsqueeze(0).cpu().numpy()
        sid_input = torch.LongTensor([speaker_id]).cpu().numpy()
        
        # Convert scalar parameters to numpy arrays with explicit dtype
        sdp_ratio_input = np.array(sdp_ratio, dtype=np.float32)
        noise_scale_input = np.array(noise, dtype=np.float32)
        noise_scale_w_input = np.array(noisew, dtype=np.float32)
        length_scale_input = np.array(length, dtype=np.float32)
        
        # Prepare inputs for ONNX Runtime
        onnx_inputs = {
            'x_tst': x_tst,
            'x_tst_lengths': x_tst_lengths,
            'sid': sid_input,
            'tones': tones_input,
            'lang_ids': lang_ids_input,
            'ja_bert': ja_bert_input,
            'style_vec': style_vec_input,
            'sdp_ratio': sdp_ratio_input,
            'noise_scale': noise_scale_input,
            'noise_scale_w': noise_scale_w_input,
            'length_scale': length_scale_input
        }
        
        # Debug info for checking data types
        for name, value in onnx_inputs.items():
            print(f"ONNX input '{name}': shape={value.shape}, dtype={value.dtype}")
        
        # Run inference
        try:
            audio = session.run(['audio'], onnx_inputs)[0]
            return audio
        except Exception as e:
            print(f"ONNX inference error: {e}")
            # Print expected input details from ONNX model
            print("\nONNX model expects these inputs:")
            for input_info in session.get_inputs():
                print(f"  {input_info.name}: shape={input_info.shape}, type={input_info.type}")
            raise

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
            
            # Ensure ONNX model is exported
            self.export_to_onnx()
            session = self.load_onnx_session()
            
            if session is None:
                print("Failed to initialize ONNX session, skipping ONNX comparison")
                onnx_compare = False
            else:
                onnx_compare = True
            
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
                            if i == 0:  # Only print for first iteration
                                print(f"  Torch Compile method: {compiled_time:.4f} seconds")
                        except Exception as e:
                            print(f"  Torch Compile method failed: {e}")
                            break
                    
                    if compiled_results:
                        avg_time = sum(compiled_results) / len(compiled_results)
                        compiled_times.append(avg_time)
                        if len(compiled_results) > 1:
                            print(f"  Torch Compile average ({len(compiled_results)} runs): {avg_time:.4f} seconds")
                
                # Then run ONNX method if available
                if onnx_compare:
                    onnx_results = []
                    for i in range(num_iterations):
                        try:
                            start_time = time.time()
                            _ = self._onnx_infer_implementation(
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
                            onnx_time = time.time() - start_time
                            onnx_results.append(onnx_time)
                            if i == 0:  # Only print for first iteration
                                print(f"  ONNX method: {onnx_time:.4f} seconds")
                        except Exception as e:
                            if i == 0:  # Only print error for first iteration
                                print(f"  ONNX method failed: {e}")
                            break
                    
                    if onnx_results:
                        avg_time = sum(onnx_results) / len(onnx_results)
                        onnx_times.append(avg_time)
                        if len(onnx_results) > 1:
                            print(f"  ONNX average ({len(onnx_results)} runs): {avg_time:.4f} seconds")
                    
                print("  " + "-" * 40)
                
            print("\n--- Performance Summary ---")
            if compiled_times:
                avg_compiled = sum(compiled_times) / len(compiled_times)                    
                print(f"Torch Compile method average time: {avg_compiled:.4f} seconds")
            if onnx_times:
                avg_onnx = sum(onnx_times) / len(onnx_times)
                print(f"ONNX method average time: {avg_onnx:.4f} seconds")
                
                if compiled_times:
                    speedup = avg_compiled / avg_onnx if avg_onnx > 0 else float('inf')
                    print(f"ONNX speedup vs Torch Compile: {speedup:.2f}x")
            print("--------------------------------------")
            
            return True, np.array([])  # Just return success flag and empty array
        
        # If not comparing, use ONNX for actual inference
        try:
            audio = self._onnx_infer_implementation(
                text=text if not isinstance(text, list) else text[0], 
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
            return True, audio
        except Exception as e:
            print(f"ONNX inference failed: {e}. Falling back to torch compile method.")
            try:
                audio = self._compiled_infer_implementation(
                    text=text if not isinstance(text, list) else text[0], 
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
                return True, audio
            except Exception as e2:
                print(f"Torch compile fallback also failed: {e2}")
                return False, np.array([])

def main(text, compare_methods=True, num_iterations=3, export_only=False):
    device = "cpu"
    model = CustomTTSModel(
        model_path=assets_root / model_file,
        config_path=assets_root / config_file,
        style_vec_path=assets_root / style_file,
        device=device,
    )
    
    if export_only:
        print("Exporting model to ONNX...")
        model.export_to_onnx()
        print("Export completed.")
        return
    
    if not compare_methods:
        print("Running single inference with ONNX model...")
        result, audio = model.infer(
            text=text if not isinstance(text, list) else text[0],
            compare_methods=False
        )
        if result:
            print(f"Inference completed successfully. Audio shape: {audio.shape}")
        else:
            print("Inference failed")
        return
    
    print(f"Model initialized. Running inference performance comparison...")
    test_texts = [
        text,
        "元気ですか？",
        "hello how are you"
    ]
    
    result, _ = model.infer(
        text=test_texts, 
        compare_methods=compare_methods,
        num_iterations=num_iterations
    )
    
    if result:
        print("Performance comparison completed successfully")
    else:
        print("Performance comparison failed")
    
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