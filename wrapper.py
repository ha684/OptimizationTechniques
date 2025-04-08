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
    
        self.inner_infer = torch.compile(self._TTSModel__net_g, fullgraph=True, backend="onnxrt")
        self.use_compile = True
        self.compiled_inner_infer = InnerInferModel(self.inner_infer, self.device, self.hyper_parameters)
        self.use_onnx = False
        self.ort_session = None
        
    def export_to_onnx(self, export_path: str = "model_assets/tts_model.onnx"):
        """Export the model to ONNX format"""
        print(f"Exporting model to ONNX format at: {export_path}")
        
        # Create a dummy InnerInferModel for export
        export_model = InnerInferModel(self._TTSModel__net_g, self.device, self.hyper_parameters)
        
        # Sample text for tracing
        sample_text = "こんにちは"
        language = Languages.JP
        
        # Get preprocessed inputs
        bert, ja_bert, en_bert, phones, tones, lang_ids = export_model.preprocess_text(
            sample_text, language
        )
        
        # Sample style vector and parameters
        style_id = self.style2id[DEFAULT_STYLE]
        style_vector = self._TTSModel__get_style_vector(style_id, DEFAULT_STYLE_WEIGHT)
        sdp_ratio = DEFAULT_SDP_RATIO
        noise = DEFAULT_NOISE
        noise_w = DEFAULT_NOISEW
        length = DEFAULT_LENGTH
        speaker_id = 0
        
        # Prepare inputs for tracing
        # We'll use these inputs for the export
        x_tst = phones.to(self.device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(self.device)
        tones_tensor = tones.to(self.device).unsqueeze(0)
        lang_ids_tensor = lang_ids.to(self.device).unsqueeze(0)
        bert_tensor = bert.to(self.device).unsqueeze(0)
        ja_bert_tensor = ja_bert.to(self.device).unsqueeze(0)
        style_vec_tensor = torch.from_numpy(style_vector).to(self.device).unsqueeze(0)
        sid_tensor = torch.LongTensor([speaker_id]).to(self.device)
        
        # Define dynamic axes for variable length inputs
        dynamic_axes = {
            'x_tst': {1: 'seq_len'},
            'x_tst_lengths': {0: 'batch'},
            'tones': {1: 'seq_len'},
            'lang_ids': {1: 'seq_len'},
            'bert': {1: 'seq_len'},
            'ja_bert': {1: 'seq_len'},
            'output': {2: 'audio_len'}
        }
        
        # Create a wrapped forward function that matches the expected signature
        def wrapped_forward(x_tst, x_tst_lengths, sid, tones, lang_ids, ja_bert, style_vec, sdp_ratio, noise_scale, noise_scale_w, length_scale):
            with torch.no_grad():
                net_g = self._TTSModel__net_g
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
                return output
        
        # Set up the inputs for export
        inputs = (
            x_tst, 
            x_tst_lengths, 
            sid_tensor,
            tones_tensor,
            lang_ids_tensor,
            ja_bert_tensor,
            style_vec_tensor,
            torch.tensor(sdp_ratio, device=self.device),
            torch.tensor(noise, device=self.device),
            torch.tensor(noise_w, device=self.device),
            torch.tensor(length, device=self.device)
        )
        
        # Make sure the export directory exists
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        
        try:
            # Export the model
            with torch.no_grad():
                torch.onnx.export(
                    wrapped_forward,
                    inputs,
                    export_path,
                    export_params=True,
                    opset_version=17,
                    do_constant_folding=True,
                    input_names=['x_tst', 'x_tst_lengths', 'sid', 'tones', 'lang_ids', 'ja_bert', 'style_vec', 
                                'sdp_ratio', 'noise_scale', 'noise_scale_w', 'length_scale'],
                    output_names=['output'],
                    dynamic_axes=dynamic_axes,
                    verbose=True
                )
            print(f"Successfully exported model to {export_path}")
            return export_path
        except Exception as e:
            print(f"Failed to export model to ONNX: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    def load_onnx_model(self, path: Path):
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
            self.use_onnx = True
            
            # Print model input details
            print(f"Loaded ONNX model from: {path}")
            print("ONNX Model Inputs:")
            for i, input_info in enumerate(self.ort_session.get_inputs()):
                print(f"  {i}: {input_info.name} - {input_info.shape} ({input_info.type})")
            return True
        else:
            print(f"ONNX model not found at: {path}")
            return False
    
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
        """Inference using ONNX Runtime"""
        if self.ort_session is None:
            raise ValueError("ONNX model not loaded")
            
        # Preprocess text using the same preprocessing function
        bert, ja_bert, en_bert, phones, tones, lang_ids = self.compiled_inner_infer.preprocess_text(
            text, language, assist_text, assist_text_weight, given_phone, given_tone
        )
        
        # Prepare inputs for ONNX Runtime session
        x_tst = phones.to(self.device).unsqueeze(0).cpu().numpy()
        x_tst_lengths = np.array([phones.size(0)], dtype=np.int64)
        tones_np = tones.to(self.device).unsqueeze(0).cpu().numpy()
        lang_ids_np = lang_ids.to(self.device).unsqueeze(0).cpu().numpy()
        bert_np = bert.to(self.device).unsqueeze(0).cpu().numpy()
        ja_bert_np = ja_bert.to(self.device).unsqueeze(0).cpu().numpy()
        style_vec_np = style_vector.reshape(1, -1).astype(np.float32)
        sid_np = np.array([speaker_id], dtype=np.int64)
        
        # Create a dictionary of inputs
        ort_inputs = {
            'x_tst': x_tst,
            'x_tst_lengths': x_tst_lengths,
            'sid': sid_np,
            'tones': tones_np,
            'lang_ids': lang_ids_np,
            'ja_bert': ja_bert_np,
            'style_vec': style_vec_np,
            'sdp_ratio': np.array(sdp_ratio, dtype=np.float32),
            'noise_scale': np.array(noise, dtype=np.float32),
            'noise_scale_w': np.array(noisew, dtype=np.float32),
            'length_scale': np.array(length, dtype=np.float32)
        }
        
        # Run inference
        ort_outputs = self.ort_session.run(None, ort_inputs)
        
        # Process outputs
        audio = ort_outputs[0][0, 0]
        return audio
    
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
    
    def _original_infer_implementation(
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
        """The original infer implementation"""
        # This is the original implementation from the TTSModel class
        # Adapted to use the same interface as the compiled version
        bert, ja_bert, en_bert, phones, tones, lang_ids = get_text(
            text,
            language,
            self.hyper_parameters,
            self.device,
            assist_text=assist_text,
            assist_text_weight=assist_text_weight,
            given_phone=given_phone,
            given_tone=given_tone,
        )
        
        with torch.no_grad():
            x_tst = phones.to(self.device).unsqueeze(0)
            tones = tones.to(self.device).unsqueeze(0)
            lang_ids = lang_ids.to(self.device).unsqueeze(0)
            bert = bert.to(self.device).unsqueeze(0)
            ja_bert = ja_bert.to(self.device).unsqueeze(0)
            en_bert = en_bert.to(self.device).unsqueeze(0)
            x_tst_lengths = torch.LongTensor([phones.size(0)]).to(self.device)
            style_vec_tensor = torch.from_numpy(style_vector).to(self.device).unsqueeze(0)
            sid_tensor = torch.LongTensor([speaker_id]).to(self.device)
            
            net_g = self._TTSModel__net_g
            
            output = cast(SynthesizerTrnJPExtra, net_g).infer(
                x_tst,
                x_tst_lengths,
                sid_tensor,
                tones,
                lang_ids,
                ja_bert,
                style_vec=style_vec_tensor,
                sdp_ratio=sdp_ratio,
                noise_scale=noise,
                noise_scale_w=noisew,
                length_scale=length,
            )
                
            audio = output[0][0, 0].data.cpu().float().numpy()
            return audio
    
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
        use_onnx: bool = self.use_onnx
        
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
            original_times = []
            compiled_times = []
            onnx_times = []
            
            print("\n--- Starting performance comparison ---")
            print(f"Running {num_iterations} iterations for each method...")
            
            for i in range(len(text)):
                print(f"\nSentence: {text[i]}")
                
                # Original Method
                start_time = time.time()
                try:
                    _ = self._original_infer_implementation(
                        text=text[i], 
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
                    original_time = time.time() - start_time
                    original_times.append(original_time)
                    print(f"  Original method: {original_time:.4f} seconds")
                except Exception as e:
                    print(f"  Original method failed: {e}")

                # Compiled Method
                if use_compiled:
                    start_time = time.time()
                    try:
                        _ = self._compiled_infer_implementation(
                            text=text[i], 
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
                        compiled_times.append(compiled_time)
                        print(f"  Compiled method: {compiled_time:.4f} seconds")
                    except Exception as e:
                        print(f"  Compiled method failed: {e}")

                # ONNX Method
                if use_onnx:
                    start_time = time.time()
                    try:
                        _ = self._onnx_infer_implementation(
                            text=text[i], 
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
                        onnx_times.append(onnx_time)
                        print(f"  ONNX method: {onnx_time:.4f} seconds")
                    except Exception as e:
                        print(f"  ONNX method failed: {e}")

            # Print summary statistics
            print("\n--- Performance Summary ---")
            
            if original_times:
                avg_original = sum(original_times) / len(original_times)
                print(f"Original method average time: {avg_original:.4f} seconds")
            
                if compiled_times:
                    avg_compiled = sum(compiled_times) / len(compiled_times)
                    speedup_compiled = avg_original / avg_compiled if avg_compiled > 0 else 0
                    
                    print(f"Compiled method average time: {avg_compiled:.4f} seconds")
                    print(f"Compiled speedup: {speedup_compiled:.2f}x ({(speedup_compiled-1)*100:.1f}% faster)")
                
                if onnx_times:
                    avg_onnx = sum(onnx_times) / len(onnx_times)
                    speedup_onnx = avg_original / avg_onnx if avg_onnx > 0 else 0
                    
                    print(f"ONNX method average time: {avg_onnx:.4f} seconds")
                    print(f"ONNX speedup: {speedup_onnx:.2f}x ({(speedup_onnx-1)*100:.1f}% faster)")
                    
                    if compiled_times:
                        # Compare ONNX vs Compiled
                        speedup_onnx_vs_compiled = avg_compiled / avg_onnx if avg_onnx > 0 else 0
                        print(f"ONNX vs Compiled: {speedup_onnx_vs_compiled:.2f}x " +
                              f"({'faster' if speedup_onnx_vs_compiled > 1 else 'slower'})")
            else:
                print("Original method could not complete successfully.")
            print("--------------------------------------")

        return True

def main(text, compare_methods=True, num_iterations=3, export_onnx=True, onnx_path="model_assets/tts_model.onnx"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"  # Force CPU for consistent comparisons
    print(f"Using device: {device}")
    
    model = CustomTTSModel(
        model_path=assets_root / model_file,
        config_path=assets_root / config_file,
        style_vec_path=assets_root / style_file,
        device=device,
    )
    
    # Export model to ONNX if requested
    if export_onnx:
        export_path = model.export_to_onnx(onnx_path)
        if export_path:
            model.load_onnx_model(export_path)
    
    if not compare_methods:
        return
    else:
        print(f"Model initialized. Running inference with {'performance comparison' if compare_methods else 'compiled method only'}...")
        
        # Test sentences
        test_texts = [
            text,
            "元気ですか？",
            "hello how are you",
            "Compare original vs compiled method performance",
            "Elasticsearch is a distributed search and analytics engine."
        ]
        
        result = model.infer(
            text=test_texts, 
            compare_methods=compare_methods,
            num_iterations=num_iterations
        )
        
        if result:
            print("First inference run completed successfully")
        else:
            print("First inference run failed")
        
        # Second run to test consistency and any warmup effects
        result = model.infer(
            text=test_texts, 
            compare_methods=compare_methods,
            num_iterations=num_iterations
        )
        
        if result:
            print("Second inference run completed successfully")
        else:
            print("Second inference run failed")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="Text to be converted to speech", default="こんにちは、元気ですか？")
    parser.add_argument("--compare", action="store_true", help="Compare inference method performance")
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations for performance comparison")
    parser.add_argument("--export_onnx", action="store_true", help="Export model to ONNX format")
    parser.add_argument("--onnx_path", type=str, default="model_assets/tts_model.onnx", help="Path to save/load ONNX model")
    args = parser.parse_args()
    
    main(
        args.text, 
        args.compare, 
        args.iterations,
        args.export_onnx,
        args.onnx_path
    )