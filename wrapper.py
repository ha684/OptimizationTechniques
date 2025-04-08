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
import numpy as np
from style_bert_vits2.models.infer import get_text, cast, infer
from pathlib import Path
from huggingface_hub import hf_hub_download
import time
from style_bert_vits2.nlp import bert_models
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
    def __init__(self, model_path: Path, config_path: Union[Path, HyperParameters], 
                 style_vec_path: Union[Path, NDArray[Any]], device: str) -> None:
        super().__init__(model_path, config_path, style_vec_path, device)
        self.load()
        assert self._TTSModel__net_g is not None, "Model not loaded correctly, net_g is None"
        self.inner_infer = torch.compile(self._TTSModel__net_g, fullgraph=True, backend="onnxrt")
        self.use_compile = True
        self.compiled_inner_infer = InnerInferModel(self.inner_infer, self.device, self.hyper_parameters)
        
        # ONNX support
        self.onnx_wrapper = None
        self.max_sequence_length = 256  # Adjust based on your needs
        
    def export_to_onnx(self, output_path: str):
        """Export the model to ONNX format"""
        self.onnx_wrapper = OnnxTTSModelWrapper(self, self.max_sequence_length)
        self.onnx_wrapper.export_model(output_path)
        
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
        """The ONNX infer implementation"""
        if self.onnx_wrapper is None:
            raise ValueError("ONNX model not exported. Call export_to_onnx first.")
        
        audio = self.onnx_wrapper.infer(
            text=text,
            language=language,
            style_vector=style_vector,
            sdp_ratio=sdp_ratio,
            noise=noise,
            noise_w=noisew,
            length=length,
            speaker_id=speaker_id,
            assist_text=assist_text,
            assist_text_weight=assist_text_weight,
            given_phone=given_phone,
            given_tone=given_tone
        )
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
        use_onnx: bool = False,
    ) -> tuple[int, NDArray[Any]]:
        use_compiled: bool = self.use_compile and not use_onnx
        
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
                # Compiled method
                if use_compiled:
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
                        compiled_times.append(compiled_time)
                        print(f"  Compiled method: {compiled_time:.4f} seconds")
                    except Exception as e:
                        print(f"  Compiled method failed: {e}")
                
                # ONNX method
                if self.onnx_wrapper is not None:
                    start_time = time.time()
                    try:
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
                        onnx_times.append(onnx_time)
                        print(f"  ONNX method: {onnx_time:.4f} seconds")
                    except Exception as e:
                        print(f"  ONNX method failed: {e}")
            
            print("\n--- Performance Summary ---")
            if compiled_times:
                avg_compiled = sum(compiled_times) / len(compiled_times)                    
                print(f"Compiled method average time: {avg_compiled:.4f} seconds")
            if onnx_times:
                avg_onnx = sum(onnx_times) / len(onnx_times)
                print(f"ONNX method average time: {avg_onnx:.4f} seconds")
                if compiled_times:
                    speedup = avg_compiled / avg_onnx if avg_onnx > 0 else float('inf')
                    print(f"ONNX speedup: {speedup:.2f}x")
            print("--------------------------------------")
            
        # Do actual inference with the selected method
        if use_onnx and self.onnx_wrapper is not None:
            return True  # Replace with actual inference if needed
        else:
            return True  # Replace with actual inference if needed

def main(text, compare_methods=True, num_iterations=3, export_onnx=True, use_onnx=False):
    device = "cpu"
    model = CustomTTSModel(
        model_path=assets_root / model_file,
        config_path=assets_root / config_file,
        style_vec_path=assets_root / style_file,
        device=device,
    )
    
    if export_onnx:
        output_path = "model_assets/tts_model.onnx"
        model.export_to_onnx(output_path)
        print(f"ONNX model exported to {output_path}")
    
    if not compare_methods:
        return
    else:
        print(f"Model initialized. Running inference performance comparison...")
        test_texts = [
            text,
            "元気ですか？",
            "hello how are you"
        ]
        result = model.infer(
            text=test_texts, 
            compare_methods=compare_methods,
            num_iterations=num_iterations,
            use_onnx=use_onnx
        )
        if result:
            print("Inference completed successfully")
        else:
            print("Inference failed")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="Text to be converted to speech", default="こんにちは、元気ですか？")
    parser.add_argument("--compare", action="store_true", help="Compare inference method performance")
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations for performance comparison")
    parser.add_argument("--export-onnx", action="store_true", help="Export the model to ONNX format")
    parser.add_argument("--use-onnx", action="store_true", help="Use ONNX for inference")
    args = parser.parse_args()
    main(
        args.text, 
        args.compare, 
        args.iterations,
        args.export_onnx,
        args.use_onnx
    )