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
            
            if is_jp_extra:
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
            else:
                output = cast(SynthesizerTrn, net_g).infer(
                    x_tst,
                    x_tst_lengths,
                    sid_tensor,
                    tones,
                    lang_ids,
                    bert,
                    ja_bert,
                    en_bert,
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
        self.inner_infer_model = InnerInferModel(self._TTSModel__net_g, self.device, self.hyper_parameters)
        try:
            self.compiled_inner_infer = torch.compile(self.inner_infer_model)
        except Exception as e:
            print(f"Failed to compile model: {e}")
            self.compiled_inner_infer = None
    
    def _original_infer_implementation(
        self,
        text: str,
        style_vector: NDArray[Any],
        sdp_ratio: float,
        noise: float,
        noise_w: float,
        length: float,
        speaker_id: int,
        language: Languages,
        assist_text: Optional[str] = None,
        assist_text_weight: float = DEFAULT_ASSIST_TEXT_WEIGHT,
        given_phone: Optional[list[str]] = None,
        given_tone: Optional[list[int]] = None,
    ) -> NDArray[Any]:
        """The original infer implementation using the imported infer function"""
        with torch.no_grad():
            audio = infer(
                text=text,
                sdp_ratio=sdp_ratio,
                noise_scale=noise,
                noise_scale_w=noise_w,
                length_scale=length,
                sid=speaker_id,
                language=language,
                hps=self.hyper_parameters,
                net_g=self._TTSModel__net_g,
                device=self.device,
                assist_text=assist_text,
                assist_text_weight=assist_text_weight,
                style_vec=style_vector,
                given_phone=given_phone,
                given_tone=given_tone,
            )
            return audio
    
    def _compiled_infer_implementation(
        self,
        text: str,
        style_vector: NDArray[Any],
        sdp_ratio: float,
        noise: float,
        noise_w: float,
        length: float,
        speaker_id: int,
        language: Languages,
        assist_text: Optional[str] = None,
        assist_text_weight: float = DEFAULT_ASSIST_TEXT_WEIGHT,
        given_phone: Optional[list[str]] = None,
        given_tone: Optional[list[int]] = None,
    ) -> NDArray[Any]:
        """The compiled infer implementation using torch.compile"""
        with torch.no_grad():
            if self._TTSModel__net_g is None:
                self.load()
                if not hasattr(self, 'inner_infer_model') or self.inner_infer_model.net_g is None:
                    self.inner_infer_model = InnerInferModel(self._TTSModel__net_g, self.device, self.hyper_parameters)
                    self.compiled_inner_infer = torch.compile(self.inner_infer_model)
            
            bert, ja_bert, en_bert, phones, tones, lang_ids = self.inner_infer_model.preprocess_text(
                text, language, assist_text, assist_text_weight, given_phone, given_tone
            )
            
            try:
                audio = self.compiled_inner_infer(
                    bert, ja_bert, en_bert, phones, tones, lang_ids,
                    style_vector, sdp_ratio, noise, noise_w, 
                    length, speaker_id
                )
                return audio
            except Exception as e:
                print(f"Error in compiled inference: {e}")
                print("Falling back to original implementation...")
                return self._original_infer_implementation(
                    text=text, 
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
        use_compiled: bool = True
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
            print("\n--- Starting performance comparison ---")
            
            print("Warming up original method...")
            try:
                _ = self._original_infer_implementation(
                    text=text, 
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
            except Exception as e:
                print(f"Error in original method warm-up: {e}")
            
            print("Warming up compiled method...")
            try:
                _ = self._compiled_infer_implementation(
                    text=text, 
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
            except Exception as e:
                print(f"Error in compiled method warm-up: {e}")
                print("Disabling compiled method for comparison")
                use_compiled = False
            
            # Run multiple iterations to get meaningful timing data
            print(f"Running {num_iterations} iterations for each method...")
            for i in range(num_iterations):
                print(f"Iteration {i+1}/{num_iterations}")
                
                # Time original inference
                start_time = time.time()
                try:
                    _ = self._original_infer_implementation(
                        text=text, 
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
                
                # Time compiled inference
                if use_compiled:
                    start_time = time.time()
                    try:
                        _ = self._compiled_infer_implementation(
                            text=text, 
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
    
def main(text, compare_methods=True, num_iterations=3):
    model = CustomTTSModel(
        model_path=assets_root / model_file,
        config_path=assets_root / config_file,
        style_vec_path=assets_root / style_file,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    print(f"Model initialized. Running inference with {'performance comparison' if compare_methods else 'compiled method only'}...")
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