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
        
        try:
            self.inner_infer = torch.compile(self._TTSModel__net_g,fullgraph=True,backend="onnxrt")
            self.use_compile = True
            self.compiled_inner_infer = InnerInferModel(self.inner_infer, self.device, self.hyper_parameters)

        except Exception as e:
            print(f"Failed to compile model: {e}")
            self.compiled_inner_infer = None
            self.use_compile = False
            self.use_compile = False
                
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
        bert, ja_bert, en_bert, phones, tones, lang_ids = self.compiled_inner_infer.preprocess_text(
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
            original_times = []
            compiled_times = []
            print("\n--- Starting performance comparison ---")
            print(f"Running {num_iterations} iterations for each method...")
            for i in range(len(text)):
                print(f"Iteration {i+1}/{num_iterations}")

                start_time = time.time()
                try:
                    _ = self._original_infer_implementation(
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
        def __init__(self, net_g):
            super().__init__()
            self.net_g = net_g
            
        def forward(self, x_tst, x_tst_lengths, sid, tones, lang_ids, ja_bert, style_vec, 
                   sdp_ratio, noise_scale, noise_scale_w, length_scale):
            """
            Forward method that matches the SynthesizerTrnJPExtra.infer method signature
            for JP-Extra models
            """
            # Call the infer method directly
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
                length_scale=length_scale
            )
            return output[0]  # Return just the audio
    
    # Create the wrapper instance
    wrapper = NetGWrapper(net_g)
    
    seq_len = 10
    
    # Basic input tensors
    x_tst = torch.zeros((1, seq_len), dtype=torch.long).to(device)
    x_tst_lengths = torch.LongTensor([seq_len]).to(device)
    sid = torch.LongTensor([0]).to(device)
    tones = torch.zeros((1, seq_len), dtype=torch.long).to(device)
    lang_ids = torch.zeros((1, seq_len), dtype=torch.long).to(device)
    
    # BERT embeddings - adjust dimensions based on your model
    bert_dim = 1024  # This might need adjustment
    ja_bert = torch.zeros((1, seq_len, bert_dim), dtype=torch.float32).to(device)
    
    # Style vector - use the actual dimension from the model
    # Based on the error, style_vec seems to expect 256 as the first dimension
    style_vec = torch.zeros((1, 256), dtype=torch.float32).to(device)
    
    # Control parameters
    sdp_ratio = torch.tensor([DEFAULT_SDP_RATIO], dtype=torch.float32).to(device)
    noise_scale = torch.tensor([DEFAULT_NOISE], dtype=torch.float32).to(device)
    noise_scale_w = torch.tensor([DEFAULT_NOISEW], dtype=torch.float32).to(device)
    length_scale = torch.tensor([DEFAULT_LENGTH], dtype=torch.float32).to(device)
    
    # Define input names
    input_names = [
        "x_tst", "x_tst_lengths", "sid", "tones", "lang_ids", "ja_bert", "style_vec",
        "sdp_ratio", "noise_scale", "noise_scale_w", "length_scale"
    ]
    
    # Define output names
    output_names = ["audio"]
    
    # Try to trace the model with the dummy inputs
    try:
        with torch.no_grad():
            dummy_output = wrapper(
                x_tst, x_tst_lengths, sid, tones, lang_ids, ja_bert, style_vec,
                sdp_ratio, noise_scale, noise_scale_w, length_scale
            )
        print(f"Dummy forward pass successful, output shape: {dummy_output.shape}")
    except Exception as e:
        print(f"Error during dummy forward pass: {e}")
        print("Adjusting input shapes...")
        
        # Try to examine the style_proj layer to get correct dimensions
        if hasattr(net_g, "enc_p") and hasattr(net_g.enc_p, "style_proj"):
            style_proj_weight = net_g.enc_p.style_proj.weight
            print(f"Style projection weight shape: {style_proj_weight.shape}")
            
            # Correctly size the style_vec based on actual weight dimensions
            in_features = style_proj_weight.shape[1]
            style_vec = torch.zeros((1, in_features), dtype=torch.float32).to(device)
            print(f"Adjusted style_vec shape to: {style_vec.shape}")
    
    # Export to ONNX
    torch.onnx.export(
        wrapper,
        (x_tst, x_tst_lengths, sid, tones, lang_ids, ja_bert, style_vec,
         sdp_ratio, noise_scale, noise_scale_w, length_scale),
        output_path,
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            "x_tst": {1: "seq_len"},
            "tones": {1: "seq_len"},
            "lang_ids": {1: "seq_len"},
            "ja_bert": {1: "seq_len"},
            "audio": {1: "audio_len", 2: "audio_channels"}
        }
    )
    
    print(f"Model successfully exported to {output_path}")
    return output_path

def main(text, compare_methods=True, num_iterations=3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print(f"Using device: {device}")
    
    model = CustomTTSModel(
        model_path=assets_root / model_file,
        config_path=assets_root / config_file,
        style_vec_path=assets_root / style_file,
        device=device,
    )
    print(model)
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