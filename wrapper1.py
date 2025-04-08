#!/usr/bin/env python
"""
Full script to load the TTS model, export it to ONNX, define two inference “engines”:
one using torch.compile (with the onnxrt backend) and one using ONNX Runtime, and
then compare their inference performance.
"""

import time
import argparse
from pathlib import Path
from typing import Any, Optional, Union

import torch
import numpy as np
from numpy.typing import NDArray

import onnxruntime as ort

# Import TTS classes and functions (ensure your PYTHONPATH includes style_bert_vits2)
from style_bert_vits2.tts_model import TTSModel
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.models.models import SynthesizerTrn
from style_bert_vits2.models.models_jp_extra import SynthesizerTrn as SynthesizerTrnJPExtra
from style_bert_vits2.models.infer import get_text, cast, infer
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
from style_bert_vits2.nlp import bert_models
from huggingface_hub import hf_hub_download

# Download asset files if needed.
model_file = "jvnv-F1-jp/jvnv-F1-jp_e160_s14000.safetensors"
config_file = "jvnv-F1-jp/config.json"
style_file = "jvnv-F1-jp/style_vectors.npy"

for file in [model_file, config_file, style_file]:
    print(f"Downloading {file}...")
    hf_hub_download("litagin/style_bert_vits2_jvnv", file, local_dir="model_assets")

assets_root = Path("model_assets")
print("Model files downloaded. Initializing model...")

#######################################
# 1. Wrap the original inference call #
#######################################
class InnerInferModel(torch.nn.Module):
    """
    Wraps the net_g.infer method and provides text pre-processing.
    """
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
        assist_text_weight: float = DEFAULT_ASSIST_TEXT_WEIGHT,
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
            audio = output[0][0, 0]
            return audio

#######################################
# 2. Wrap the net model for ONNX export #
#######################################
class OnnxWrapperModel(torch.nn.Module):
    """
    A wrapper for ONNX export. It takes only the necessary inputs and calls the model's infer method.
    
    Note:
      - For simplicity, this wrapper accepts the dummy tensors as inputs.
    """
    def __init__(self, net_g, device, hps):
        super().__init__()
        self.net_g = net_g
        self.device = device
        self.hps = hps

    def forward(
        self,
        phones: torch.Tensor,      # shape: [seq_len] (LongTensor)
        tones: torch.Tensor,       # shape: [seq_len] (LongTensor)
        lang_ids: torch.Tensor,    # shape: [seq_len] (LongTensor)
        ja_bert: torch.Tensor,     # shape: [seq_len, emb_dim] (FloatTensor)
        style_vec: torch.Tensor,   # shape: [style_dim] (FloatTensor)
        sdp_ratio: torch.Tensor,   # scalar tensor (Float)
        noise_scale: torch.Tensor,  # scalar tensor (Float)
        noise_scale_w: torch.Tensor,# scalar tensor (Float)
        length_scale: torch.Tensor, # scalar tensor (Float)
        sid: torch.Tensor,         # scalar tensor (Long)
    ):
        x_tst = phones.unsqueeze(0)
        tones = tones.unsqueeze(0)
        lang_ids = lang_ids.unsqueeze(0)
        ja_bert = ja_bert.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(x_tst.device)
        style_vec_tensor = style_vec.unsqueeze(0)
        sid_tensor = sid.unsqueeze(0)

        output = cast(SynthesizerTrnJPExtra, self.net_g).infer(
            x_tst,
            x_tst_lengths,
            sid_tensor,
            tones,
            lang_ids,
            ja_bert,
            style_vec=style_vec_tensor,
            sdp_ratio=sdp_ratio.item() if isinstance(sdp_ratio, torch.Tensor) else float(sdp_ratio),
            noise_scale=noise_scale.item() if isinstance(noise_scale, torch.Tensor) else float(noise_scale),
            noise_scale_w=noise_scale_w.item() if isinstance(noise_scale_w, torch.Tensor) else float(noise_scale_w),
            length_scale=length_scale.item() if isinstance(length_scale, torch.Tensor) else float(length_scale),
        )
        return output[0][0, 0]

#######################################
# 3. Define an ONNX runtime inference model class #
#######################################
class OnnxInferModel:
    """
    Loads the exported ONNX model and provides a forward-like function.
    """
    def __init__(self, onnx_model_path: Union[str, Path]):
        self.session = ort.InferenceSession(str(onnx_model_path))

    def onnx_forward(
        self,
        phones: np.ndarray,
        tones: np.ndarray,
        lang_ids: np.ndarray,
        ja_bert: np.ndarray,
        style_vec: np.ndarray,
        sdp_ratio: np.ndarray,   # scalar in np.array
        noise_scale: np.ndarray, # scalar in np.array
        noise_scale_w: np.ndarray, # scalar in np.array
        length_scale: np.ndarray, # scalar in np.array
        sid: np.ndarray,         # scalar in np.array
    ) -> np.ndarray:
        inputs = {
            "phones": phones,
            "tones": tones,
            "lang_ids": lang_ids,
            "ja_bert": ja_bert,
            "style_vec": style_vec,
            "sdp_ratio": sdp_ratio,
            "noise_scale": noise_scale,
            "noise_scale_w": noise_scale_w,
            "length_scale": length_scale,
            "sid": sid,
        }
        outputs = self.session.run(["audio"], inputs)
        return outputs[0]

#######################################
# 4. Update the CustomTTSModel to include ONNX export and inference
#######################################
class CustomTTSModel(TTSModel):
    def __init__(self, model_path: Path, config_path: Union[Path, HyperParameters], style_vec_path: Union[Path, NDArray[Any]], device: str) -> None:
        super().__init__(model_path, config_path, style_vec_path, device)
        self.load()
        assert self._TTSModel__net_g is not None, "Model not loaded correctly, net_g is None"
        # Create a compiled version with torch.compile using the onnxrt backend.
        self.inner_infer = torch.compile(self._TTSModel__net_g, fullgraph=True, backend="onnxrt")
        self.use_compile = True
        self.compiled_inner_infer = InnerInferModel(self.inner_infer, self.device, self.hyper_parameters)
        # Export the uncompiled net_g to ONNX and instantiate an ONNX runtime inference object.
        self.onnx_model_path = assets_root / "model.onnx"
        self.export_to_onnx()
        self.onnx_infer = OnnxInferModel(self.onnx_model_path)

    def export_to_onnx(self):
        """
        Exports the current net_g to an ONNX model using dummy inputs.
        Adjust the dummy input shapes to match the expected model input dimensions.
        """
        print("Exporting model to ONNX...")
        self._TTSModel__net_g.eval()
        wrapper = OnnxWrapperModel(self._TTSModel__net_g, self.device, self.hyper_parameters).to(self.device)
        # Dummy inputs: adjust dimensions as required by your model.
        seq_len = 50   # Dummy sequence length.
        emb_dim = 1024  # Updated embedding dimension for ja_bert (changed from 768).
        style_dim = 256  # Dummy style vector dimension (adjust if needed).

        dummy_phones = torch.randint(low=0, high=100, size=(seq_len,), dtype=torch.long, device=self.device)
        dummy_tones = torch.randint(low=0, high=10, size=(seq_len,), dtype=torch.long, device=self.device)
        dummy_lang_ids = torch.randint(low=0, high=2, size=(seq_len,), dtype=torch.long, device=self.device)
        dummy_ja_bert = torch.randn(seq_len, emb_dim, device=self.device)
        dummy_style_vec = torch.randn(style_dim, device=self.device)
        dummy_sdp_ratio = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        dummy_noise_scale = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        dummy_noise_scale_w = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        dummy_length_scale = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        dummy_sid = torch.tensor(0, dtype=torch.long, device=self.device)

        torch.onnx.export(
            wrapper,
            (dummy_phones, dummy_tones, dummy_lang_ids, dummy_ja_bert, dummy_style_vec,
             dummy_sdp_ratio, dummy_noise_scale, dummy_noise_scale_w, dummy_length_scale, dummy_sid),
            str(self.onnx_model_path),
            input_names=['phones','tones','lang_ids','ja_bert','style_vec','sdp_ratio','noise_scale','noise_scale_w','length_scale','sid'],
            output_names=['audio'],
            dynamic_axes={
                'phones': {0: 'seq_len'},
                'tones': {0: 'seq_len'},
                'lang_ids': {0: 'seq_len'},
                'ja_bert': {0: 'seq_len'},
            },
            opset_version=16,  # Use an opset version as needed.
        )
        print(f"ONNX model exported to {self.onnx_model_path}")

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
        bert, ja_bert, en_bert, phones, tones, lang_ids = self.compiled_inner_infer.preprocess_text(
            text, language, assist_text, assist_text_weight, given_phone, given_tone
        )
        audio = self.compiled_inner_infer(
            bert, ja_bert, en_bert, phones, tones, lang_ids,
            style_vector, sdp_ratio, noise, noisew, 
            length, speaker_id
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
        # Preprocess text similar to the compiled method.
        bert, ja_bert, en_bert, phones, tones, lang_ids = self.compiled_inner_infer.preprocess_text(
            text, language, assist_text, assist_text_weight, given_phone, given_tone
        )
        phones_np = phones.cpu().numpy()
        tones_np = tones.cpu().numpy()
        lang_ids_np = lang_ids.cpu().numpy()
        ja_bert_np = ja_bert.cpu().numpy()
        style_vec_np = style_vector.astype(np.float32)
        # Scalars as numpy arrays.
        sdp_ratio_np = np.array(sdp_ratio, dtype=np.float32)
        noise_scale_np = np.array(noise, dtype=np.float32)
        noise_scale_w_np = np.array(noisew, dtype=np.float32)
        length_scale_np = np.array(length, dtype=np.float32)
        sid_np = np.array(speaker_id, dtype=np.int64)
        audio = self.onnx_infer.onnx_forward(phones_np, tones_np, lang_ids_np, ja_bert_np,
                                             style_vec_np, sdp_ratio_np, noise_scale_np,
                                             noise_scale_w_np, length_scale_np, sid_np)
        return audio

    def infer(
        self,
        text: Union[str, list],
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
    ) -> bool:
        # Prepare style vector (from style id or reference audio).
        if language != "JP" and self.hyper_parameters.version.endswith("JP-Extra"):
            raise ValueError("The model is trained with JP-Extra, but the language is not JP")
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

        print("\n--- Starting performance comparison ---")
        print(f"Running {num_iterations} iterations for each method...")
        test_texts = text if isinstance(text, list) else [text]
        compiled_times = []
        onnx_times = []
        for text_item in test_texts:
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
                print(f"Compiled method: {compiled_time:.4f} seconds")
            except Exception as e:
                print(f"Compiled method failed: {e}")

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
                print(f"ONNX method: {onnx_time:.4f} seconds")
            except Exception as e:
                print(f"ONNX method failed: {e}")

        print("\n--- Performance Summary ---")
        if compiled_times:
            avg_compiled = sum(compiled_times) / len(compiled_times)
            print(f"Compiled method average time: {avg_compiled:.4f} seconds")
        if onnx_times:
            avg_onnx = sum(onnx_times) / len(onnx_times)
            print(f"ONNX method average time: {avg_onnx:.4f} seconds")
        print("--------------------------------------")
        return True

#######################################
# 5. Main function and argument parsing #
#######################################
def main(text, compare_methods=True, num_iterations=3):
    device = "cpu"  # or "cuda" if available and configured
    model = CustomTTSModel(
        model_path=assets_root / model_file,
        config_path=assets_root / config_file,
        style_vec_path=assets_root / style_file,
        device=device,
    )
    if not compare_methods:
        return
    else:
        print("Model initialized. Running inference performance comparison...")
        test_texts = [
            text,
            "元気ですか？",
            "hello how are you"
        ]
        result = model.infer(
            text=test_texts, 
            compare_methods=compare_methods,
            num_iterations=num_iterations
        )
        if result:
            print("Inference runs completed successfully")
        else:
            print("Inference runs failed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="Text to be converted to speech", default="こんにちは、元気ですか？")
    parser.add_argument("--compare", action="store_true", help="Compare inference method performance")
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations for performance comparison")
    args = parser.parse_args()
    main(
        args.text, 
        args.compare, 
        args.iterations
    )
