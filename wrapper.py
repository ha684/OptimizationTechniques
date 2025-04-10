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
from Bert_VITS2.onnx_modules.V220_OnnxInference import OnnxInferenceSession

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
    def __init__(self, model_path: Path, config_path: Union[Path, HyperParameters], style_vec_path: Union[Path, NDArray[Any]], device: str) -> None:
        super().__init__(model_path, config_path, style_vec_path, device)
        self.load()
        assert self._TTSModel__net_g is not None, "Model not loaded correctly, net_g is None"
        self.inner_infer = torch.compile(self._TTSModel__net_g)
        self.use_compile = True
        self.compiled_inner_infer = InnerInferModel(self.inner_infer, self.device, self.hyper_parameters)
        self.onnx_session = OnnxInferenceSession(
            {
                "enc": "Bert_VITS2/onnx/BertVits/BertVits_enc_p.onnx",
                "emb_g": "Bert_VITS2/onnx/BertVits/BertVits_emb.onnx",
                "dp": "Bert_VITS2/onnx/BertVits/BertVits_dp.onnx",
                "sdp": "Bert_VITS2/onnx/BertVits/BertVits_sdp.onnx",
                "flow": "Bert_VITS2/onnx/BertVits/BertVits_flow.onnx",
                "dec": "Bert_VITS2/onnx/BertVits/BertVits_dec.onnx",
            },
            Providers=["CPUExecutionProvider"],
        )
    
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
        """The onnx infer implementation"""
        bert, ja_bert, en_bert, phones, tones, lang_ids = self.compiled_inner_infer.preprocess_text(
            text, language, assist_text, assist_text_weight, given_phone, given_tone
        )
        
        # Convert PyTorch tensors to NumPy arrays
        phones_np = phones.cpu().numpy() if isinstance(phones, torch.Tensor) else np.array(phones)
        tones_np = tones.cpu().numpy() if isinstance(tones, torch.Tensor) else np.array(tones)
        lang_ids_np = lang_ids.cpu().numpy() if isinstance(lang_ids, torch.Tensor) else np.array(lang_ids)
        
        seq_len = phones_np.shape[0]
        
        bert_np = bert.cpu().numpy() if isinstance(bert, torch.Tensor) else np.array(bert)
        if bert_np.shape[1] != 1024:
            if bert_np.shape[0] == 1024:
                bert_np = bert_np.T
            else:
                temp_bert = np.zeros((seq_len, 1024), dtype=np.float32)
                min_dim1 = min(bert_np.shape[0], seq_len)
                min_dim2 = min(bert_np.shape[1] if len(bert_np.shape) > 1 else 0, 1024)
                if min_dim2 > 0:
                    temp_bert[:min_dim1, :min_dim2] = bert_np[:min_dim1, :min_dim2]
                bert_np = temp_bert
        
        ja_bert_np = ja_bert.cpu().numpy() if isinstance(ja_bert, torch.Tensor) else np.array(ja_bert)
        if ja_bert_np.shape[1] != 1024:
            temp_ja_bert = np.zeros((seq_len, 1024), dtype=np.float32)
            min_dim1 = min(ja_bert_np.shape[0], seq_len)
            min_dim2 = min(ja_bert_np.shape[1] if len(ja_bert_np.shape) > 1 else 0, 1024)
            if min_dim2 > 0:
                temp_ja_bert[:min_dim1, :min_dim2] = ja_bert_np[:min_dim1, :min_dim2]
            ja_bert_np = temp_ja_bert
            
        en_bert_np = en_bert.cpu().numpy() if isinstance(en_bert, torch.Tensor) else np.array(en_bert)
        if en_bert_np.shape[1] != 1024:
            temp_en_bert = np.zeros((seq_len, 1024), dtype=np.float32)
            min_dim1 = min(en_bert_np.shape[0], seq_len)
            min_dim2 = min(en_bert_np.shape[1] if len(en_bert_np.shape) > 1 else 0, 1024)
            if min_dim2 > 0:
                temp_en_bert[:min_dim1, :min_dim2] = en_bert_np[:min_dim1, :min_dim2]
            en_bert_np = temp_en_bert
            
        sid_np = np.array([speaker_id], dtype=np.int64)
        
        audio = self.onnx_session(
            seq=phones_np,
            tone=tones_np,
            language=lang_ids_np,
            bert_zh=bert_np,  # Chinese BERT
            bert_jp=ja_bert_np,  # Japanese BERT
            bert_en=en_bert_np,  # English BERT
            sid=sid_np,
            seed=114514,  # Default seed
            seq_noise_scale=noise,
            sdp_noise_scale=noisew,
            length_scale=length,
            sdp_ratio=sdp_ratio
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
            results = {"compiled": [], "onnx": []}
            test_texts = text if isinstance(text, list) else [text]
            text_categories = {
                "very_short": test_texts[:3],
                "short": test_texts[3:7],
                "medium": test_texts[7:10],
                "long": test_texts[10:13],
                "very_long": test_texts[13:],
            }
            
            print("\n" + "="*60)
            print(f"PERFORMANCE COMPARISON: Running {num_iterations} iterations for each method")
            print("="*60)
            
            for category, category_texts in text_categories.items():
                print(f"\n--- Testing {category.replace('_', ' ').title()} Texts ---")
                
                for i, text_item in enumerate(category_texts, 1):
                    char_count = len(text_item)
                    print(f"\nText {i} ({char_count} characters): {text_item[:30]}{'...' if len(text_item) > 30 else ''}")
                    
                    # Compile each method's timing results for this text
                    compiled_times_for_text = []
                    onnx_times_for_text = []
                    
                    for iter_num in range(1, num_iterations + 1):
                        print(f"\n  Iteration {iter_num}/{num_iterations}:")
                        
                        # Test compiled method
                        if use_compiled:
                            try:
                                start_time = time.time()
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
                                compiled_times_for_text.append(compiled_time)
                                print(f"    Compiled: {compiled_time:.4f} seconds")
                            except Exception as e:
                                print(f"    Compiled method failed: {str(e)[:100]}")
                        
                        # Test ONNX method
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
                            onnx_times_for_text.append(onnx_time)
                            print(f"    ONNX:     {onnx_time:.4f} seconds")
                        except Exception as e:
                            print(f"    ONNX method failed: {str(e)[:100]}")
                    
                    # Calculate and store results for this text
                    if compiled_times_for_text:
                        avg_compiled = sum(compiled_times_for_text) / len(compiled_times_for_text)
                        results["compiled"].append({
                            "category": category,
                            "text_length": char_count,
                            "text": text_item[:30] + ('...' if len(text_item) > 30 else ''),
                            "avg_time": avg_compiled,
                            "min_time": min(compiled_times_for_text),
                            "max_time": max(compiled_times_for_text)
                        })
                    
                    if onnx_times_for_text:
                        avg_onnx = sum(onnx_times_for_text) / len(onnx_times_for_text)
                        results["onnx"].append({
                            "category": category,
                            "text_length": char_count,
                            "text": text_item[:30] + ('...' if len(text_item) > 30 else ''),
                            "avg_time": avg_onnx,
                            "min_time": min(onnx_times_for_text),
                            "max_time": max(onnx_times_for_text)
                        })
                    
                    # Print summary for this text
                    if compiled_times_for_text and onnx_times_for_text:
                        avg_compiled = sum(compiled_times_for_text) / len(compiled_times_for_text)
                        avg_onnx = sum(onnx_times_for_text) / len(onnx_times_for_text)
                        speedup = avg_compiled / avg_onnx if avg_onnx > 0 else float('inf')
                        
                        print(f"\n  Summary for this text:")
                        print(f"    Compiled avg: {avg_compiled:.4f}s ({min(compiled_times_for_text):.4f}s - {max(compiled_times_for_text):.4f}s)")
                        print(f"    ONNX avg:     {avg_onnx:.4f}s ({min(onnx_times_for_text):.4f}s - {max(onnx_times_for_text):.4f}s)")
                        
                        if speedup > 1:
                            print(f"    Result: ONNX is {speedup:.2f}x faster")
                        else:
                            print(f"    Result: Compiled is {1/speedup:.2f}x faster")
            
            # Generate overall performance report after all tests
            print("\n\n" + "="*80)
            print("PERFORMANCE TEST RESULTS SUMMARY")
            print("="*80)
            
            # Calculate category averages
            category_results = {}
            for category in text_categories.keys():
                category_results[category] = {"compiled": {"times": [], "char_counts": []}, 
                                            "onnx": {"times": [], "char_counts": []}}
                
                for result in results["compiled"]:
                    if result["category"] == category:
                        category_results[category]["compiled"]["times"].append(result["avg_time"])
                        category_results[category]["compiled"]["char_counts"].append(result["text_length"])
                        
                for result in results["onnx"]:
                    if result["category"] == category:
                        category_results[category]["onnx"]["times"].append(result["avg_time"])
                        category_results[category]["onnx"]["char_counts"].append(result["text_length"])
            
            # Print performance table by category
            print("\nPerformance by Text Length Category:")
            print("-" * 80)
            print(f"{'Category':<12} | {'Avg Chars':<10} | {'Compiled Time':<20} | {'ONNX Time':<20} | {'Speedup':<10}")
            print("-" * 80)
            
            for category, data in category_results.items():
                if data["compiled"]["times"] and data["onnx"]["times"]:
                    compiled_avg = sum(data["compiled"]["times"]) / len(data["compiled"]["times"])
                    onnx_avg = sum(data["onnx"]["times"]) / len(data["onnx"]["times"])
                    char_avg = sum(data["compiled"]["char_counts"]) / len(data["compiled"]["char_counts"])
                    
                    speedup = compiled_avg / onnx_avg if onnx_avg > 0 else float('inf')
                    faster = "ONNX" if speedup > 1 else "Compiled"
                    speedup_factor = speedup if speedup > 1 else 1/speedup
                    
                    print(f"{category.replace('_', ' ').title():<12} | "
                        f"{char_avg:<10.1f} | "
                        f"{compiled_avg:.4f}s ({min(data['compiled']['times']):.4f}-{max(data['compiled']['times']):.4f}) | "
                        f"{onnx_avg:.4f}s ({min(data['onnx']['times']):.4f}-{max(data['onnx']['times']):.4f}) | "
                        f"{faster} {speedup_factor:.2f}x")
            
            print("-" * 80)
            
            # Calculate overall statistics
            if results["compiled"] and results["onnx"]:
                overall_compiled_avg = sum(r["avg_time"] for r in results["compiled"]) / len(results["compiled"])
                overall_onnx_avg = sum(r["avg_time"] for r in results["onnx"]) / len(results["onnx"])
                
                # Calculate chars per second
                total_chars_compiled = sum(r["text_length"] for r in results["compiled"])
                total_chars_onnx = sum(r["text_length"] for r in results["onnx"])
                
                total_time_compiled = sum(r["avg_time"] for r in results["compiled"])
                total_time_onnx = sum(r["avg_time"] for r in results["onnx"])
                
                chars_per_sec_compiled = total_chars_compiled / total_time_compiled if total_time_compiled > 0 else 0
                chars_per_sec_onnx = total_chars_onnx / total_time_onnx if total_time_onnx > 0 else 0
                
                overall_speedup = overall_compiled_avg / overall_onnx_avg if overall_onnx_avg > 0 else float('inf')
                
                print("\nOverall Performance:")
                print(f"  Compiled method: {overall_compiled_avg:.4f}s avg ({chars_per_sec_compiled:.1f} chars/sec)")
                print(f"  ONNX method:     {overall_onnx_avg:.4f}s avg ({chars_per_sec_onnx:.1f} chars/sec)")
                
                if overall_speedup > 1:
                    print(f"\nConclusion: ONNX is faster by {overall_speedup:.2f}x overall")
                    return "onnx"
                else:
                    print(f"\nConclusion: Compiled method is faster by {1/overall_speedup:.2f}x overall")
                    return "compiled"
            else:
                print("\nInsufficient data to determine which method is faster")
                return None

def main(compare_methods=True, num_iterations=3): 
    device = "cuda"
    model = CustomTTSModel(
        model_path=assets_root / model_file,
        config_path=assets_root / config_file,
        style_vec_path=assets_root / style_file,
        device=device,
    )

    test_texts = [
        # Very Short
        "こんにちは",
        "Hi",
        "はい",
        
        # Short
        "今日はいい天気ですね。",
        "Good morning!",
        "おはようございます。",
        "This is a test.",
        
        # Medium
        "明日の予定は何ですか？午後に会議があるかもしれません。",
        "Let's meet at the station around 3 PM. Is that okay?",
        "今日はとても忙しかったけど、充実していました。",
        
        # Long
        "私は先週末に京都へ旅行に行きました。古いお寺を見たり、美味しい料理を食べたりして、とても楽しい時間を過ごしました。",
        "I had a great time at the conference last week. The keynote speakers were inspiring, and I learned a lot from the breakout sessions.",
        
        # Very Long
        "昨日の夜は嵐のような雨が降っていて、外に出るのはとても危険でした。そのため、家の中で読書をしたり、音楽を聴いたりして、静かに過ごしました。こういう時間も大切だと思いました。",
        "Modern text-to-speech systems have significantly evolved, offering highly natural-sounding speech. This is especially useful for accessibility, entertainment, and customer service applications where synthetic voices need to sound more human-like and emotionally expressive."
    ]
    
    try:
        print("\n\n" + "="*80)
        print("TTS MODEL INFERENCE PERFORMANCE TEST")
        print("="*80)
        print(f"Device: {device}")
        print(f"Test configuration: {num_iterations} iterations per text sample")
        print(f"Text samples: {len(test_texts)} (across 5 length categories)")
        
        # Run main performance test
        results = {"onnx_wins": 0, "compiled_wins": 0, "errors": 0}
        for run in range(1, 4):
            print(f"\n\nRUN {run}/3")
            try:
                result = model.infer(
                    text=test_texts, 
                    compare_methods=compare_methods,
                    num_iterations=num_iterations
                )
                if result == "onnx":
                    results["onnx_wins"] += 1
                elif result == "compiled":
                    results["compiled_wins"] += 1
                else:
                    results["errors"] += 1
            except Exception as e:
                results["errors"] += 1
                print(f"Error in run {run}: {str(e)[:200]}")
        
        # Final summary table
        print("\n\n" + "="*80)
        print("FINAL TEST RESULTS")
        print("="*80)
        print(f"{'Method':<15} | {'Wins':<10} | {'Win Rate':<10}")
        print("-" * 40)
        total_valid_runs = results["onnx_wins"] + results["compiled_wins"]
        if total_valid_runs > 0:
            onnx_rate = results["onnx_wins"] / total_valid_runs * 100
            compiled_rate = results["compiled_wins"] / total_valid_runs * 100
            print(f"{'ONNX':<15} | {results['onnx_wins']:<10} | {onnx_rate:.1f}%")
            print(f"{'Compiled':<15} | {results['compiled_wins']:<10} | {compiled_rate:.1f}%")
        else:
            print("No valid results collected")
        
        print("-" * 40)
        print(f"{'Error runs':<15} | {results['errors']}")
        print(f"{'Total runs':<15} | {3}")
        print("="*80)
        
        # Final recommendation
        if results["onnx_wins"] > results["compiled_wins"]:
            print("\nRECOMMENDATION: Use ONNX for better performance")
        elif results["compiled_wins"] > results["onnx_wins"]:
            print("\nRECOMMENDATION: Use Compiled method for better performance")
        else:
            print("\nRECOMMENDATION: Both methods perform similarly, consider other factors")
            
    except Exception as e:
        print(f"Error in main test routine: {str(e)}")
        import traceback
        traceback.print_exc()
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true", help="Compare inference method performance")
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations for performance comparison")
    args = parser.parse_args()
    main(
        args.compare, 
        args.iterations
    )