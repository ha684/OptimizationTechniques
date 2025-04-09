import argparse
import os
import torch
import numpy as np
from pathlib import Path
from huggingface_hub import hf_hub_download
from typing import Optional
from symbols import symbols
from style_bert_vits2.constants import Languages
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.models.models import SynthesizerTrn
from style_bert_vits2.models.models_jp_extra import SynthesizerTrn as SynthesizerTrnJPExtra
from style_bert_vits2.nlp import bert_models
from utils import get_hparams_from_file

def load_model_files(repo_id="litagin/style_bert_vits2_jvnv", model_dir="model_assets"):
    """Download model files from Hugging Face Hub"""
    model_file = "jvnv-F1-jp/jvnv-F1-jp_e160_s14000.safetensors"
    config_file = "jvnv-F1-jp/config.json"
    style_file = "jvnv-F1-jp/style_vectors.npy"

    files = [model_file, config_file, style_file]
    local_paths = {}
    
    for file in files:
        print(f"Downloading {file}...")
        local_path = hf_hub_download(repo_id, file, local_dir=model_dir)
        local_paths[os.path.basename(file)] = local_path
    
    return local_paths

def load_bert_models(language=Languages.JP):
    """Load BERT models for the specified language"""
    if language == Languages.JP:
        bert_models.load_model(language, "ku-nlp/deberta-v2-large-japanese-char-wwm")
        bert_models.load_tokenizer(language, "ku-nlp/deberta-v2-large-japanese-char-wwm")
    else:
        raise ValueError(f"BERT model for language {language} not configured")

def prepare_dummy_inputs(hps, device="cuda:0" if torch.cuda.is_available() else "cpu"):
    """Create dummy inputs for ONNX export based on the actual infer method signature"""
    # Create dummy inputs with appropriate shapes for the model
    seq_len = 100  # Typical sequence length
    
    # Phone sequence
    x = torch.randint(0, len(symbols), (1, seq_len)).to(device)
    x_lengths = torch.LongTensor([seq_len]).to(device)
    
    # Speaker ID
    sid = torch.LongTensor([0]).to(device)  # Use speaker ID 0
    
    # Tones and language IDs
    tone = torch.randint(0, 10, (1, seq_len)).to(device)
    language = torch.ones(1, seq_len).long().to(device)
    
    # BERT embeddings
    bert_dim = hps.bert_embedding_dim if hasattr(hps, 'bert_embedding_dim') else 1024
    bert = torch.randn(1, seq_len, bert_dim).to(device)
    ja_bert = torch.randn(1, seq_len, bert_dim).to(device)
    en_bert = torch.randn(1, seq_len, bert_dim).to(device)
    
    # Style vector
    style_vector_dim = hps.style_vector_dim if hasattr(hps, 'style_vector_dim') else 256
    style_vec = torch.randn(1, style_vector_dim).to(device)
    
    # Other parameters
    noise_scale = torch.tensor([0.667], dtype=torch.float32).to(device)
    length_scale = torch.tensor([1.0], dtype=torch.float32).to(device)
    noise_scale_w = torch.tensor([0.8], dtype=torch.float32).to(device)
    sdp_ratio = torch.tensor([0.2], dtype=torch.float32).to(device)
    
    # Max length (optional, passing as None causes issues in ONNX export)
    max_len = torch.tensor([1000], dtype=torch.int64).to(device)
    
    return {
        "x": x,
        "x_lengths": x_lengths,
        "sid": sid,
        "tone": tone,
        "language": language,
        "bert": bert,
        "ja_bert": ja_bert,
        "en_bert": en_bert,
        "style_vec": style_vec,
        "noise_scale": noise_scale,
        "length_scale": length_scale,
        "noise_scale_w": noise_scale_w,
        "max_len": max_len,
        "sdp_ratio": sdp_ratio
    }

class ONNXExportWrapper(torch.nn.Module):
    """Wrapper module for ONNX export for Style-BERT-VITS2 JP model"""
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x, x_lengths, sid, tone, language, bert, ja_bert, en_bert, style_vec,
                noise_scale, length_scale, noise_scale_w, max_len, sdp_ratio):
        # Forward pass for the model using the correct signature
        y_hat, _, masks, _ = self.model.infer(
            x=x,
            x_lengths=x_lengths,
            sid=sid,
            tone=tone,
            language=language,
            bert=bert,
            ja_bert=ja_bert,
            en_bert=en_bert,
            style_vec=style_vec,
            noise_scale=noise_scale.item(),
            length_scale=length_scale.item(),
            noise_scale_w=noise_scale_w.item(),
            max_len=max_len.item() if max_len is not None else None,
            sdp_ratio=sdp_ratio.item(),
            y=None
        )
        return y_hat

def export_model_to_onnx(
    model_path,
    config_path,
    onnx_save_path="style_bert_vits2_model.onnx",
    device="cuda:0" if torch.cuda.is_available() else "cpu",
    dynamic_axes=True,
    simplify=True
):
    """Export the model to ONNX format"""
    print(f"Loading model configuration from {config_path}")
    if isinstance(config_path, str):
        config_path = Path(config_path)
    
    # Load hyper-parameters
    hps = get_hparams_from_file(config_path)
    # Load model
    print(f"Creating model instance with version: {hps.version}")
    if hps.version.endswith("JP-Extra"):
        net_g = SynthesizerTrnJPExtra(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        )
    else:
        net_g = SynthesizerTrn(
            len(hps.symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        )
    
    # print(f"Loading model weights from {model_path}")
    # if torch.cuda.is_available():
    #     state_dict = torch.load(model_path, map_location=device)
    # else:
    #     state_dict = torch.load(model_path, map_location="cpu")
    
    # net_g.load_state_dict(state_dict["weight"], strict=False)
    net_g.state_dict()
    net_g = net_g.to(device)
    net_g.eval()
    
    # Wrap the model for ONNX export
    wrapped_model = ONNXExportWrapper(net_g)
    
    # Prepare dummy inputs
    dummy_inputs = prepare_dummy_inputs(hps, device)
    
    # Define input and output names
    input_names = [
        "x", "x_lengths", "sid", "tone", "language", "bert", "ja_bert", "en_bert", "style_vec",
        "noise_scale", "length_scale", "noise_scale_w", "max_len", "sdp_ratio"
    ]
    output_names = ["audio"]
    
    # Set up dynamic axes if needed
    dynamic_axes_dict = None
    if dynamic_axes:
        dynamic_axes_dict = {
            "x": {1: "seq_length"},
            "tone": {1: "seq_length"},
            "language": {1: "seq_length"},
            "bert": {1: "seq_length"},
            "ja_bert": {1: "seq_length"},
            "en_bert": {1: "seq_length"},
            "audio": {2: "audio_length"},
        }
    
    # Export the model to ONNX
    print(f"Exporting model to ONNX: {onnx_save_path}")
    try:
        torch.onnx.export(
            wrapped_model,
            tuple(dummy_inputs.values()),
            onnx_save_path,
            export_params=True,
            opset_version=16,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes_dict,
            verbose=False
        )
        print("ONNX export successful!")
    except Exception as e:
        print(f"Error during ONNX export: {e}")
        # Try with a simpler export as fallback
        try:
            print("Attempting fallback export with fewer optimizations...")
            torch.onnx.export(
                wrapped_model,
                tuple(dummy_inputs.values()),
                onnx_save_path,
                export_params=True,
                opset_version=14,
                do_constant_folding=False,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=None,  # Disable dynamic axes
                verbose=True
            )
            print("Fallback ONNX export successful!")
        except Exception as e2:
            print(f"Fallback export also failed: {e2}")
            return None
    
    # Simplify the model if requested
    if simplify:
        try:
            import onnxsim
            import onnx
            print("Simplifying ONNX model...")
            model_onnx = onnx.load(onnx_save_path)
            model_onnx, check = onnxsim.simplify(model_onnx)
            if check:
                onnx.save(model_onnx, onnx_save_path)
                print("Successfully simplified ONNX model")
            else:
                print("Failed to simplify ONNX model")
        except ImportError:
            print("onnxsim not installed. Skipping model simplification.")
            print("To install: pip install onnxsim")
        except Exception as e:
            print(f"Error during model simplification: {e}")
    
    print(f"ONNX model saved to {onnx_save_path}")
    return onnx_save_path

def test_onnx_inference(onnx_model_path, test_text="こんにちは", device="cpu"):
    """Test the exported ONNX model with a sample input"""
    try:
        import onnxruntime as ort
        print(f"\nTesting ONNX model: {onnx_model_path}")
        
        # Create ONNX Runtime session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ['CPUExecutionProvider']
        if device != "cpu" and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')
        
        session = ort.InferenceSession(onnx_model_path, sess_options, providers=providers)
        
        print(f"ONNX Runtime providers: {session.get_providers()}")
        print("ONNX model inputs:")
        for i, input_info in enumerate(session.get_inputs()):
            print(f"  {i}: {input_info.name} - {input_info.shape} - {input_info.type}")
        
        print("ONNX model outputs:")
        for i, output_info in enumerate(session.get_outputs()):
            print(f"  {i}: {output_info.name} - {output_info.shape} - {output_info.type}")
            
        print("\nONNX model test passed! You can now use this model for inference with ONNX Runtime.")
        
    except ImportError:
        print("\nonnxruntime not installed. Skipping model testing.")
        print("To install: pip install onnxruntime")
    except Exception as e:
        print(f"\nError during ONNX model testing: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to model file", default=None)
    parser.add_argument("--config", type=str, help="Path to config file", default=None)
    parser.add_argument("--output", type=str, help="Output ONNX model path", default="style_bert_vits2_model.onnx")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", 
                        help="Device to use for model export")
    parser.add_argument("--no-dynamic", action="store_true", help="Disable dynamic axes for ONNX export")
    parser.add_argument("--no-simplify", action="store_true", help="Disable ONNX model simplification")
    parser.add_argument("--test", action="store_true", help="Test the exported ONNX model")
    args = parser.parse_args()
    
    # Load BERT models first
    print("Loading BERT models...")
    load_bert_models()
    
    # Use downloaded model if not specified
    if args.model is None or args.config is None:
        print("No model or config specified, downloading default model from Hugging Face Hub...")
        model_files = load_model_files()
        
        assets_root = Path("model_assets")
        model_file_path = assets_root / "jvnv-F1-jp/jvnv-F1-jp_e160_s14000.safetensors"
        config_file_path = assets_root / "jvnv-F1-jp/config.json"
    else:
        model_file_path = args.model
        config_file_path = args.config
    
    # Export the model to ONNX
    onnx_path = export_model_to_onnx(
        model_path=model_file_path,
        config_path=config_file_path,
        onnx_save_path=args.output,
        device=args.device,
        dynamic_axes=not args.no_dynamic,
        simplify=not args.no_simplify
    )
    
    # Test the exported model if requested
    if args.test and onnx_path:
        test_onnx_inference(onnx_path, device=args.device)

if __name__ == "__main__":
    main()