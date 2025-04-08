import torch
import onnxruntime as ort
import numpy as np
import time
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any, Union

class OnnxTTSModelWrapper:
    def __init__(self, parent_model, max_sequence_length=128):
        self.parent_model = parent_model
        self.max_sequence_length = max_sequence_length
        self.onnx_model_path = None
        self.onnx_session = None
        
    def export_model(self, output_path: str):
        """Export the TTS model to ONNX format"""
        print(f"Creating ONNX export wrapper for max sequence length: {self.max_sequence_length}")
        
        # Create a wrapper class for ONNX export
        class OnnxExportWrapper(torch.nn.Module):
            def __init__(self, net_g, hps, max_length):
                super().__init__()
                self.net_g = net_g
                self.hps = hps
                self.max_length = max_length
                
            def forward(self, phones, phone_lengths, sid, tones, lang_ids, ja_bert, style_vec, 
                      sdp_ratio, noise_scale, noise_scale_w, length_scale):
                return self.net_g.infer(
                    phones, phone_lengths, sid, tones, lang_ids, ja_bert,
                    style_vec=style_vec, sdp_ratio=sdp_ratio, 
                    noise_scale=noise_scale, noise_scale_w=noise_scale_w,
                    length_scale=length_scale
                )
        
        # Create export wrapper
        net_g = self.parent_model._TTSModel__net_g
        hps = self.parent_model.hyper_parameters
        export_wrapper = OnnxExportWrapper(net_g, hps, self.max_sequence_length)
        
        # Create dummy inputs for ONNX export
        batch_size = 1
        max_len = self.max_sequence_length
        
        dummy_phones = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.parent_model.device)
        dummy_phone_lengths = torch.LongTensor([max_len]).to(self.parent_model.device)
        dummy_sid = torch.LongTensor([0]).to(self.parent_model.device)
        dummy_tones = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.parent_model.device)
        dummy_lang_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.parent_model.device)
        dummy_ja_bert = torch.zeros((batch_size, max_len, 1024), dtype=torch.float, device=self.parent_model.device)  # Adjust dimension as needed
        dummy_style_vec = torch.zeros((batch_size, 1024), dtype=torch.float, device=self.parent_model.device)  # Adjust dimension as needed
        
        # Scalar inputs
        dummy_sdp_ratio = torch.tensor([0.2], dtype=torch.float, device=self.parent_model.device)
        dummy_noise_scale = torch.tensor([0.6], dtype=torch.float, device=self.parent_model.device)
        dummy_noise_scale_w = torch.tensor([0.8], dtype=torch.float, device=self.parent_model.device)
        dummy_length_scale = torch.tensor([1.0], dtype=torch.float, device=self.parent_model.device)
        
        # Export model
        print(f"Exporting model to {output_path}...")
        with torch.no_grad():
            torch.onnx.export(
                export_wrapper,
                (dummy_phones, dummy_phone_lengths, dummy_sid, dummy_tones, dummy_lang_ids, 
                 dummy_ja_bert, dummy_style_vec, dummy_sdp_ratio, dummy_noise_scale, 
                 dummy_noise_scale_w, dummy_length_scale),
                output_path,
                input_names=['phones', 'phone_lengths', 'sid', 'tones', 'lang_ids', 'ja_bert', 
                           'style_vec', 'sdp_ratio', 'noise_scale', 'noise_scale_w', 'length_scale'],
                output_names=['audio'],
                dynamic_axes={
                    'phones': {1: 'seq_len'},
                    'tones': {1: 'seq_len'},
                    'lang_ids': {1: 'seq_len'},
                    'ja_bert': {1: 'seq_len'},
                    'audio': {1: 'audio_len'}
                },
                opset_version=17,
                verbose=True
            )
        
        self.onnx_model_path = output_path
        print(f"Model exported to {output_path}")
        
    def load_onnx_model(self):
        """Load the ONNX model for inference"""
        if self.onnx_model_path is None:
            raise ValueError("ONNX model not exported yet. Call export_model first.")
            
        print(f"Loading ONNX model from {self.onnx_model_path}")
        providers = ['CPUExecutionProvider']
        if torch.cuda.is_available():
            providers.insert(0, 'CUDAExecutionProvider')
            
        self.onnx_session = ort.InferenceSession(
            self.onnx_model_path, 
            providers=providers
        )
        print("ONNX model loaded successfully")
        
    def infer(self, text, language, style_vector, sdp_ratio, noise, noise_w, length, speaker_id=0, 
              assist_text=None, assist_text_weight=0.7, given_phone=None, given_tone=None):
        """Run inference using the ONNX model"""
        if self.onnx_session is None:
            self.load_onnx_model()
            
        # Reuse the text preprocessing from the parent model
        bert, ja_bert, en_bert, phones, tones, lang_ids = self.parent_model.compiled_inner_infer.preprocess_text(
            text, language, assist_text, assist_text_weight, given_phone, given_tone
        )
        
        # Pad to fixed length if needed
        seq_len = phones.size(0)
        if seq_len > self.max_sequence_length:
            print(f"Warning: Input sequence length {seq_len} exceeds max sequence length {self.max_sequence_length}")
            # Truncate to max sequence length
            phones = phones[:self.max_sequence_length]
            tones = tones[:self.max_sequence_length]
            lang_ids = lang_ids[:self.max_sequence_length]
            ja_bert = ja_bert[:, :self.max_sequence_length]
            seq_len = self.max_sequence_length
        
        # Pad if sequence is shorter than max length
        if seq_len < self.max_sequence_length:
            pad_len = self.max_sequence_length - seq_len
            phones = torch.nn.functional.pad(phones, (0, pad_len), value=0)
            tones = torch.nn.functional.pad(tones, (0, pad_len), value=0)
            lang_ids = torch.nn.functional.pad(lang_ids, (0, pad_len), value=0)
            ja_bert = torch.nn.functional.pad(ja_bert, (0, 0, 0, pad_len), value=0)
            
        # Convert inputs to numpy arrays for ONNX
        phones_np = phones.unsqueeze(0).cpu().numpy()
        phone_lengths_np = np.array([seq_len], dtype=np.int64)
        sid_np = np.array([speaker_id], dtype=np.int64)
        tones_np = tones.unsqueeze(0).cpu().numpy()
        lang_ids_np = lang_ids.unsqueeze(0).cpu().numpy()
        ja_bert_np = ja_bert.unsqueeze(0).cpu().numpy()
        style_vector_np = style_vector.reshape(1, -1).astype(np.float32)
        
        # Scalar inputs
        sdp_ratio_np = np.array([sdp_ratio], dtype=np.float32)
        noise_np = np.array([noise], dtype=np.float32)
        noise_w_np = np.array([noise_w], dtype=np.float32)
        length_np = np.array([length], dtype=np.float32)
        
        # Run ONNX inference
        ort_inputs = {
            'phones': phones_np,
            'phone_lengths': phone_lengths_np,
            'sid': sid_np,
            'tones': tones_np,
            'lang_ids': lang_ids_np,
            'ja_bert': ja_bert_np,
            'style_vec': style_vector_np,
            'sdp_ratio': sdp_ratio_np,
            'noise_scale': noise_np,
            'noise_scale_w': noise_w_np,
            'length_scale': length_np
        }
        
        ort_outputs = self.onnx_session.run(None, ort_inputs)
        audio = ort_outputs[0][0, 0]
        
        return audio