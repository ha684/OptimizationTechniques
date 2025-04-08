from utils import get_hparams_from_file, load_checkpoint
import json
import os
from symbols import symbols
from style_bert_vits2.models.models import SynthesizerTrn
from typing import Any
import torch
import commons
from pathlib import Path
class CustomSynthesizerTrn(SynthesizerTrn):
    def __init__(
        self, 
        n_vocab: int, 
        spec_channels: int, 
        segment_size: int, 
        inter_channels: int, 
        hidden_channels: int, 
        filter_channels: int, 
        n_heads: int, 
        n_layers: int, 
        kernel_size: int, 
        p_dropout: float, 
        resblock: str, 
        resblock_kernel_sizes: list[int], 
        resblock_dilation_sizes: list[list[int]], 
        upsample_rates: list[int], 
        upsample_initial_channel: int, 
        upsample_kernel_sizes: list[int], 
        n_speakers: int = 256, 
        gin_channels: int = 256, 
        use_sdp: bool = True, 
        n_flow_layer: int = 4, 
        n_layers_trans_flow: int = 4, 
        flow_share_parameter: bool = False, 
        use_transformer_flow: bool = True, 
        **kwargs: Any
    ) -> None:
        super().__init__(
            n_vocab, 
            spec_channels, 
            segment_size, 
            inter_channels, 
            hidden_channels, 
            filter_channels, 
            n_heads, 
            n_layers, 
            kernel_size, 
            p_dropout, 
            resblock, 
            resblock_kernel_sizes, 
            resblock_dilation_sizes, 
            upsample_rates, 
            upsample_initial_channel, 
            upsample_kernel_sizes, 
            n_speakers, 
            gin_channels, 
            use_sdp, 
            n_flow_layer, 
            n_layers_trans_flow, 
            flow_share_parameter, 
            use_transformer_flow, 
            **kwargs
        )
        
    def export_onnx(
        self,
        path,
        max_len=None,
        sdp_ratio=0,
        y=None,
    ):
        noise_scale = 0.667
        length_scale = 1
        noise_scale_w = 0.8
        x = (
            torch.LongTensor(
                [
                    0,
                    97,
                    0,
                    8,
                    0,
                    78,
                    0,
                    8,
                    0,
                    76,
                    0,
                    37,
                    0,
                    40,
                    0,
                    97,
                    0,
                    8,
                    0,
                    23,
                    0,
                    8,
                    0,
                    74,
                    0,
                    26,
                    0,
                    104,
                    0,
                ]
            )
            .unsqueeze(0)
            .cpu()
        )
        tone = torch.zeros_like(x).cpu()
        language = torch.zeros_like(x).cpu()
        x_lengths = torch.LongTensor([x.shape[1]]).cpu()
        sid = torch.LongTensor([0]).cpu()
        bert = torch.randn(size=(x.shape[1], 1024)).cpu()
        ja_bert = torch.randn(size=(x.shape[1], 1024)).cpu()
        en_bert = torch.randn(size=(x.shape[1], 1024)).cpu()

        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
            torch.onnx.export(
                self.emb_g,
                (sid),
                f"onnx/{path}/{path}_emb.onnx",
                input_names=["sid"],
                output_names=["g"],
                verbose=True,
            )
        else:
            g = self.ref_enc(y.transpose(1, 2)).unsqueeze(-1)

        torch.onnx.export(
            self.enc_p,
            (x, x_lengths, tone, language, bert, ja_bert, en_bert, g),
            f"onnx/{path}/{path}_enc_p.onnx",
            input_names=[
                "x",
                "x_lengths",
                "t",
                "language",
                "bert_0",
                "bert_1",
                "bert_2",
                "g",
            ],
            output_names=["xout", "m_p", "logs_p", "x_mask"],
            dynamic_axes={
                "x": [0, 1],
                "t": [0, 1],
                "language": [0, 1],
                "bert_0": [0],
                "bert_1": [0],
                "bert_2": [0],
                "xout": [0, 2],
                "m_p": [0, 2],
                "logs_p": [0, 2],
                "x_mask": [0, 2],
            },
            verbose=True,
            opset_version=16,
        )
        x, m_p, logs_p, x_mask = self.enc_p(
            x, x_lengths, tone, language, bert, ja_bert, en_bert, g=g
        )
        zinput = (
            torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype)
            * noise_scale_w
        )
        torch.onnx.export(
            self.sdp,
            (x, x_mask, zinput, g),
            f"onnx/{path}/{path}_sdp.onnx",
            input_names=["x", "x_mask", "zin", "g"],
            output_names=["logw"],
            dynamic_axes={"x": [0, 2], "x_mask": [0, 2], "zin": [0, 2], "logw": [0, 2]},
            verbose=True,
        )
        torch.onnx.export(
            self.dp,
            (x, x_mask, g),
            f"onnx/{path}/{path}_dp.onnx",
            input_names=["x", "x_mask", "g"],
            output_names=["logw"],
            dynamic_axes={"x": [0, 2], "x_mask": [0, 2], "logw": [0, 2]},
            verbose=True,
        )
        logw = self.sdp(x, x_mask, zinput, g=g) * (sdp_ratio) + self.dp(
            x, x_mask, g=g
        ) * (1 - sdp_ratio)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(
            x_mask.dtype
        )
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        torch.onnx.export(
            self.flow,
            (z_p, y_mask, g),
            f"onnx/{path}/{path}_flow.onnx",
            input_names=["z_p", "y_mask", "g"],
            output_names=["z"],
            dynamic_axes={"z_p": [0, 2], "y_mask": [0, 2], "z": [0, 2]},
            verbose=True,
        )

        z = self.flow(z_p, y_mask, g=g, reverse=True)
        z_in = (z * y_mask)[:, :, :max_len]

        torch.onnx.export(
            self.dec,
            (z_in, g),
            f"onnx/{path}/{path}_dec.onnx",
            input_names=["z_in", "g"],
            output_names=["o"],
            dynamic_axes={"z_in": [0, 2], "o": [0, 2]},
            verbose=True,
        )
        o = self.dec((z * y_mask)[:, :, :max_len], g=g)


def export_onnx(export_path, model_path, config_path, novq, dev, Extra):
    hps = get_hparams_from_file(config_path)
    version = hps.version[0:3]
    enable_emo = False
    net_g = CustomSynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    )
    _ = net_g.eval()
    _ = load_checkpoint(model_path, net_g, None, skip_optimizer=True)
    net_g.cpu()
    net_g.export_onnx(export_path)

    spklist = []
    for key in hps.data.spk2id.keys():
        spklist.append(key)

    LangDict = {"ZH": [0, 0], "JP": [1, 6], "EN": [2, 8]}
    BertSize = 1024
    if version == "2.4":
        BertPaths = (
            ["Erlangshen-MegatronBert-1.3B-Chinese"]
            if Extra == "chinese"
            else ["deberta-v2-large-japanese-char-wwm"]
        )
        if Extra == "chinese":
            BertSize = 2048

    MoeVSConf = {
        "Folder": f"{export_path}",
        "Name": f"{export_path}",
        "Type": "BertVits",
        "Symbol": symbols,
        "Cleaner": "",
        "Rate": hps.data.sampling_rate,
        "CharaMix": True,
        "Characters": spklist,
        "LanguageMap": LangDict,
        "Dict": "BasicDict",
        "BertPath": BertPaths,
        "Clap": ("clap-htsat-fused" if enable_emo else False),
        "BertSize": BertSize,
    }

    with open(f"onnx/{export_path}.json", "w") as MoeVsConfFile:
        json.dump(MoeVSConf, MoeVsConfFile, indent=4)

if __name__ == "__main__":
    model_file = "jvnv-F1-jp/jvnv-F1-jp_e160_s14000.safetensors"
    config_file = "jvnv-F1-jp/config.json"
    style_file = "jvnv-F1-jp/style_vectors.npy"
    assets_root = Path("model_assets")
    export_path = assets_root / "BertVits"
    novq = False
    dev = False
    Extra = "japanese" 
    if not os.path.exists("onnx"):
        os.makedirs("onnx")
    if not os.path.exists(f"onnx/{export_path}"):
        os.makedirs(f"onnx/{export_path}")
    export_onnx(export_path, assets_root / model_file, assets_root / config_file, novq, dev, Extra)
