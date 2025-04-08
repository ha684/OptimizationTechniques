import torch


def export_onnx(
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