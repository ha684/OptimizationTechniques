# Đây là project áp dụng torch.compile và ONNX để tối ưu hiệu suất của mô hình TTS.

> Sẽ có 2 phần 1 là về torch.compile, 2 là về ONNX và cuối cùng là so sánh hiệu năng của cả 2.

# Tổng quan về `torch.compile`

## Tham số của hàm `torch.compile`

```python
torch.compile(
    model: Callable[[_InputT], _RetT],
    *,
    fullgraph: bool = False,
    dynamic: Optional[bool] = None,
    backend: Union[str, Callable] = 'inductor',
    mode: Optional[str] = None,
    options: Optional[Dict[str, Union[str, int, bool]]] = None,
    disable: bool = False
) → Callable[[_InputT], _RetT]
```

### Giải thích chi tiết các tham số:

- **`model` (Callable)**  
  Hàm hoặc module mà bạn muốn tối ưu hóa bằng cách biên dịch.

- **`fullgraph` (bool)**  
  - Mặc định là `False`: PyTorch sẽ cố gắng tìm các phần trong model có thể được biên dịch.  
  - Nếu đặt là `True`: yêu cầu toàn bộ hàm phải được biểu diễn thành một biểu đồ duy nhất. Nếu không thể (có "graph breaks"), sẽ báo lỗi.

- **`dynamic` (bool hoặc None)**  
  - `True`: sử dụng dynamic shape tracing để tránh phải biên dịch lại khi kích thước đầu vào thay đổi.  
  - `False`: luôn luôn chuyên biệt hóa (specialize), không sinh kernel động.  
  - `None` (mặc định): tự động phát hiện thay đổi shape và biên dịch lại khi cần thiết.

- **`backend` (str hoặc Callable)**  
  - Backend để thực hiện biên dịch. Mặc định là `"inductor"` – cân bằng tốt giữa hiệu năng và chi phí.  
  - Dùng `torch._dynamo.list_backends()` để xem các backend không thử nghiệm.  
  - Dùng `torch._dynamo.list_backends(None)` để xem cả backend thử nghiệm/debug.  
  - Có thể đăng ký backend tuỳ chỉnh theo hướng dẫn tại:  
    https://pytorch.org/docs/main/torch.compiler_custom_backends.html#registering-custom-backends

- **`mode` (str)**  
  Một số chế độ có thể sử dụng:
  - `"default"`: chế độ mặc định, cân bằng giữa hiệu năng và overhead.
  - `"reduce-overhead"`: giảm chi phí vận hành Python, hữu ích với batch nhỏ, sử dụng CUDA Graphs.
  - `"max-autotune"`: kích hoạt tối đa khả năng autotuning (dùng Triton, CUDA graphs nếu có).
  - `"max-autotune-no-cudagraphs"`: như trên nhưng không sử dụng CUDA graphs.

  Có thể dùng `torch._inductor.list_mode_options()` để xem cụ thể các cấu hình tương ứng với từng mode.

- **`options` (dict)**  
  Tuỳ chọn nâng cao cho backend. Một vài option đáng chú ý:
  - `epilogue_fusion`: gộp các phép toán pointwise, cần kết hợp với `max_autotune`.
  - `max_autotune`: cho phép profiling để chọn cấu hình tốt nhất cho các phép nhân ma trận.
  - `fallback_random`: hỗ trợ debug các vấn đề về độ chính xác.
  - `shape_padding`: pad thêm kích thước tensor để tối ưu truy xuất bộ nhớ GPU.
  - `triton.cudagraphs`: giảm overhead của Python khi dùng CUDA.
  - `trace.enabled`: flag hỗ trợ debug hiệu quả.
  - `trace.graph_diagram`: hiển thị sơ đồ biểu đồ sau khi thực hiện fusion.

  Để xem danh sách đầy đủ các option hỗ trợ, dùng `torch._inductor.list_options()`.

- **`disable` (bool)**  
  Nếu đặt `True`, `torch.compile()` sẽ không làm gì cả — phù hợp khi cần tắt compile tạm thời để test.

## Ví dụ sử dụng

```python
@torch.compile(options={"triton.cudagraphs": True}, fullgraph=True)
def foo(x):
    return torch.sin(x) + torch.cos(x)
```

## Một vài lưu ý sau khi thử nghiệm thực tế

Theo như testing em đã làm thì việc sử dụng `torch.compile` cần được cân nhắc kỹ, không thể "compile" một model một cách máy móc.  

Cụ thể:
- `torch.compile` chỉ hoạt động hiệu quả với phần **tính toán thuần túy**.
- Những phần liên quan đến I/O (ví dụ: đọc file, ghi log, print, hoặc các thao tác không thuần tensor) cần được tách **ra khỏi hàm `forward`** hoặc không nên nằm trong phạm vi biên dịch.
- Khi sử dụng sai cách, PyTorch sẽ không thể dựng được biểu đồ hoặc hiệu năng sẽ không như mong đợi.

## Liệu sử dụng `torch.compile` có guarantee về việc improve performance khi sử dụng?

Câu trả lời là **có** và **không**.

Cụ thể thì với **single batch inference**, có những trường hợp sẽ chạy nhanh hơn, nhưng cũng có những trường hợp hiệu năng sẽ kém hơn. Tuy nhiên, với **multi-batch inference** (inference nhiều lần với cùng một model), theo kết quả thử nghiệm của em thì **ít nhiều sẽ có cải thiện hiệu năng sau khi compile**. Điều này có thể lý giải là do cơ chế caching và tối ưu hoá kernel chỉ phát huy hiệu quả sau một vài lần chạy đầu tiên.

## Tại sao với mô hình TTS chỉ có thể compile vào `g_net` thay vì compile như cách em đã làm?

Sau khi tìm hiểu lại thì em nhận ra mình đã sử dụng `torch.compile` chưa đúng cách. Dù về mặt kỹ thuật thì việc compile toàn bộ mô hình vẫn chạy được, nhưng hiệu năng thực tế lại **không ổn định**, và **đa số trường hợp còn chậm hơn cả khi không dùng compile**.

Lý do chính là:
- Trong mô hình TTS, không phải toàn bộ pipeline đều mang tính "tính toán thuần tensor" (pure tensor computation). Có rất nhiều thao tác liên quan đến I/O (như xử lý văn bản, thao tác với waveform, tiền xử lý hoặc hậu xử lý), những phần này **không phù hợp để compile**, hoặc **làm gián đoạn graph**, dẫn đến mất tối ưu.
- Khi compile toàn bộ pipeline, PyTorch sẽ khó tối ưu do có quá nhiều phần không thể biểu diễn thành graph, dẫn đến việc tạo ra nhiều **"graph break"** và **recompilation không cần thiết**.
- Trong khi đó, `g_net` là phần model tính toán chính, thường là một mạng neural network thuần tuý. Việc compile riêng phần này sẽ:
  - Giảm chi phí tracing lại graph.
  - Tận dụng tốt nhất khả năng tối ưu kernel.
  - Không bị ảnh hưởng bởi các thao tác I/O khác.

=> Vì vậy, **việc compile chỉ riêng `g_net` là cách tối ưu nhất**, đảm bảo tận dụng được hiệu năng của `torch.compile` mà không ảnh hưởng đến phần còn lại của pipeline.

## Sự khác biệt giữa **mode** và **backend** trong `torch.compile()`

Hai tham số `mode` và `backend` đều ảnh hưởng đến cách PyTorch tối ưu mô hình khi sử dụng `torch.compile`, nhưng vai trò và mức độ kiểm soát của chúng là **khác nhau**.

### 1. `backend` – *Tham số này là kiểu lựa chọn bộ máy để biên dịch code*

- Đây là thành phần chính xử lý việc **chuyển đổi model sang dạng tối ưu hóa** và thực thi chúng.
- Hiểu đơn giản thì `backend` giống như **"công cụ" hoặc "engine"** đứng sau việc biên dịch và thực thi mô hình.
- Mặc định là `"inductor"` – một backend do PyTorch phát triển, tối ưu tốt cho GPU/CPU.
- Có thể thay bằng các backend khác (ví dụ: `"onnxrt"` hoặc backend tùy chỉnh).

Ví dụ:
```python
torch.compile(model, backend='inductor')
```

> Với thử nghiệm của em thì việc thay đổi giữa các backend em không thực sự nhận thấy sự khác nhau lắm (có thể là vì chỉ thực hiện infer trên 7 samples)

### 2. `mode` – *Hướng tối ưu*

- `mode` xác định **chiến lược tối ưu hóa** được áp dụng trong quá trình compile.
- Đây giống như **"preset cấu hình"** cho backend – giúp bạn chọn giữa việc ưu tiên tốc độ, bộ nhớ, hay autotuning.
- Mỗi mode sẽ set các flag nội bộ khác nhau trong backend (thường là inductor).

Các mode phổ biến:
- `"default"`: cân bằng hiệu năng và độ ổn định, phù hợp cho hầu hết trường hợp.
- `"reduce-overhead"`: giảm chi phí thực thi Python, thích hợp cho batch nhỏ (dùng CUDA graphs).
- `"max-autotune"`: bật autotune tối đa, tìm cấu hình tốt nhất cho các phép toán như matmul/convolution.
- `"max-autotune-no-cudagraphs"`: như trên, nhưng không dùng CUDA graphs (phù hợp với một số constraint đặc biệt).

Ví dụ:
```python
torch.compile(model, mode='max-autotune')
```

> Tương tự như `backend`, em không thấy có sự khác nhau rõ ràng

## Về việc thử nghiệm compile xong hiệu năng còn kém hơn

Việc này "điêu" thật, chính vì em đã có giải thích ở trên, đó là em làm sai

## bert-vits chuyển từ `.pt` sang `.safetensors`, lý do là gì?

Theo như research của em thì lý do chính nằm ở **tính an toàn và tốc độ tải mô hình**.

### 1. `.safetensors` giúp **tránh các vấn đề bảo mật**:

- File `.pt` (hoặc `.pth`) thường được lưu bằng `torch.save()`, mà bản chất là sử dụng `pickle` – cơ chế này cho phép thực thi **mã Python tùy ý** khi load lại model (`torch.load()`).
- Điều này tiềm ẩn rủi ro bảo mật, đặc biệt khi load mô hình từ nguồn không tin cậy: ai đó có thể nhúng mã độc và nó sẽ **tự động chạy khi load mô hình**.
- Trong khi đó, `.safetensors` là định dạng **hoàn toàn không thực thi code**, chỉ lưu dữ liệu tensor thuần túy, vì vậy loại bỏ khả năng chèn mã độc.

### 2. `.safetensors` có tốc độ load **nhanh hơn**:

- File `.safetensors` được thiết kế để **cho phép memory mapping** và truy cập song song, giúp tăng tốc độ load mô hình, đặc biệt là khi dùng trên GPU.
- Cấu trúc của định dạng này giúp tránh việc giải nén từng phần như `.pt`, do đó tiết kiệm thời gian khởi tạo model.

### 3. Hỗ trợ rộng trong cộng đồng hiện tại:

- Các dự án lớn như Hugging Face Transformers đã **mặc định hỗ trợ và khuyến nghị dùng `.safetensors`** cho các model pre-trained.
- Với update gần đây của Pytoch (2.6.0 onward) thì khi sử dụng `torch.load()`, tham số `weights_only` đã được set thành **True** mặc định thay vì **False** (load cả code) như trước đây.
- Với các mô hình như `bert-vits`, việc chuyển sang `.safetensors` cũng giúp đồng bộ tốt hơn với các pipeline hiện đại, dễ dàng chia sẻ mà vẫn đảm bảo an toàn.

## Lưu ý khi sử dụng `torch.compile()`

### `torch.compile()` **chỉ hỗ trợ GPU có compute capability ≥ 7.0**

- `torch.compile()` tận dụng các công nghệ như **Triton**, **Inductor**, và các kỹ thuật kernel fusion hiện đại, vốn được thiết kế để hoạt động hiệu quả nhất trên kiến trúc GPU từ **Volta** trở lên (tức là compute capability ≥ 7.0).
- Nếu dùng GPU với compute capability thấp hơn sẽ không thể compile.

### Cách kiểm tra compute capability của GPU:

Có thể chạy lệnh sau trong Python:

```python
import torch
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
```


# Tổng quan về **ONNX**

## ONNX là gì?

**ONNX** (Open Neural Network Exchange) là một định dạng trung gian dùng để **xuất và chia sẻ mô hình học sâu** giữa các framework khác nhau (ví dụ: PyTorch, TensorFlow, ONNX Runtime, v.v).

Nó cho phép bạn **xuất model từ PyTorch**, sau đó **chạy inference bằng ONNX Runtime**, thường sẽ tối ưu hóa hiệu suất hơn nhờ engine inference nhẹ, đặc biệt trong môi trường production hoặc mobile.

---

## Các bước chính để sử dụng ONNX trong PyTorch

### 1. Export mô hình sang ONNX

```python
torch.onnx.export(
    model,                      # Mô hình PyTorch
    sample_input,               # Một batch input mẫu để trace
    "model.onnx",               # Tên file đầu ra
    input_names=["input"],      # (Optional) Tên tensor đầu vào
    output_names=["output"],    # (Optional) Tên tensor đầu ra
    dynamic_axes={              # (Optional) Cho phép kích thước động
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    opset_version=17,           # Phiên bản opset ONNX
    do_constant_folding=True,   # Tối ưu biểu thức hằng
    export_params=True          # Lưu trọng số vào file ONNX
)
```

> 🎯 **Lưu ý:** `opset_version` nên là 17 trở lên để đảm bảo tương thích tốt với ONNX Runtime mới.

#### Hoặc có thể dùng Optimum-CLI như ví dụ sau đây

```bash
pip install optimum[exporters]
```
Set --model dùng để export model của Pytorch hoặc TensorFlow
```bash
optimum-cli export onnx --model distilbert/distilbert-base-uncased-distilled-squad distilbert_base_uncased_squad_onnx/
```
---

### 2. Load và chạy inference bằng ONNX Runtime

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("model.onnx")
inputs = {"input": input_tensor.numpy()}
outputs = session.run(None, inputs)
```

### 3. Lưu ý về sự khác biệt giữa **ONNX** và **torch.compile()** khi compile

Với mô hình TTS, nếu dùng **ONNX**, **không thể export cả model một cách trực tiếp**. Vậy nên thường phải **chia nhỏ ra**, export riêng từng phần như `encode`, `decode`,... thì mới chạy được — khá phiền và dễ phát sinh lỗi.

Trong khi đó, với **`torch.compile()`**, chỉ cần **tạo một wrapper cho `forward()`** chứa phần tính toán chính, miễn là không có thao tác I/O hoặc logic phức tạp. Vậy là xong, compile được ngay. **Dễ hơn** so với việc phải chia nhỏ để export ONNX.

---

## So sánh hiệu năng: `torch.compile` vs ONNX

Dưới đây là kết quả thử nghiệm với mô hình Text-to-Speech (TTS), được đo theo độ dài văn bản:

| Độ dài văn bản | Số ký tự TB | `torch.compile` Time | ONNX Time | Tốc độ (`compiled` vs `ONNX`) |
|----------------|-------------|-----------------------|-----------|-------------------------------|
| Very Short     | 3.0         | 0.17s                 | 0.33s     | 🟢 `torch.compile` nhanh hơn 1.89x |
| Short          | 12.2        | 0.18s                 | 0.82s     | 🟢 Nhanh hơn 4.61x              |
| Medium         | 34.0        | 0.26s                 | 2.10s     | 🟢 Nhanh hơn 7.98x              |
| Long           | 91.7        | 0.61s                 | 6.15s     | 🟢 Nhanh hơn 10.12x             |
| Very Long      | 273.0       | 1.91s                 | 22.23s    | 🟢 Nhanh hơn 11.61x             |

### Tổng kết:

- `torch.compile` trung bình nhanh hơn **8.89 lần** so với ONNX Runtime.
- Trong tất cả các case thử nghiệm, `torch.compile` đều chiến thắng về tốc độ.

| Phương pháp    | Tổng lượt thắng | Tỷ lệ thắng |
|----------------|------------------|--------------|
| ONNX           | 0                | ❌ 0.0%      |
| Compiled       | 3                | ✅ 100.0%    |

> ✅ **Recommendation:** Dùng `torch.compile` nếu mục tiêu là inference tốc độ cao với model PyTorch.

---

## Khi nào nên dùng ONNX thay vì `torch.compile`?

Dù hiệu năng ONNX không bằng compile trong thử nghiệm này, ONNX vẫn có các lợi thế:

| ONNX | `torch.compile` |
|------|-----------------|
| ✅ **Tốt để chia sẻ hoặc deploy mô hình** sang các hệ thống khác (ví dụ: C++, Web, Mobile) | ❌ Phụ thuộc PyTorch runtime |
| ✅ Chạy được với ONNX Runtime – nhẹ hơn, dễ tích hợp | ❌ Cần toàn bộ PyTorch backend |
| ✅ Hỗ trợ tốt cho deployment cloud (Azure, Triton, etc) | ❌ Không tương thích với mọi nền tảng |
| ❌ Hiệu năng thấp hơn với TTS | ✅ Tối ưu cao với tính toán thuần tensor |
| ❌ Không linh hoạt với cấu trúc model phức tạp (phụ thuộc export) | ✅ Chạy được cả logic phức tạp không cần export |

---

## Những vấn đề thường gặp khi dùng ONNX

- ❗ **Không export được**: Một số model PyTorch dùng logic phức tạp (control flow, custom layers) không thể export sang ONNX.
- ❗ **Khác biệt về kết quả**: Do backend khác nhau, ONNX có thể cho kết quả hơi khác với PyTorch (đặc biệt với float32 → float16).
- ❗ **Cần thêm bước xử lý dữ liệu**: Các tensor đầu vào cần được convert sang `numpy`, khác với workflow PyTorch.
---

## 🙏 Credits & Acknowledgements

<div align="center">

[![Style-Bert-VITS2](https://img.shields.io/badge/Style--Bert--VITS2-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/litagin02/Style-Bert-VITS2)

Dự án này dựa trên ý tưởng và mã nguồn của **StyleBertVits2** được phát triển bởi [litagin02](https://github.com/litagin02).

</div>
