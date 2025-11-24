## PyTorch C++ helpers (`TORCH_CHECK`)

In `fast_matmul_extension.cu`, the `TORCH_CHECK` calls are not defined in this repo; they come from
PyTorch’s C++ API.

- The include `#include <torch/extension.h>` pulls in PyTorch’s C++/CUDA extension headers.
- Those headers (via `c10/util/Exception.h`) define the `TORCH_CHECK` macro.
- Usage pattern: `TORCH_CHECK(condition, "error message", optional, extra, args...)`.
  - If `condition` is `true`, execution continues normally.
  - If `condition` is `false`, it throws a `c10::Error` exception.
- When the extension is called from Python, a thrown `c10::Error` is translated into a regular
  Python exception with the provided message.

So in `fast_matmul_extension.cu`, `TORCH_CHECK` is just a convenient runtime assertion macro
provided by PyTorch, brought into scope by `#include <torch/extension.h>`.

