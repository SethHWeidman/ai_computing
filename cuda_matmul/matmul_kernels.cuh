#pragma once

// Shared entry point exposed at the repo root so other demos (e.g. PyTorch extension) can include
// the kernels without knowing about the intro_to_cuda submodule layout. The real definitions live
// inside the subrepo so it still builds independently.
#include "../intro_to_cuda/demo3_matmul/matmul_kernels.cuh"
