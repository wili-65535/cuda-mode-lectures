# Look at this test for inspiration
# https://github.com/pytorch/pytorch/blob/main/test/test_cpp_extensions_jit.py

import os

import torch
from torch.utils.cpp_extension import load_inline

def case_hello_world():
    cpp_source = """
    std::string hello_world()
    {
        return "Hello World!";
    }
    """

    module = load_inline(
        name="module",
        cpp_sources=cpp_source,
        cuda_sources=None,
        functions=["hello_world"],
        extra_cflags=None,
        extra_cuda_cflags=None,
        extra_ldflags=None,
        extra_include_paths=None,
        build_directory="./tmp",
        verbose=True,
        with_cuda=None,
        is_python_module=True,
        with_pytorch_error_handling=True,
        keep_intermediates=True,
        use_pch=False,
    )

    print(module.hello_world())

def case_square_matrix_torch():
    cpp_source = """
    at::Tensor my_square(at::Tensor x)
    {
        return x * x;
    }
    """

    module = load_inline(
        name="module",
        cpp_sources=cpp_source,  # can be a list
        functions=["my_square"],
        verbose=True,
        build_directory="./tmp",
    )
    a = torch.tensor([[1., 2., 3.], [4., 5., 6.]], device="cuda")
    print(module.my_square(a))

def case_square_matrix_cpp():
    cuda_source = """
    __global__ void square_matrix_kernel(const float* __restrict__ p_in, float* __restrict__ p_out, int width, int height)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < height && col < width)
        {
            int idx = row * width + col;
            p_out[idx] = p_in[idx] * p_in[idx];
        }
    }

    torch::Tensor square_matrix(torch::Tensor p_in)
    {
        const auto height = p_in.size(0);
        const auto width = p_in.size(1);

        auto p_out = torch::empty_like(p_in);

        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

        square_matrix_kernel<<<grid, block>>>(p_in.data_ptr<float>(), p_out.data_ptr<float>(), width, height);

        return p_out;
        }
    """

    cpp_source = "torch::Tensor square_matrix(torch::Tensor p_in);"

    module = load_inline(
        name="square_matrix_extension",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["square_matrix"],
        verbose=True,
        with_cuda=True,
        build_directory="./tmp",
    )

    a = torch.tensor([[1., 2., 3.], [4., 5., 6.]], device="cuda")
    print(module.square_matrix(a))

def case_square_matrix_cuda():
    cuda_kernel = """
    extern "C" __global__
    void square_kernel(const float* __restrict__ input, float* __restrict__ output, int size)
    {
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < size) {
            output[index] = input[index] * input[index];
        }
    }
    """

    module = load_inline(
        name="square",
        cpp_sources="",
        cuda_sources=cuda_kernel,
        functions=["square_kernel"],
        verbose=True,
        with_cuda=True,
        build_directory="./tmp",
    )

    def square(input):
        output = torch.empty_like(input)
        block = 1024
        grid = (input.numel() + (block - 1)) // block
        module.square_kernel(grid, block, input.data(), output.data(), input.numel())
        return output

    a = torch.randn(100, device="cuda")
    print(square(a))

if __name__ == "__main__":
    os.system("rm -rf tmp && mkdir tmp")

    case_hello_world()
    case_square_matrix_torch()
    case_square_matrix_cpp()
    #case_square_matrix_cuda()  # problem to build

    print("Finish")
