# Look at this test for inspiration
# https://github.com/pytorch/pytorch/blob/main/test/test_cpp_extensions_jit.py

import os

import torch
from torch.utils.cpp_extension import load_inline


def case_hello():
    cpp_source = """
    std::string hello_world()
    {
        return "Hello World!";
    }
    """

    module = load_inline(name='my_module',
                            cpp_sources=[cpp_source],
                            functions=['hello_world'],
                            verbose=True,
                            build_directory='./tmp',)

    print(module.hello_world())

def case_square_matrix():
    # Define the CUDA kernel and C++ wrapper
    cuda_source = """
    __global__ void square_matrix_kernel(const float* matrix, float* result, int width, int height)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < height && col < width)
        {
            int idx = row * width + col;
            result[idx] = matrix[idx] * matrix[idx];
        }
    }

    torch::Tensor square_matrix(torch::Tensor matrix)
    {
        const auto height = matrix.size(0);
        const auto width = matrix.size(1);

        auto result = torch::empty_like(matrix);

        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

        square_matrix_kernel<<<grid, block>>>(
            matrix.data_ptr<float>(), result.data_ptr<float>(), width, height);

        return result;
        }
    """

    cpp_source = "torch::Tensor square_matrix(torch::Tensor matrix);"

    # Load the CUDA kernel as a PyTorch extension
    module = load_inline(
        name='square_matrix_extension',
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=['square_matrix'],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
        build_directory='./tmp',
        # extra_cuda_cflags=['--expt-relaxed-constexpr']
    )

    a = torch.tensor([[1., 2., 3.], [4., 5., 6.]], device='cuda')
    print(module.square_matrix(a))

if __name__ == "__main__":
    os.system("rm -rf ./tmp/*")
    
    case_hello()
    case_square_matrix()
    
    print("Finish")