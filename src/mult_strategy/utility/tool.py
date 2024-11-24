

import platform
from matplotlib import pyplot as plt
import numpy as np
import torch

from rich.console import Console

console = Console() 



def get_local_device():
    # Check for macOS and if MPS is available
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS backend on macOS")
    # Check for CUDA (NVIDIA GPUs)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA backend")
    # Default to CPU
    else:
        device = torch.device("cpu")
        print("Using CPU backend")

    console.print(f"Selected device: {device}")
    
    return device


def log_gpu_usage(global_step, device, writer):
    if device.type == "cuda":  # GPU CUDA
        writer.add_scalar('GPU Memory Allocated (MB)', torch.cuda.memory_allocated() / (1024 ** 2), global_step)
        writer.add_scalar('GPU Memory Reserved (MB)', torch.cuda.memory_reserved() / (1024 ** 2), global_step)
        writer.add_scalar('GPU Memory Free (MB)', (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / (1024 ** 2), global_step)


        
        
def create_and_log_image(global_step, writer):
    # 创建一个简单的示例图像，假设是 GPU 使用情况
    x = np.arange(10)
    y = np.random.rand(10)

    # 使用 Matplotlib 生成图像
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title('GPU Usage Example')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Usage Value')

    # 将图像转换为 NumPy 数组
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # 记录图像到 TensorBoard
    writer.add_image('/res/GPU_Usage_Plot', image, global_step, dataformats='HWC')

    plt.close(fig)  # 关闭图像防止内存泄漏
