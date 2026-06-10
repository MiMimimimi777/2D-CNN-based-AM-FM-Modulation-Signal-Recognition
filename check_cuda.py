
import sys
import os

print("=" * 60)
print("CUDA环境检测")
print("=" * 60)

print(f"\nPython版本: {sys.version}")
print(f"Python可执行文件: {sys.executable}")

print(f"\n当前工作目录: {os.getcwd()}")

try:
    import torch
    print(f"\nPyTorch已安装: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  总显存: {props.total_memory / 1024**3:.2f} GB")
            print(f"  计算能力: {props.major}.{props.minor}")
        
        print(f"\n当前CUDA设备: {torch.cuda.current_device()}")
        print(f"当前设备名称: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("\nCUDA不可用，PyTorch将使用CPU")
except ImportError as e:
    print(f"\nPyTorch未安装或导入失败: {e}")

print("\n" + "=" * 60)
