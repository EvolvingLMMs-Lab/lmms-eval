#!/usr/bin/env python3
"""
GPU占用程序 - 使用PyTorch
占用GPU显存并确保GPU计算单元满载
"""

import torch
import time
import sys
import threading
import argparse

def check_gpus():
    """检查可用的GPU"""
    if not torch.cuda.is_available():
        print("错误: CUDA不可用，无法运行GPU占用程序")
        sys.exit(1)
    
    num_gpus = torch.cuda.device_count()
    print(f"检测到 {num_gpus} 个GPU设备")
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"  GPU {i}: {gpu_name}")
    return num_gpus

def occupy_single_gpu(device_id, memory_gb=20, duration=None):
    """
    占用单个GPU资源 - 使用PyTorch矩阵乘法确保GPU计算单元满载
    
    参数:
        device_id: GPU设备ID
        memory_gb: 要占用的显存大小(GB)，默认20GB
        duration: 运行时长(秒)，None表示无限运行
    """
    try:
        device = torch.device(f'cuda:{device_id}')
        print(f"[GPU {device_id}] 开始初始化...")
        
        # 计算矩阵大小以占用指定显存
        # 每个float32占4字节，矩阵是n*n，需要2个输入矩阵+1个输出矩阵=3个矩阵
        # memory_gb * 1024 * 1024 * 1024 / 4 / 3 = 每个矩阵的元素数
        elements_per_matrix = (memory_gb * 1024 * 1024 * 1024) // 4 // 3
        # 矩阵是方阵，n = sqrt(elements_per_matrix)
        n = int(elements_per_matrix ** 0.5)
        # 确保n是合理的值，至少1024
        n = max(1024, n)
        # 重新计算实际使用的内存
        actual_memory_gb = (n * n * 3 * 4) / (1024 * 1024 * 1024)
        
        print(f"[GPU {device_id}] 矩阵大小: {n}x{n}")
        print(f"[GPU {device_id}] 实际占用显存: {actual_memory_gb:.2f}GB")
        
        # 在GPU上创建矩阵
        A = torch.randn(n, n, dtype=torch.float32, device=device)
        B = torch.randn(n, n, dtype=torch.float32, device=device)
        
        print(f"[GPU {device_id}] 开始矩阵乘法计算，确保GPU满载...")
        
        start_time = time.time()
        iteration = 0
        
        # 使用torch.backends.cudnn.benchmark优化性能
        torch.backends.cudnn.benchmark = True
        
        while True:
            # 执行矩阵乘法 - 持续运行确保GPU计算单元满载
            # 使用 @ 运算符进行矩阵乘法，这是GPU最擅长的操作
            C = A @ B
            
            # 立即使用结果，避免被优化掉，同时增加计算量
            A = B @ C
            B = C @ A
            
            # 同步确保计算完成
            torch.cuda.synchronize(device)
            
            iteration += 1
            if iteration % 100 == 0:
                elapsed = time.time() - start_time
                # 获取GPU利用率信息
                memory_allocated = torch.cuda.memory_allocated(device) / (1024**3)
                memory_reserved = torch.cuda.memory_reserved(device) / (1024**3)
                print(f"[GPU {device_id}] 已运行 {iteration} 次迭代, 耗时 {elapsed:.2f} 秒, "
                      f"显存: {memory_allocated:.2f}GB/{memory_reserved:.2f}GB")
            
            # 检查是否达到指定时长
            if duration is not None:
                if time.time() - start_time >= duration:
                    print(f"[GPU {device_id}] 达到指定时长 {duration} 秒，停止运行")
                    break
                    
    except KeyboardInterrupt:
        print(f"\n[GPU {device_id}] 收到中断信号，停止运行")
    except Exception as e:
        print(f"[GPU {device_id}] 发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            # 清理GPU内存
            if 'A' in locals():
                del A
            if 'B' in locals():
                del B
            if 'C' in locals():
                del C
            torch.cuda.empty_cache()
            print(f"[GPU {device_id}] GPU资源已释放")
        except:
            pass

def occupy_all_gpus(memory_gb_per_gpu=20, duration=None):
    """
    占用所有GPU资源 - 使用PyTorch矩阵乘法
    
    参数:
        memory_gb_per_gpu: 每个GPU占用的显存大小(GB)，默认20GB
        duration: 运行时长(秒)，None表示无限运行
    """
    num_gpus = check_gpus()
    
    if num_gpus == 0:
        print("错误: 未检测到GPU设备")
        return
    
    print(f"\n开始占用所有 {num_gpus} 个GPU，每个GPU占用 {memory_gb_per_gpu}GB")
    print("使用PyTorch矩阵乘法确保GPU计算单元满载")
    print("按Ctrl+C停止所有GPU占用\n")
    
    # 为每个GPU创建线程
    threads = []
    for device_id in range(num_gpus):
        thread = threading.Thread(
            target=occupy_single_gpu,
            args=(device_id, memory_gb_per_gpu, duration),
            daemon=False
        )
        thread.start()
        threads.append(thread)
        time.sleep(0.5)  # 错开启动时间
    
    # 等待所有线程完成
    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        print("\n收到中断信号，正在停止所有GPU占用...")
        # 线程会在KeyboardInterrupt后自然结束

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='GPU占用程序 (使用PyTorch) - 默认占用所有GPU')
    parser.add_argument('-d', '--device', type=int, default=None,
                       help='指定GPU设备ID，不指定则占用所有GPU (默认: 占用所有)')
    parser.add_argument('-m', '--memory', type=int, default=20,
                       help='每个GPU占用的显存大小(GB) (默认: 20GB)')
    parser.add_argument('-t', '--time', type=int, default=None,
                       help='运行时长(秒)，不指定则无限运行')
    
    args = parser.parse_args()
    
    # 检查GPU
    num_gpus = check_gpus()
    
    if args.device is not None:
        # 占用指定GPU
        if args.device >= num_gpus:
            print(f"错误: GPU设备 {args.device} 不存在，只有 {num_gpus} 个GPU")
            sys.exit(1)
        occupy_single_gpu(
            device_id=args.device,
            memory_gb=args.memory,
            duration=args.time
        )
    else:
        # 占用所有GPU
        occupy_all_gpus(
            memory_gb_per_gpu=args.memory,
            duration=args.time
        )

if __name__ == '__main__':
    main()