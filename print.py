import torch
import subprocess
import re
from loguru import logger
def print_gpu_memory():
    """Print available memory for all GPUs detected by PyTorch, including system-wide usage."""
    if not torch.cuda.is_available():
        logger.info("CUDA is not available. No GPUs detected.")
        return
    
    gpu_count = torch.cuda.device_count()
    logger.info(f"Found {gpu_count} GPU(s):")
    
    # Get system-wide GPU usage using nvidia-smi
    try:
        nvidia_smi_output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,gpu_name,memory.total,memory.used,memory.free', 
             '--format=csv,noheader,nounits'],
            universal_newlines=True
        )
        
        # Parse the nvidia-smi output
        system_gpu_info = {}
        for line in nvidia_smi_output.strip().split('\n'):
            values = [val.strip() for val in line.split(',')]
            if len(values) >= 5:
                gpu_idx = int(values[0])
                gpu_name = values[1]
                total_mem_mb = float(values[2])
                used_mem_mb = float(values[3])
                free_mem_mb = float(values[4])
                
                system_gpu_info[gpu_idx] = {
                    'name': gpu_name,
                    'total_memory_gb': total_mem_mb / 1024,
                    'used_memory_gb': used_mem_mb / 1024,
                    'free_memory_gb': free_mem_mb / 1024
                }
        
        system_info_available = True
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.warning("Warning: Could not get system-wide GPU usage (nvidia-smi not available)")
        system_info_available = False
    
    # Get PyTorch-specific memory usage
    for i in range(gpu_count):
        # Get device properties
        device_properties = torch.cuda.get_device_properties(i)
        device_name = device_properties.name
        
        # Get total memory in GB
        total_memory = device_properties.total_memory / (1024**3)
        
        # Get allocated and cached memory by PyTorch
        allocated_memory = torch.cuda.memory_allocated(i) / (1024**3)
        reserved_memory = torch.cuda.memory_reserved(i) / (1024**3)
        
        # Calculate free memory from PyTorch's perspective
        free_memory_pytorch = total_memory - allocated_memory
        
        logger.info(f"GPU {i}: {device_name}")
        logger.info(f"  PyTorch's view:")
        logger.info(f"    - Total memory: {total_memory:.2f} GB")
        logger.info(f"    - Allocated memory: {allocated_memory:.2f} GB")
        logger.info(f"    - Reserved memory: {reserved_memory:.2f} GB")
        logger.info(f"    - Available memory (PyTorch): {free_memory_pytorch:.2f} GB")
        
        # Print system-wide information if available
        if system_info_available and i in system_gpu_info:
            info = system_gpu_info[i]
            logger.info(f"  System-wide view (all processes):")
            logger.info(f"    - Total memory: {info['total_memory_gb']:.2f} GB")
            logger.info(f"    - Used memory (all processes): {info['used_memory_gb']:.2f} GB")
            logger.info(f"    - Free memory: {info['free_memory_gb']:.2f} GB")
        logger.info("")

# Call the function to print GPU memory information
print_gpu_memory()