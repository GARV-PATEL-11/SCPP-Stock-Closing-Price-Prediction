# check_cuda_config.py

import torch

def check_cuda_configuration():
    """
    Checks and prints the current CUDA configuration, including
    device count, name, memory usage, and CUDA version.
    """
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)

        print(f"CUDA is available with {device_count} device(s).")
        print(f"Current device: {current_device} - {device_name}")
        print(f"CUDA version: {torch.version.cuda}")

        # Memory usage details
        props = torch.cuda.get_device_properties(current_device)
        total_mem = props.total_memory / 1e9  # Convert bytes to GB
        allocated_mem = torch.cuda.memory_allocated(current_device) / 1e9
        reserved_mem = torch.cuda.memory_reserved(current_device) / 1e9

        print(f"Total GPU memory     : {total_mem:.8f} GB")
        print(f"Currently allocated  : {allocated_mem:.8f} GB")
        print(f"Currently reserved   : {reserved_mem:.8f} GB")

        return True
    else:
        print("CUDA is not available. Using CPU instead.")
        return False


if __name__ == "__main__":
    # Select appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Run the CUDA check
    check_cuda_configuration()
