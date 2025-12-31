import sys


def test_vllm_installation():
    print("Testing vLLM installation...\n")

    try:
        import vllm
        print(f"✓ vLLM installed successfully")
        print(f"  Version: {vllm.__version__}")
    except ImportError as e:
        print(f"✗ vLLM not installed: {e}")
        sys.exit(1)

    try:
        import torch
        print(f"\n✓ PyTorch installed")
        print(f"  Version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")
            print(f"  GPU 0: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"✗ PyTorch check failed: {e}")

    try:
        from vllm import LLM, SamplingParams
        print(f"\n✓ vLLM core imports successful")
        print(f"  LLM class available")
        print(f"  SamplingParams class available")
    except ImportError as e:
        print(f"✗ vLLM imports failed: {e}")
        sys.exit(1)

    print("\n" + "="*60)
    print("vLLM is ready for RLVR training!")
    print("="*60)


if __name__ == "__main__":
    test_vllm_installation()
