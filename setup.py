from setuptools import setup, find_packages

setup(
    name="alpha-newton",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "transformers>=4.40.0",
        "datasets>=2.14.0",
        "trl>=0.8.0",
        "pyyaml>=6.0",
        "accelerate>=0.20.0",
        "wandb>=0.15.0",
        "peft>=0.8.0",
        "bitsandbytes>=0.41.0",
    ],
    description="Alpha-Newton LLM Post-Training Framework",
)
