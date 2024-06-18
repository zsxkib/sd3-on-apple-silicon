# Stable Diffusion 3 on Apple Silicon (MPS)

This code allows you to run Stable Diffusion 3 on Apple Silicon devices using the MPS backend.

## Prerequisites

- Python 3.11
- Conda
- Hugging Face API token (optional)

## Setup

1. Create a new Conda environment:

```bash
conda create -n sd3 python=3.11 -y
conda activate sd3
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Set your Hugging Face API token (if required):

```bash
export HF_API_TOKEN=your_token_here
```

You can add this line to your `.zshrc` file to make it persistent.

## Usage

1. Save the provided code as `sd3-on-mps.py`.

2. Run the script:

```bash
python sd3-on-mps.py
```

3. The generated image will be saved as `sd3-output-mps.png` in the same directory.

## Notes

- The code automatically detects the available device (MPS, CUDA, or CPU) and uses the best one.
- If your system has less than 64 GB of RAM, the code enables attention slicing to reduce memory usage.
- The model cache is set to `./sd3-cache` to avoid downloading the model repeatedly.
- The image generation process may take some time depending on your hardware.

That's it! You can now generate images using Stable Diffusion 3 on your Apple Silicon device.