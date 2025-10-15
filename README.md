🚧 This project is under active development and currently unstable. A stable version is expected by the end of October.

## Quick Start: Set Up Three Separate Environments (Recommended)

This project consists of three parts. We recommend using separate conda environments to deploy and run each part independently:
- Segmentation API (SAM + GroundingDINO): start `api_server_sam.py`, default port 8050.
- Inpainting API (FLUX ControlNet Inpainting): start `api_server_flux_inpainting.py`, default port 8051.
- Multi-Agent Amodal Completion: run `main.py`.

---

## Part 1: Set Up the SAM Segmentation API

Directory: `Grounded-Segment-Anything/`

1) Create and activate the environment:

```zsh
conda create -n sam-api python=3.10 -y
conda activate sam-api
```

2) Install dependencies:
Follow the instructions from Grounded-Segment-Anything (see the official repository):
[IDEA-Research/Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)

3) Start the Segmentation API:

```zsh
python api_server_sam.py
```

## Part 2: Set Up the FLUX Inpainting API

Directory: `FLUX-Controlnet-Inpainting/`

1) Create and activate the environment:

```zsh
conda create -n flux-api python=3.10 -y
conda activate flux-api
```

2) Install dependencies:
Follow the instructions from FLUX-Controlnet-Inpainting (see the official repository):
[alimama-creative/FLUX-Controlnet-Inpainting](https://github.com/alimama-creative/FLUX-Controlnet-Inpainting)


3) Start the Inpainting API (default 8051):

```zsh
python api_server_flux_inpainting.py
```

## Part 3: Set Up the Multi-Agent Orchestration Environment

1) Create and activate the environment:

```zsh
conda create -n maa python=3.10 -y
conda activate maa
```

2) Install dependencies:

```zsh
pip install -r requirements.txt
```

3) Configure required service endpoints for orchestration (make sure Part 1 and Part 2 services are running and reachable):

```zsh
export SEG_BACKEND=sam
export SEG_API_BASE=http://127.0.0.1:8050
export INPAINT_API_BASE=http://127.0.0.1:8051

# LLM (choose ONE)
# OpenRouter:
export OPENROUTER_API_KEY="your_openrouter_key"
# or Azure:
export AZURE_OPENAI_API_KEY="your_azure_key"
export AZURE_OPENAI_ENDPOINT="https://your-resource-name.openai.azure.com"
export AZURE_OPENAI_API_VERSION="2024-05-01-preview"
export AZURE_OPENAI_DEPLOYMENT="gpt-4o"
```


4) CLI example (direct inpainting based on segmentation text):

```zsh
python main.py \
  --image asset/phone.jpg \
  --seg-text phone \
  --boundary-mode boundary_bbox \
  --seg-backend sam 
```

Outputs are saved by default to `./results/<seg-text>/`, e.g., `./results/phone/`.

---