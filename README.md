<div align="center">
  <img src="public/logo.png" alt="Kani TTS Logo" width="150"/>

  [![](https://dcbadge.limes.pink/api/server/https://discord.gg/NzP3rjB4SB?style=flat)](https://discord.gg/NzP3rjB4SB) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

  # Kani TTS
A fast, modular and human-like TTS that generates high-quality speech from text input.
</div>

## Models

- [kani-tts-400m-en](https://huggingface.co/nineninesix/kani-tts-400m-en) - English

- [kani-tts-400m-zh](https://huggingface.co/nineninesix/kani-tts-400m-zh) - Chinese

- [kani-tts-400m-de](https://huggingface.co/nineninesix/kani-tts-400m-de) - German

- [kani-tts-400m-ar](https://huggingface.co/nineninesix/kani-tts-400m-ar) - Arabic

- [kani-tts-400m-es](https://huggingface.co/nineninesix/kani-tts-400m-es) - Spanish

- [kani-tts-400m-ko](https://huggingface.co/nineninesix/kani-tts-400m-ko) - Korean

- [kani-tts-370m-expo2025-osaka-ja](https://huggingface.co/nineninesix/kani-tts-370m-expo2025-osaka-ja) - Japanese

- [kani-tts-400m-0.3-pt](https://huggingface.co/nineninesix/kani-tts-400m-0.3-pt) - Pretrained checkpoint v0.3

- [kani-tts-370m multilingual](https://huggingface.co/nineninesix/kani-tts-370m) - English, Spanish, Chinese, German, Korean, Arabic

- [kani-tts-370m-mlx](https://huggingface.co/nineninesix/kani-tts-370m-MLX) - Multilingual model for Apple Silicon

- [kani-tts-450m-0.2-pt](https://huggingface.co/nineninesix/kani-tts-450m-0.2-pt) - Pretrained checkpoint v0.2 for posttraining and fine-tuning on custom datasets.

- [nemo-nano-codec-22khz-0.6kbps-12.5fps-MLX](https://huggingface.co/nineninesix/nemo-nano-codec-22khz-0.6kbps-12.5fps-MLX) - MLX implementation of NVIDIA NeMo NanoCodec, a lightweight neural audio codec.


**Notes:**
- Primarily optimized for English
- Performance degrades with inputs >1000 tokens
- Limited emotional expressivity without fine-tuning


## Inference

Kani TTS offers multiple inference options optimized for different hardware:

### Basic Example (GPU/CPU)
The basic inference example runs on both GPU and CPU, making it accessible for various hardware setups. Check the `examples/basic` in this repository for getting started.

### vLLM (NVIDIA GPU)
For high-performance inference on NVIDIA GPUs, use [KaniTTS-vLLM](https://github.com/nineninesix-ai/kanitts-vllm). This option is super fast and provides an OpenAI compatible API, making it easy to integrate with existing tools and workflows.

### MLX (Apple Silicon)
For Apple Silicon users, we provide an optimized [KaniTTS-MLX](https://github.com/nineninesix-ai/kani-mlx) that takes full advantage of the unified memory architecture and Neural Engine on M1/M2/M3 chips.


### NeMo NanoCodec
- **Base Model:** [NVIDIA NeMo NanoCodec](https://developer.nvidia.com/nemo)
- **License:** [NVIDIA Open Model License](https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf)
- **Sample Rate:** 22.05 kHz
- **Purpose:** Neural audio compression/decompression for TTS pipelines

---

#### GPU Benchmark Results

| GPU Model | VRAM | Cost ($/hr) | RTF |
|-----------|------|-------------|-----|
| RTX 5090 | 32GB | $0.423 | 0.190 |
| RTX 4080 | 16GB | $0.220 | 0.200 |
| RTX 5060 Ti | 16GB | $0.138 | 0.529 |
| RTX 4060 Ti | 16GB | $0.122 | 0.537 |
| RTX 3060 | 12GB | $0.093 | 0.600 |

*Lower RTF is better (< 1.0 means faster than real-time). Benchmarks conducted on [Vast AI](https://vast.ai/).*

---

## Dataset Preparation

### 1. Audio Dataset Collection

You can prepare your audio dataset using [Datamio](https://app.datamio.dev/), our active community members. Datamio provides tools to help you collect, organize, and manage high-quality audio datasets for TTS training.

### 2. Audio Processing Pipeline

After collecting your raw audio dataset, you need to process it for training. Check out this audio processing pipeline: [nano-codec-dataset-pipeline](https://github.com/nineninesix-ai/nano-codec-dataset-pipeline)

This pipeline handles:
- Audio preprocessing and normalization
- Feature extraction
- Dataset formatting for training
- Quality validation

---

## Finetuning

For finetuning KaniTTS on your own dataset, check out this comprehensive finetuning pipeline: [KaniTTS-Finetune-pipeline](https://github.com/nineninesix-ai/KaniTTS-Finetune-pipeline)

This pipeline provides:
- Step-by-step finetuning guides
- Configuration templates
- Training scripts optimized for different hardware setups
- Evaluation code to assess model performance
- Best practices for achieving high-quality results

---

## App Examples

- [ComfyUI node](https://github.com/wildminder/ComfyUI-KaniTTS) by [WildAi](https://github.com/wildminder)

- [NextJS basic app](https://github.com/nineninesix-ai/open-audio). It uses the OpenAI npm package to connect to the OpenAI-compatible server API provided by [kanitts-vllm](https://github.com/nineninesix-ai/kanitts-vllm).

- [Livekit Agent](https://github.com/nineninesix-ai/livekit-agent) - A real-time voice AI assistant built with LiveKit Agents framework, featuring speech-to-text, language processing, and text-to-speech capabilities.

---

## Areas of improvement

We're continuously working to enhance KaniTTS. Here are key areas where we're focusing our efforts:

### Core Architecture
- **Create new LLM for TTS exclusively** - Develop a specialized LLM designed specifically for text-to-speech generation, optimized for audio token prediction rather than adapted from general-purpose LLMs

### Model Enhancements
- **Add more languages** - Expand support beyond the current languages to cover more language families and dialects
- **Add more speakers** - Increase speaker diversity with different accents, age groups, and voice characteristics
- **Voice cloning examples** - Provide tutorials and code examples for cloning custom voices from audio samples

### Audio Codec Improvements
- **Fine-tune codec** - Optimize the existing NanoCodec for better audio quality and compression efficiency
- **Create new codec** - Develop a next-generation neural audio codec with improved naturalness and lower latency

### Dataset Development
Build and release high-quality, diverse audio datasets for training and fine-tuning.
- Multi-speaker datasets across different languages
- Domain-specific datasets (conversational, storytelling, professional voice-over)
- Benchmark datasets for evaluation


If you're interested in contributing to any of these areas, please check our [Contributing](#contributing) section and join our [Discord server](https://discord.gg/NzP3rjB4SB).

---

## License

Apache 2. See [LICENSE](LICENSE) file for details.

---

## Contributing

We're **open for community contributions**! KaniTTS is built with the community, and we welcome contributions of all kinds:

- **Code contributions** - Bug fixes, new features, optimizations, and documentation improvements
- **Model contributions** - Fine-tuned models, voice clones, and language-specific adaptations
- **Dataset contributions** - High-quality audio datasets for training and evaluation
- **Examples and tutorials** - Integration examples, use cases, and guides
- **Bug reports and feature requests** - Help us improve by reporting issues and suggesting enhancements

**How to contribute:**
1. Check our [Areas of improvement](#areas-of-improvement) section for current priorities
2. Join our [Discord server](https://discord.gg/NzP3rjB4SB) to discuss ideas and get support
3. Submit issues or pull requests on GitHub
4. Share your projects and use cases with the community







