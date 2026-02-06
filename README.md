# 🧠 L.U.M.I - Logical Utility Machine Intelligence

<p align="center">
  <img src="gui/assets/logo.png" alt="L.U.M.I Logo" width="120" height="120">
</p>

**L.U.M.I** (Logical Utility Machine Intelligence) is a **fully local, privacy-focused AI assistant** for Windows. It combines a beautiful modern GUI with powerful voice control capabilities—all running entirely on YOUR computer with no cloud dependency.

> 🔒 **Your data stays on your machine.** No API keys required for core functionality. No subscriptions. No data collection.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🎤 **Voice Control** | Wake word detection ("Lumi") with natural language commands |
| 💬 **AI Chat** | Interactive chat with local LLMs via Ollama with streaming responses |
| 🏠 **Smart Home** | Control TP-Link Kasa smart lights and plugs from the app |
| 📅 **Planner** | Manage calendar events, alarms, and timers |
| 📰 **Daily Briefing** | AI-curated news from Technology, Science, and Top Stories |
| 🌤️ **Weather** | Current weather and hourly forecast on your dashboard |
| 🔍 **Web Search** | Search the web through voice or chat commands |
| 🖥️ **System Monitor** | Real-time CPU and memory usage in the title bar |

---

## 📸 Screenshots

*The application features a sleek Windows 11 Fluent Design aesthetic with dark mode support.*

---

## 📋 Prerequisites

Before you begin, make sure you have:

### Required Software

| Software | Purpose | Download |
|----------|---------|----------|
| **Miniconda** | Python environment manager | [miniconda.io](https://docs.anaconda.com/miniconda/) |
| **Ollama** | Local AI model server | [ollama.com](https://ollama.com/download) |
| **NVIDIA GPU** (Recommended) | Faster AI inference | GPU with 4GB+ VRAM |

### Hardware Recommendations

- **Minimum**: 8GB RAM, any modern CPU
- **Recommended**: 16GB RAM, NVIDIA GPU with 6GB+ VRAM
- **Storage**: ~5GB for models and voice data

---

## 🚀 Quick Start Guide

Follow these steps to get L.U.M.I running on your system.

### Step 1: Install Miniconda

1. Download from [miniconda.io](https://docs.anaconda.com/miniconda/)
2. Run the installer (use default options)
3. Open **Anaconda Prompt** (Windows) or your terminal (macOS/Linux)

### Step 2: Install Ollama

1. Download and install from [ollama.com/download](https://ollama.com/download)
2. Run the installer (Ollama will start automatically as a background service)

> ✅ **Ollama runs in the background** - no need to start it manually after installation.

### Step 3: Download an AI Model

Open a terminal and pull your preferred model. You can choose from:

**🔹 Option A: Qwen3 (Recommended for most users)**
```bash
# Fast and efficient - great balance of speed and quality
ollama pull qwen3:1.7b
```

**🔹 Option B: DeepSeek R1 (Better reasoning)**
```bash
# Stronger reasoning capabilities - slightly slower
ollama pull deepseek-r1:1.5b
```

> 💡 **Tip**: You can switch models anytime in `config.py` by changing `RESPONDER_MODEL`.

Verify your model is installed:
```bash
ollama list
```

### Step 4: Clone & Set Up the Project

```bash
# Clone the repository
git clone https://github.com/your-username/project_lumi.git
cd ProjectLUMI

# Create a conda environment
conda create -n lumi python=3.11 -y

# Activate the environment
conda activate lumi

# Install dependencies
pip install -r requirements.txt
```

> ⏱️ **Note**: First installation may take 5-10 minutes as PyTorch and other large packages are downloaded.

### Step 5: GPU Setup (NVIDIA Users)

For **significantly faster** AI inference, install PyTorch with CUDA support:

```bash
# Install PyTorch with CUDA 12.4 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Verify CUDA is working:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

> 💡 **CPU-only users**: Skip this step—PyTorch will use CPU by default. It's slower but works fine.

### Step 6: Run the Application

```bash
python main.py
```

🎉 **That's it!** L.U.M.I will launch with a beautiful GUI.

---

## 🎮 GPU Acceleration

L.U.M.I benefits greatly from GPU acceleration. Here's what runs on your GPU:

| Component | GPU Benefit | Without GPU |
|-----------|-------------|-------------|
| **Router Model** | ~50ms inference | ~200ms inference |
| **Ollama LLM** | Fast streaming responses | Slower, but functional |
| **Whisper STT** | Real-time transcription | Slight delay |

### CUDA Requirements

- **NVIDIA GPU** with CUDA Compute Capability 5.0+ (GTX 900 series or newer)
- **CUDA Toolkit**: Bundled with PyTorch—no separate install needed
- **VRAM**: 4GB minimum, 6GB+ recommended

### Check Your GPU

```bash
# View GPU info and VRAM
nvidia-smi
```

---

## 🤖 Automatic Model Downloads

The following models are **downloaded automatically** on first run—no manual setup required:

| Model | Purpose | Size | Downloaded From |
|-------|---------|------|-----------------|
| **Router Model** | Intent classification | ~500MB | [Hugging Face](https://huggingface.co/nlouis/pocket-ai-router) |
| **TTS Voice** | Text-to-speech | ~50MB | [Piper Voices](https://huggingface.co/rhasspy/piper-voices) |
| **STT Model** | Speech-to-text (Whisper) | ~150MB | OpenAI Whisper |

> 📦 **First launch will take a few minutes** while these models download. Subsequent launches are instant.

---

## 🎙️ Voice Assistant Setup

L.U.M.I includes Alexa-like voice control with wake word detection.

### How It Works

1. Say **"Lumi"** to wake the assistant
2. Speak your command naturally
3. L.U.M.I processes your request and responds

### Example Voice Commands

| Command | What It Does |
|---------|--------------|
| *"Lumi, turn on the office lights"* | Controls smart lights |
| *"Lumi, set a timer for 10 minutes"* | Creates a countdown timer |
| *"Lumi, what's on my schedule today?"* | Reads your calendar |
| *"Lumi, search the web for Python tutorials"* | Performs web search |
| *"Lumi, add buy groceries to my to-do list"* | Creates a task |

### Voice Configuration

Edit `config.py` to customize:

```python
# Change wake word (default: "lumi")
WAKE_WORD = "lumi"

# Adjust sensitivity (0.0-1.0, lower = less false positives)
WAKE_WORD_SENSITIVITY = 0.4

# Enable/disable voice assistant
VOICE_ASSISTANT_ENABLED = True
```

---

## ⚙️ Configuration

All configuration is centralized in `config.py`:

### AI Models

Change the chat model in `config.py`:

```python
# The main chat model (runs on Ollama)
# Options: "qwen3:1.7b" (fast) or "deepseek-r1:1.5b" (better reasoning)
RESPONDER_MODEL = "qwen3:1.7b"

# Ollama server URL (usually no need to change)
OLLAMA_URL = "http://localhost:11434/api"
```

### Text-to-Speech

```python
# Voice model (downloads automatically on first run)
TTS_VOICE_MODEL = "en_GB-northern_english_male-medium"
```

### Weather Location

The default location is New York City. To change it:

1. Open the app
2. Go to **Settings** tab
3. Enter your latitude and longitude

---

## 🔧 Troubleshooting

### Common Issues

<details>
<summary><strong>❌ Ollama connection refused</strong></summary>

**Problem**: The app can't connect to Ollama.

**Solution**:
1. Make sure Ollama is running: `ollama serve`
2. Check if the model is downloaded: `ollama list`
3. Verify the URL in `config.py` matches your setup

</details>

<details>
<summary><strong>❌ CUDA/GPU not detected</strong></summary>

**Problem**: PyTorch is running on CPU instead of GPU.

**Solution**:
1. Install CUDA-compatible PyTorch:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```
2. Verify CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

</details>

<details>
<summary><strong>❌ Voice assistant not working</strong></summary>

**Problem**: Wake word isn't being detected.

**Solution**:
1. Check your microphone permissions
2. Ensure `realtimestt` is installed: `pip install realtimestt`
3. Try lowering `WAKE_WORD_SENSITIVITY` in `config.py`

</details>

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a pull request

---

## 📜 License

This project is open source. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [Ollama](https://ollama.com/) - Local LLM inference
- [QFluentWidgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets) - Beautiful UI components
- [Piper TTS](https://github.com/rhasspy/piper) - Lightweight text-to-speech
- [python-kasa](https://github.com/python-kasa/python-kasa) - Kasa device control
- [RealTimeSTT](https://github.com/KoljaB/RealtimeSTT) - Speech recognition

---

<p align="center">
  Made with ❤️ for local AI enthusiasts
</p>
