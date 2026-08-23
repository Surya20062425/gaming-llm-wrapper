# 🎮 GameMaster AI - Gaming LLM Wrapper

A specialized LLM wrapper for gaming assistance using open-source models (Llama 3, Mistral, etc.). Provides builds, strategies, lore exploration, patch analysis, coaching, and more for popular games.

## ✨ Features

- **🏗️ Build Optimizer** - Create optimized builds for champions/heroes/characters
- **🧠 Strategy Advisor** - Get strategic advice for game situations
- **📜 Lore Explorer** - Explore game lore, stories, and character backgrounds
- **📋 Patch Analyzer** - Analyze game patches and balance changes
- **📊 Meta Analyzer** - Analyze current meta, tier lists, and pro play trends
- **🎓 Coach Mode** - Personalized coaching and improvement plans
- **⚔️ Counter Picks** - Get counter-pick recommendations
- **👥 Team Composition** - Analyze and suggest team compositions
- **🛡️ Itemization Guide** - Detailed itemization guides
- **⚡ Mechanics Guide** - Mechanics guides, combos, and techniques

## 🎮 Supported Games

- League of Legends
- Dota 2
- Counter-Strike 2
- Valorant
- World of Warcraft
- Final Fantasy XIV
- Elden Ring
- Baldur's Gate 3
- Cyberpunk 2077
- The Witcher 3
- Minecraft
- Roblox
- Fortnite
- Apex Legends
- Overwatch 2

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended) or CPU
- 8GB+ RAM (16GB+ recommended for 8B models)
- 10GB+ disk space for model weights

### Installation

```bash
# Clone the repository
cd gaming-llm-wrapper

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For RAG knowledge base (optional but recommended)
pip install sentence-transformers chromadb
```

### Configuration

Edit `config/config.yaml` to customize:
- Model selection (Llama 3 8B/70B, Mistral 7B, Zephyr 7B, Phi-3 Mini)
- Quantization (4bit, 8bit, none)
- Device (auto, cuda, cpu, mps)
- Gaming features toggles
- Knowledge base settings

### Running the Application

#### Streamlit Web UI (Recommended)
```bash
streamlit run ui/streamlit_app.py
```
Then open http://localhost:8501

#### Gradio Web UI
```bash
python ui/gradio_app.py
```
Then open http://localhost:7860

#### Command Line Interface
```bash
# Interactive mode
python -m src.cli --interactive

# Single question
python -m src.cli --game "League of Legends" --question "Best build for Jinx" --type build_optimizer --champion "Jinx" --role "ADC"

# Get coaching
python -m src.cli --game "Valorant" --type coach_mode --rank "Gold 2" --role "Duelist" --champion "Jett" --question "How to improve aim"

# List supported games
python -m src.cli --list-games

# List prompt types
python -m src.cli --list-types
```

## 📁 Project Structure

```
gaming-llm-wrapper/
├── config/
│   └── config.yaml          # Main configuration file
├── src/
│   ├── __init__.py          # Package exports
│   ├── model_manager.py     # Model loading, quantization, inference
│   ├── prompts.py           # Gaming-specific prompt templates
│   ├── gaming_llm.py        # Main wrapper class
│   ├── knowledge_base.py    # RAG system for gaming knowledge
│   └── cli.py               # Command-line interface
├── ui/
│   ├── streamlit_app.py     # Streamlit web interface
│   └── gradio_app.py        # Gradio web interface
├── data/
│   └── gaming_knowledge/    # Knowledge base documents (auto-created)
├── models/                  # Model cache directory
├── cache/                   # General cache
├── logs/                    # Log files
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🔧 Configuration Options

### Model Configuration

```yaml
model:
  name: "llama-3-8b"              # Model identifier
  model_id: "meta-llama/Meta-Llama-3-8B-Instruct"  # HuggingFace model ID
  quantization: "4bit"            # 4bit, 8bit, or none
  device: "auto"                  # auto, cuda, cpu, mps
  max_context_length: 8192        # Context window
  temperature: 0.7                # Generation temperature
  top_p: 0.9                      # Nucleus sampling
  max_new_tokens: 2048            # Max response length
```

### Knowledge Base (RAG)

```yaml
knowledge_base:
  enabled: true
  data_path: "./data/gaming_knowledge"
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  top_k: 5
  chunk_size: 512
  chunk_overlap: 50
```

Add your own gaming guides, patch notes, and strategies to `data/gaming_knowledge/{game}/{category}/` as `.txt` or `.md` files.

## 💡 Usage Examples

### Build Optimization
```python
from src import create_gaming_llm

gaming_llm = create_gaming_llm()
gaming_llm.load_model()

response = gaming_llm.get_build(
    game="League of Legends",
    champion_hero="Jinx",
    role="ADC",
    player_rank="Diamond 2",
    patch_version="14.12"
)
print(response.content)
```

### Strategy Advice
```python
response = gaming_llm.get_strategy(
    game="Dota 2",
    situation="How to play from behind as a position 4 support",
    player_rank="Ancient 3",
    role="Support"
)
```

### Coaching
```python
response = gaming_llm.get_coaching(
    game="Valorant",
    struggles="I'm hardstuck Platinum, main Jett, struggle with entry fragging and trading",
    player_rank="Platinum 2",
    role="Duelist",
    champion_hero="Jett"
)
```

### Counter Picks
```python
response = gaming_llm.get_counters(
    game="League of Legends",
    enemy_champion="Yasuo",
    role="Mid",
    player_rank="Gold 1"
)
```

## 🧠 Knowledge Base

The RAG system enhances responses with relevant gaming knowledge. Add documents to:

```
data/gaming_knowledge/
├── League of Legends/
│   ├── build/
│   │   ├── jinx_adc.md
│   │   └── ...
│   ├── strategy/
│   ├── lore/
│   ├── patch/
│   └── meta/
├── Dota 2/
│   └── ...
└── Valorant/
    └── ...
```

Run `create_sample_knowledge_base()` to populate with example data.

## 🐳 Docker Support

```bash
# Build image
docker build -t gamemaster-ai .

# Run with GPU support
docker run --gpus all -p 8501:8501 gamemaster-ai

# Or use docker-compose
docker-compose up
```

## 📝 Model Recommendations

| Model | VRAM (4bit) | VRAM (8bit) | Quality | Speed |
|-------|-------------|-------------|---------|-------|
| Llama 3 8B Instruct | ~6 GB | ~10 GB | ⭐⭐⭐⭐⭐ | Fast |
| Llama 3 70B Instruct | ~40 GB | ~70 GB | ⭐⭐⭐⭐⭐ | Slow |
| Mistral 7B Instruct | ~5 GB | ~9 GB | ⭐⭐⭐⭐ | Fast |
| Zephyr 7B Beta | ~5 GB | ~9 GB | ⭐⭐⭐⭐ | Fast |
| Phi-3 Mini 4K | ~3 GB | ~5 GB | ⭐⭐⭐ | Very Fast |

## ⚠️ Important Notes

1. **Model Access**: Llama 3 models require [HuggingFace access](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
2. **Knowledge Cutoff**: Model knowledge is limited to training data cutoff
3. **Verification**: Always verify AI advice with current game resources
4. **Performance**: 4bit quantization recommended for consumer GPUs
5. **First Run**: Model download may take 5-15 minutes depending on connection

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your changes
4. Submit a pull request

## 📄 License

MIT License - Feel free to use and modify for your gaming needs!

## 🙏 Acknowledgments

- Meta AI for Llama 3
- HuggingFace for Transformers
- Streamlit & Gradio teams for UI frameworks
- Gaming communities for knowledge and inspiration

---

**Happy Gaming! 🎮** 

*GameMaster AI - Your Intelligent Gaming Companion*