# Pipeline: Legislative Framing Analysis

A comprehensive end-to-end pipeline for analyzing framing in legislative debates using Large Language Models (LLMs) and extractive methods.

## Overview

This project implements a robust methodology for extracting and analyzing political frames from Chilean legislative session transcripts. It combines multiple approaches:

- **LLM-based summarization** (GPT models)
- **Frame extraction** (open-world and closed taxonomy)
- **Taxonomic normalization** using canonical categories
- **Reliability assessment** (ICC, κ coefficients)
- **Validity evaluation** (Pearson correlation)
- **Ablation studies** and baselines
- **LaTeX-ready outputs** for academic publication

## Features

### Core Analysis
- Multi-model LLM summarization with cost tracking
- Open-world and closed-world frame identification
- Automatic canonicalization to 13-category taxonomy
- Frame×Session binary and count matrices

### Validation & Reliability
- Inter-rater reliability using Intraclass Correlation Coefficient (ICC)
- Narrative coherence assessment via LLM judges
- Mathematical coherence using TF-IDF cosine similarity
- Macro-F1 and Cohen's κ against gold standard (optional)

### Methodology
- Extractive baseline using TF-IDF centroids
- Comprehensive ablation studies (5 configurations)
- Procedural content filtering
- Multi-encoding file support (UTF-8, Latin1, CP1252)

## 🛠️ Installation

### Prerequisites
- Python 3.11+
- Conda or pip
- OpenAI API key

### Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd wdke-pipeline

# Create conda environment
conda env create -f environment.yml
conda activate wdke

# Install additional dependencies
pip install python-dotenv jinja2

# Set up OpenAI API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## 🚀 Usage

### Basic Execution

```bash
python pipeline_wdke.py
```

This will:
1. Process all `.txt` files in `data/WDKE/`
2. Generate summaries using LLM
3. Extract frames (open and closed taxonomy)
4. Create LaTeX tables and figures
5. Output all artifacts to `out/`

### Configuration

Edit the `CONFIG` class in `pipeline_wdke.py`:

```python
class CONFIG:
    SEED = 42
    DATA_DIR = "data/WDKE"
    OUT_DIR = "out"
    MAX_TEXT_CHARS = 16000
    LLM_MODELS = [
        {"name": "gpt-4o-mini", "temperature": 0.0},
        {"name": "gpt-4o-mini", "temperature": 0.7},
    ]
    GOLD_PATH = None  # Optional: path to gold standard
```

## 📁 Project Structure

```
├── data/
│   └── WDKE/                    # Legislative session transcripts
│       ├── sesion59.txt
│       ├── session61.txt
│       └── ...
├── out/                         # Generated outputs
│   ├── *.csv                    # Data tables
│   ├── *.tex                    # LaTeX tables
│   ├── *.pdf                    # Figures
│   └── repro_*.json            # Reproducibility info
├── pipeline_wdke.py             # Main pipeline script
├── environment.yml              # Conda dependencies
├── .env                         # Environment variables
└── README.md                    # This file
```

## 📈 Outputs

The pipeline generates publication-ready artifacts:

### Tables (CSV + LaTeX)
- `frames_frecuencia_filtrada.csv` - Raw frame frequencies
- `frames_top10_normalizados.tex` - Top 10 canonical frames
- `frame_session_matrix_binary.tex` - Binary frame presence matrix
- `frame_session_matrix_counts.tex` - Frame count matrix
- `table_ablations.tex` - Ablation study results

### Figures (PDF)
- `fig_scatter_coh_vs_judges.pdf` - Validity scatter plot
- `fig_heatmap_frames_sessions.pdf` - Frame×Session heatmap

### Data Files
- `eval_coherence_scores.csv` - Coherence evaluation results
- `baseline_multillm_costs.csv` - Model cost comparison
- `repro_runtime.json` - Reproducibility metadata

## 🔬 Methodology

### Frame Taxonomy (13 Categories)
- Crisis/Urgency
- Human Rights
- Equity/Inequality
- Institutionality/Legality
- Responsibility/Accountability
- Participation/Dialogue/Unity
- Bureaucracy
- Social Protection
- Municipal Autonomy
- Economy/Prices
- Security/Public Order
- Democracy/Constitution
- Environment

### Ablation Configurations
1. **Full**: All components enabled
2. **No Procedural**: Without procedural filtering
3. **No Regex**: Without regex canonicalization
4. **No LLM Remap**: Without LLM-based "Other" remapping
5. **No Closed Tax**: Without closed taxonomy constraint

## 📊 Evaluation Metrics

- **ICC(2,1) & ICC(2,k)**: Inter-rater reliability for LLM judges
- **Pearson r**: Correlation between mathematical and LLM coherence
- **Macro-F1**: Frame identification performance vs. gold standard
- **Cohen's κ**: Agreement coefficient vs. gold standard

## 🔧 Technical Details

### Robust File Handling
Supports multiple encodings (UTF-8, Latin1, CP1252, ISO-8859-1) for diverse input files.

### Reproducibility
- Fixed random seeds
- Version tracking for all dependencies
- Complete prompt logging
- Runtime environment capture

### Error Handling
- API retry logic with exponential backoff
- Graceful degradation for encoding issues
- Comprehensive error logging

## 📝 Citation

If you use this pipeline in your research, please cite:

```bibtex
@misc{wdke-pipeline-2024,
  title={WDKE Pipeline: Legislative Framing Analysis using Large Language Models},
  author={[Your Name]},
  year={2024},
  url={[Repository URL]}
}
```

## 📄 License

[Add your preferred license]

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📞 Support

For questions or issues, please open a GitHub issue or contact [your-email].

---

**Note**: This pipeline requires an OpenAI API key with sufficient quota. Costs vary depending on text volume and model selection.