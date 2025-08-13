# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GAOT-3D is a 3D implementation of the Geometry-Aware Operator Transformer (GAOT), a neural surrogate model for PDEs on arbitrary domains. The model combines a Multiscale Attentional Graph Neural Operator (MAGNO) encoder/decoder with a Vision Transformer processor for handling 3D unstructured mesh data.

## Common Commands

### Training and Inference
```bash
# Train a model with a single config file
python main.py --config config/examples/drivaernet/pressure/pressure.json

# Run all configs in a folder
python main.py --folder config/examples/drivaernet/

# Debug mode (no multiprocessing)
python main.py --config [config_file] --debug

# Specify GPU devices
python main.py --config [config_file] --visible_devices 0 1

# Inference only (modify config: setup.train=false, setup.test=true)
python main.py --config [config_file]
```

### Testing
```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/model/test_gaot_3d.py

# Run tests with verbose output
python -m pytest tests/ -v
```

### Data Preprocessing
```bash
# Process DrivAerNet VTK files to PyTorch Geometric format
python dataset/drivaernet/drivaer_process.py

# Process NASA CRM data
python dataset/nasa_crm/nasa_crm_process.py

# Generate DrivAerML boundary data
cd dataset/drivaerml && bash download_drivaerml_boundary.sh
```

### Environment Setup
```bash
# Create virtual environment
python -m venv venv-gaot3d
source venv-gaot3d/bin/activate  # Linux/Mac
# venv-gaot3d\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Architecture Overview

### Core Components

1. **GAOT3D Model** (`src/model/gaot_3d.py`):
   - Main model combining MAGNO encoder → Transformer processor → MAGNO decoder
   - Handles 3D latent token grids and coordinate transformations
   - Supports both full-grid and neural field training strategies

2. **MAGNO Layers** (`src/model/layers/magno.py`):
   - Graph Neural Operator for encoder/decoder operations
   - Handles message passing between physical nodes and latent tokens
   - Configurable neighbor strategies: knn, radius, bidirectional, reverse

3. **Transformer Processor** (`src/model/layers/attn.py`):
   - Vision Transformer operating on 3D patches of latent tokens
   - Supports absolute and RoPE positional embeddings
   - Configurable attention mechanisms and FFN layers

4. **Training Infrastructure** (`src/trainer/stat.py`):
   - StaticTrainer3D for time-independent 3D problems
   - Supports distributed training with DistributedSampler
   - Handles dataset statistics calculation and normalization
   - Neural field and full-grid training strategies

### Key Data Flow

1. **Input Processing**: VTK mesh data → PyTorch Geometric `.pt` files
2. **Model Forward Pass**:
   - Physical coordinates + features → MAGNO Encoder → Latent tokens
   - Latent tokens → Vision Transformer (patch-based) → Processed tokens
   - Processed tokens → MAGNO Decoder → Physical node predictions

### Configuration System

All experiments use JSON/TOML configuration files in `config/` directory:
- `dataset`: Data paths, normalization, batch size, training strategy
- `model`: GAOT3D architecture, MAGNO config, Transformer config  
- `setup`: Training/testing flags, distributed settings, device selection
- `path`: Output paths for checkpoints, plots, results, metrics database

### Important Configuration Notes

- **Edge Precomputation**: Set `dataset.update_pt_files_with_edges=true` for initial run to precompute graph edges, then set to `false` and `model.args.magno.precompute_edges=true` for training
- **Neural Field Training**: Use `dataset.training_strategy="neural_field"` for memory-efficient training on large meshes
- **Distributed Training**: Set `setup.distributed=true` and use appropriate world_size/rank settings

### Dataset Structure

Expected directory structure:
```
dataset_base_path/
├── processed_pyg/          # Processed .pt files
├── order_processed_pyg.txt # File ordering for train/val/test splits
└── [dataset_name]_norm_stats.pt  # Normalization statistics
```

### Testing Infrastructure

- Unit tests in `tests/` directory
- Model component tests: `tests/model/`
- Integration tests for GAOT3D model functionality
- Memory profiling tools in `analysis/memory/`

### Development Notes

- The codebase uses PyTorch Geometric for graph operations
- All coordinate normalization should use `rescale()` or `rescale_new()` functions
- Model supports variable numbers of nodes per sample (unstructured meshes)
- Distributed training uses PyTorch DDP with proper synchronization barriers
- Statistics are calculated once and cached for efficiency