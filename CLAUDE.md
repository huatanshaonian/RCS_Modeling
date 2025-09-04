# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a radar cross-section (RCS) analysis project that implements Proper Orthogonal Decomposition (POD) and deep learning autoencoder methods for dimensionality reduction of RCS data. The project analyzes 100 aircraft models at 1.5GHz and 3GHz frequencies, with each model having RCS data at 8281 angle combinations (91×91 elevation/azimuth angles).

## Commands

### Running the Analysis
- **Main analysis**: `python run.py` (command-line interface with arguments)
- **Direct execution**: `python main.py` (uses default parameters)
- **Traditional GUI**: `python run_gui_fixed.bat` or `python run_gui_fixed.ps1` (fixed encoding and environment issues)
- **Modern Web Interface**: `python run_streamlit.bat` or via Streamlit
- **Test CUDA availability**: `python check_cuda.py`

### Command Line Arguments (run.py)
- `--params_path`: Path to design parameters CSV (default: ../parameter/parameters_sorted.csv)
- `--rcs_dir`: RCS data directory (default: ../parameter/csv_output)
- `--output_dir`: Output directory (default: ./results)
- `--freq`: Frequency to analyze (1.5G, 3G, or both)
- `--num_models`: Number of models to analyze
- `--num_train`: Training set sizes, comma-separated (e.g., "60,70,80")
- `--predict_mode`: Enable prediction mode
- `--param_file`: Parameter file for predictions
- `--pod_modes`: POD mode counts, comma-separated (e.g., "10,20,30,40")

### Dependencies
Install required packages: `pip install -r requirements.txt`
- Core: numpy, pandas, matplotlib, scikit-learn, scipy
- Optional: torch, torchvision, torchaudio (for autoencoder analysis)
- Web Interface: streamlit, plotly (for modern UI)

## Architecture

### Data Flow
1. **Data Loading** (`data_loader.py`): Loads design parameters and RCS data from CSV files with multi-encoding support
2. **POD Analysis** (`pod_analysis.py`): Performs singular value decomposition for dimensionality reduction
3. **Modal Analysis** (`model_analysis.py`): Visualizes modes, parameter sensitivity, and reconstruction
4. **Autoencoder Analysis** (optional): Deep learning-based dimensionality reduction using PyTorch

### Key Components

**Main Control Flow** (`main.py`, `run.py`):
- Orchestrates the entire analysis pipeline
- Handles multiple training set sizes and frequencies
- Manages data splitting and result organization

**POD Implementation** (`pod_analysis.py`):
- Uses SVD method for numerical stability
- Includes comprehensive error handling and diagnostics
- Implements energy analysis to determine optimal mode count
- Function: `perform_pod()` returns modes, eigenvalues, and mean data

**Autoencoder Module** (modular structure):
- `autoencoder_analysis.py`: Main coordinator
- `autoencoder_models.py`: StandardAutoencoder and VariationalAutoencoder classes
- `autoencoder_training.py`: Training loops with early stopping
- `autoencoder_utils.py`: Device management and data utilities
- `autoencoder_visualization.py`: Plotting and comparison functions
- `autoencoder_prediction.py`: Prediction pipeline from parameters to RCS

**User Interface System**:
- `rcs_gui.py`: Traditional tkinter GUI with full parameter configuration
- `streamlit_app.py`: Modern web interface with real-time monitoring
- `run_gui_fixed.bat/.ps1`: Fixed environment variables for GUI startup
- `run_streamlit.bat`: Streamlit web application launcher

### Data Structure

**Input Data**:
- Parameters: `parameters_sorted.csv` (24 design parameters per model, supports GBK encoding)
- RCS Data: `{model_number}_{frequency}.csv` (91×91 angle combinations, multi-encoding support)

**Output Structure**:
```
results/
├── {frequency}/              # 1.5GHz or 3GHz
│   └── train_{size}/         # Different training set sizes
│       ├── modes/            # POD mode visualizations
│       ├── reconstruction_train/  # Training set reconstruction analysis
│       ├── reconstruction_test/   # Test set reconstruction analysis
│       ├── autoencoder/      # Deep learning results
│       │   ├── {config}/     # Different configuration results
│       │   └── comparison/   # Comparison analysis
│       └── *.npy            # Saved arrays (modes, coefficients, etc.)
```

### Error Handling Strategy

The codebase implements robust error handling:
- **Data Validation**: Checks for NaN/Inf values in RCS data
- **Encoding Handling**: Multi-encoding format attempts (UTF-8, GBK, GB2312, Latin1, CP1252)
- **Numerical Stability**: SVD fallback for POD, regularization for ill-conditioned matrices  
- **GPU Management**: Automatic CPU fallback if CUDA unavailable
- **Memory Management**: Batch size optimization and memory cleanup
- **Environment Compatibility**: Tcl/Tk environment variable fixes, Windows path handling

### Key Functions

**Data Processing**:
- `load_rcs_data()`: Loads and validates RCS CSV files with multi-encoding support
- `load_parameters()`: Loads parameter files with automatic encoding detection
- `perform_pod()`: Main POD decomposition with diagnostics
- `energy_analysis()`: Determines optimal number of modes

**Analysis Functions**:
- `analyze_frequency_data()`: Main analysis loop for each frequency
- `parameter_sensitivity()`: Regression analysis between parameters and POD coefficients
- `reconstruct_rcs()`: Validates reconstruction quality

**Model Training** (if PyTorch available):
- `train_autoencoder()`: Training with early stopping and learning rate scheduling
- `perform_autoencoder_analysis()`: Main autoencoder analysis coordinator
- `create_autoencoder_prediction_pipeline()`: Creates prediction pipeline

**Interface Management**:
- `streamlit_app.py`: Multi-page web application with real-time monitoring
- `rcs_gui.py`: Traditional GUI with complete parameter configuration

## Special Considerations

- **Chinese Text Support**: Uses SimHei font for matplotlib Chinese character display
- **Large Data Handling**: Implements memory-efficient processing for 8281-dimensional RCS data
- **GPU Optimization**: Automatic device selection and mixed precision training when available
- **Reproducibility**: Uses fixed random seeds (42) for consistent results across runs
- **Modular Design**: Autoencoder components can work independently or be disabled if PyTorch unavailable
- **Encoding Compatibility**: Automatic detection and handling of different encoding formats for Chinese CSV files
- **Environment Fixes**: Automatic handling of Tcl/Tk version conflicts and path issues

## Development Notes

- The project handles both POD (traditional) and autoencoder (deep learning) approaches for comparison
- Results are saved in multiple formats (PNG images, NPY arrays, CSV statistics) for comprehensive analysis
- The codebase gracefully degrades functionality when optional dependencies are missing
- Supports both traditional GUI (tkinter) and modern web interface (Streamlit)
- Implements complete prediction pipeline from design parameters to RCS data
- Includes comprehensive statistical analysis and visualization capabilities

## Troubleshooting

**Encoding errors**: Use the fixed `data_loader.py` which automatically tries multiple encoding formats
**GUI startup issues**: Use `run_gui_fixed.bat` or `run_gui_fixed.ps1` to set correct environment variables
**Web interface**: Use `run_streamlit.bat` to launch modern web interface
**CUDA issues**: Run `python check_cuda.py` to check GPU availability

## Branch Strategy

- **main**: Primary development branch with stable features
- **streamlit**: Web interface development branch for UI enhancements and beautification
- **dev**: Experimental feature development