# Global Physicochemical & Structural Analyzer

A comprehensive Python utility to recursively aggregate mutation screening data across multiple projects, evaluate physicochemical and structural region distributions, calculate statistical error bars (SEM), and export publication-ready plots and Excel summary reports.

## Features

- Automatically scans project subdirectories for position summary reports.
- Normalizes numerical energy values and computes relative sequence positions (N-Terminal, Core/Middle, C-Terminal).
- Ranks introduced amino acids by binding DeltaDeltaG with standard error margins.
- Exports high-resolution matplotlib visualizations and a consolidated master Excel report.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run in the current directory:
```bash
python global_analysis.py
```

Or specify a custom root directory and output filename:
```bash
python global_analysis.py /path/to/screening_root -o Custom_Global_Report.xlsx
```
