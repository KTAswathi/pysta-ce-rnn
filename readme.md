# pysta-cernn

(Ongoing research) code for training and analysing cortically embedded recurrent neural networks for planning representations.

This repository is adapted from Kris Jensen’s spacetime attractor/RNN codebase and is currently being extended to test whether planning representations in RNNs can develop spatial gradients when the recurrent units are embedded on an mPFC cortical surface.

The project is under active development

## Installation

Create and activate a conda environment:

```bash
conda create -n pysta python=3.12 pip
conda activate pysta
```

Install the package requirements:

```bash
pip install -r requirements.txt
pip install -e .
```

Current `requirements.txt`:

```text
numpy<2
scikit-learn
scipy

matplotlib
svgpathtools
svgpath2mpl

torch==2.2.2

gdist
nibabel
nilearn
```
