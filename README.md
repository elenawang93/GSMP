# Supplementary Code

Code to reproduce all experiments from *Geometry-Aware Simplicial Message
Passing*.

## Files

- `models.py` — Neural network architectures (SimplicialNet, GraphNet, DeepSets, MLP).
- `utils.py` — Topology construction, mesh deformation, ECT computation, training loops.
- `run_experiments.py` — Experiment runner with CLI for all paper results.

## Requirements

```
torch >= 2.0
numpy
scipy
```

For MANTRA experiments: `pip install mantra`
For FAUST experiments: `pip install torch-geometric openmesh`

## Usage

```bash
# Synthetic experiments
python run_experiments.py --exp family --hd 32 --L 4 --epochs 80
python run_experiments.py --exp ect --hd 32 --L 4 --epochs 80
python run_experiments.py --exp coboundary --hd 32 --L 4 --epochs 80
python run_experiments.py --exp permutation --hd 32 --L 4 --epochs 80
python run_experiments.py --exp confounded --hd 32 --L 4 --epochs 120

# MANTRA experiments
python run_experiments.py --exp mantra --hd 32 --L 4 --epochs 100
python run_experiments.py --exp mantra_ect --hd 32 --L 4 --epochs 100

# FAUST experiments
python run_experiments.py --exp faust_pose --hd 16 --L 3 --epochs 200
python run_experiments.py --exp faust_reg --hd 16 --L 3 --epochs 200

# Run everything
python run_experiments.py --exp all
```

Add `--device cuda` for GPU acceleration (recommended for FAUST).
