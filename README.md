# micrograd (notebook)

This repository contains a learning-oriented reimplementation of Andrej Karpathy's micrograd ideas. I built a small scalar-valued automatic differentiation engine (Value class), demonstrated forward computations, and performed reverse-mode backpropagation with graph visualization using Graphviz. Lastly, I trained the MLP in 20 iteration.

## Features

- Simple Value class that tracks data, gradients, and a backward function.
- Operator overloads for add, multiply, power, negation, subtraction, division, and utilities like exp() and tanh().
- Topological sort-based backward() to run reverse-mode autodiff.
- Graphviz helpers to visualize computation graphs.
- Educational examples: scalar derivatives, finite differences, a single neuron example, and cases showing gradient accumulation when nodes are reused.

## Requirements

- Python 3.8+
- numpy
- matplotlib
- graphviz (Python package) and the Graphviz system package (to render diagrams)

Install Python deps with pip:

```bash
pip install numpy matplotlib graphviz
```

On Debian/Ubuntu also install the system binary:

```bash
sudo apt-get install graphviz
```

## Running

Open the notebook in Jupyter or Colab:

```bash
jupyter notebook Micrograd.ipynb
```

or upload the notebook to Google Colab.

Work through the notebook cells to see how the Value class is built and how backward() propagates gradients.

## Differences / Notes

- This notebook is intended for learning and may differ in small implementation details from karpathy/micrograd. It follows the same core ideas (scalar Value, reverse-mode autodiff, simple NN example).
- One implementation detail to be aware of: most ops use incremental gradient accumulation (+=) in their _backward closures. The tanh implementation in the notebook assigns to self.grad rather than uses +=; if you reuse nodes feeding into tanh multiple times, you may want to change that to use += for correct accumulation semantics.

