# MazeBench

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

MazeBench is a benchmarking framework for evaluating the spatial reasoning and navigation capabilities of Large Language Models (LLMs). It assesses LLMs' ability to generate valid, step-by-step solutions to maze-solving tasks of varying complexity.

## Features

- **Multiple Model Backends**: Support for various model interfaces including HuggingFace Transformers, VLLM, and OpenAI-compatible APIs
- **Batch Processing**: Efficient batch evaluation for optimal performance
- **Detailed Metrics**: Comprehensive metrics including overall accuracy and level-specific performance
- **Extensible Design**: Easy to extend with new model backends or evaluation metrics
- **Robust Evaluation**: Thorough validation of model solutions against maze constraints

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- HuggingFace Transformers (for HuggingFace models)
- VLLM (for VLLM backend)

### Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Command Line Interface

MazeBench provides a command-line interface for easy use:

```bash
python main.py \
  --model-type <vllm|hf|openai> \
  --model-name <model_name> \
  --batch-size <batch_size>
```

### Examples

#### Using VLLM backend:

```bash
python main.py \
  --model-type vllm \
  --model-name "homebrewltd/AlphaMaze-v0.2-1.5B" \
  --temperature 0.6 \
  --batch-size 8 \
  --tensor-parallel-size 4
```

#### Using HuggingFace backend:

```bash
python main.py \
  --model-type hf \
  --model-name "homebrewltd/AlphaMaze-v0.2-1.5B" \
  --temperature 0.6 \
  --batch-size 4
```

#### Using OpenAI-compatible API:

```bash
python main.py \
  --model-type openai \
  --model-name "gpt-4-o3" \
  --api-key "your-api-key" \
  --api-base "https://api.openai.com/v1"
```

### Python API

You can also use MazeBench programmatically:

```python
from evaluator import MazeBenchEvaluator
from models.vllm_model import VLLMModel
# or from models.hf_model import HuggingFaceModel
# or from models.openai_model import OpenAIModel

# Initialize model
model = VLLMModel(
    model_name="meta-llama/Llama-2-70b-hf",
    batch_size=8
)

# Run evaluation
evaluator = MazeBenchEvaluator(model)
results = evaluator.evaluate()

# Save results
evaluator.save_results("results")
```

## Dataset

MazeBench uses the `homebrewltd/Maze-Bench-v0.2` dataset from HuggingFace, which contains maze navigation challenges at various difficulty levels.

The dataset includes:
- Mazes of different sizes and complexities
- Wall configurations that must be navigated around
- Origin and target positions
- Multiple difficulty levels
- 
### Maze Format

Mazes are represented in a structured format using tokens:

```
<|row-col|><|wall_configuration|><|marker|>
```

Where:
- `row-col`: Grid position (e.g., `0-0`)
- `wall_configuration`: Describes walls (e.g., `up_left_wall`)
- `marker`: Indicates special cells (e.g., `origin`, `target`, `blank`)

### Solution Format

Solutions are sequences of directional moves:

```
<|up|> <|down|> <|left|> <|right|>
```

The solution is correct if it navigates from the origin to the target without crossing walls.

## Output Format

The evaluation results are saved as a JSON file with the following structure:

```json
{
  "model_name": "model-name",
  "timestamp": "2025-02-25T12:00:00.000000",
  "overall_accuracy": 75.0,
  "level_accuracies": {
    "1": {
      "accuracy": 90.0,
      "correct": 9,
      "total": 10
    },
    "2": {
      "accuracy": 60.0,
      "correct": 6,
      "total": 10
    }
  },
  "detailed_results": [
    {
      "level": "1",
      "prompt": "...",
      "solution": "...",
      "is_correct": true
    }
  ]
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.