import os
import re
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import logging
from typing import Dict, List, Optional
import json
from datetime import datetime
from models.base import BaseModel
from utils import extract_answer, benchmark_maze_solution
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mazebench_eval.log')
    ]
)

class MazeBenchEvaluator:
    """Main evaluator class for MazeBench."""
    
    def __init__(self, model: BaseModel):
        self.model = model
        self.results = {
            "model_name": model.model_name,
            "timestamp": datetime.now().isoformat(),
            "overall_accuracy": 0.0,
            "level_accuracies": {},
            "detailed_results": []
        }

    def load_dataset(self) -> pd.DataFrame:
        """Load the MazeBench dataset."""
        dataset = load_dataset("homebrewltd/Maze-Bench-v0.2", split="test")
        return dataset.remove_columns(['Maze_image'])

    def evaluate_solution(self, maze_text: str, solution: str) -> bool:
        """Evaluate a single solution."""
        return benchmark_maze_solution(maze_text, solution)

    def evaluate(self) -> Dict:
        """Run the full evaluation."""
        dataset = self.load_dataset()
        total_correct = 0
        level_stats = {}

        logging.info(f"Starting evaluation of model: {self.model.model_name}")

        batch_size = self.model.batch_size
        num_batches = len(dataset) // batch_size + (len(dataset) % batch_size > 0)

        with tqdm(total=num_batches, desc="Evaluating", unit="batch") as pbar:
            for batch in dataset.iter(batch_size=batch_size):
                prompts = [self.model.format_prompt(prompt) for prompt in batch['Prompt']]

                try:
                    solutions = []
                    answers = self.model.generate(prompts)
                    for answer in answers:
                        solutions.append(extract_answer(answer))

                    for maze, level, solution in zip(batch['Prompt'], batch['Level'], solutions):
                        is_correct = self.evaluate_solution(maze, solution)
                        if level not in level_stats:
                            level_stats[level] = {"correct": 0, "total": 0}
                        level_stats[level]["total"] += 1
                        if is_correct:
                            level_stats[level]["correct"] += 1
                            total_correct += 1
                        self.results["detailed_results"].append({
                            "level": level,
                            "maze": maze,
                            "solution": solution,
                            "is_correct": is_correct
                        })
                        
                except Exception as e:
                    logging.error(f"Error processing batch: {e}")
                    continue

                pbar.update(1)

        # Calculate metrics
        total_problems = len(dataset)
        self.results["overall_accuracy"] = (total_correct / total_problems) * 100

        for level, stats in level_stats.items():
            accuracy = (stats["correct"] / stats["total"]) * 100
            self.results["level_accuracies"][level] = {
                "accuracy": accuracy,
                "correct": stats["correct"],
                "total": stats["total"]
            }

        return self.results

    def save_results(self, output_dir: str = "results") -> None:
        """Save evaluation results."""
        os.makedirs(output_dir, exist_ok=True)  # Create the output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Clean the model name: replace / with - and remove any other invalid chars
        cleaned_model_name = re.sub(r'[\\/*?:"<>|]', "-", self.model.model_name)
        filename = os.path.join(output_dir, f"mazebench_{cleaned_model_name}_{timestamp}.json")

        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=4)  # Save the results as JSON

        logging.info(f"Results saved to {filename}")