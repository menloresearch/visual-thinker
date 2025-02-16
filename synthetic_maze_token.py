import argparse
import itertools
import os
from typing import Callable, Optional
import numpy as np
import pyarrow as pa
from pathlib import Path
import pyarrow.parquet as pq
from maze_dataset import MazeDataset, MazeDatasetConfig
from maze_dataset.generation import LatticeMazeGenerators
from maze_dataset.plotting import MazePlot
from maze_dataset.tokenization import MazeTokenizer, TokenizationMode
from maze_dataset.maze import SolvedMaze
import matplotlib.pyplot as plt
import io
from tqdm import tqdm
from PIL import Image 
from utils import generate_maze_tokens_cot, process_input
from datasets import Dataset

def generate_mazes(
    min_length: int,
    max_length: int,
    num_mazes: int,
    grid_n: int,
    output_path: str,
    num_selected: Optional[int] = None,
    gen_parallel: bool = True
) -> None:
    """
    Generate maze dataset with configurable parameters.
    
    Args:
        min_length: Minimum path length for maze solutions
        max_length: Maximum path length for maze solutions
        num_mazes: Number of mazes to generate
        grid_n: Size of the maze grid (grid_n x grid_n)
        output_path: Path to save the generated dataset
        gen_parallel: Whether to use parallel generation
    """
    
    prompt_template = """You are a helpful assistant that solves mazes.  You will be given a maze represented by a series of tokens.  
    The tokens represent:
    - Coordinates: <|row-col|> (e.g., <|0-0|>, <|2-4|>)
    - Walls:  <|no_wall|>, <|up_wall|>, <|down_wall|>, <|left_wall|>, <|right_wall|>, <|up_down_wall|>, etc.
    - Origin: <|origin|>
    - Target: <|target|>
    - Movement: <|up|>, <|down|>, <|left|>, <|right|>, <|blank|>

    Your task is to output the sequence of movements (<|up|>, <|down|>, <|left|>, <|right|>) required to navigate from the origin to the target, based on the provided maze representation.  Think step by step. At each step, predict only the next movement token. Output only the move tokens, separated by spaces.
    MAZE:
    {maze_prompt}"""

    # Generate dataset of solved mazes
    cfg = MazeDatasetConfig(
        name=f"reasoning_data_{grid_n}x{grid_n}",
        grid_n=grid_n,
        n_mazes=num_mazes,
        maze_ctor=LatticeMazeGenerators.gen_dfs,
    )

    dataset = MazeDataset.from_config(
        cfg,
        do_download=False,
        load_local=False,
        do_generate=True,
        save_local=False,
        verbose=True,
        gen_parallel=gen_parallel,
    )
    
    # Apply length filters
    dataset = dataset.filter_by.path_length(min_length=min_length)
    dataset = dataset.filter_by.max_path_length(max_length=max_length)

    tokenizer = MazeTokenizer(
        tokenization_mode=TokenizationMode.AOTP_UT_rasterized,
        max_grid_size=grid_n
    )

    # Process mazes
    images_data = []
    prompts_data = []
    cot_responses_data = []
    responses_data = []
    raw_token_data = []
    
    for maze in tqdm(dataset, desc="Processing Mazes"):
        maze_tok = maze.as_tokens(maze_tokenizer=tokenizer)
        raw_tokens = " ".join(maze_tok)
        
        adj_list_str, origin_str, target_str, path_str = process_input(raw_tokens)
        prompt, cot_steps, instructions, golden_answer = generate_maze_tokens_cot(
            adj_list_str, origin_str, target_str, path_str
        )
        prompt = prompt_template.format(maze_prompt=prompt.strip())
        
        cot_response = ""
        for i, step in enumerate(cot_steps):
            if i == len(cot_steps)-1:
                cot_response += f"Step {i+1}: {instructions[i]}\n{step.strip()}"
            else: 
                cot_response += f"Step {i+1}: {instructions[i]}\n{step.strip()}\n\n"
                
        solved_mazed = SolvedMaze.from_tokens(raw_tokens, tokenizer)
        fig = MazePlot(solved_mazed).plot()

        buf = io.BytesIO()   
        plt.savefig(buf, format="png")
        plt.close() 
        buf.seek(0)
        image = Image.open(buf)

        images_data.append(image) 
        prompts_data.append(prompt)
        cot_responses_data.append(cot_response)
        responses_data.append(golden_answer)
        raw_token_data.append(raw_tokens)

    # Create and process dataset
    data = {
        "Maze_image": images_data,
        "Prompt": prompts_data,
        "Cot_Response": cot_responses_data,
        "Response": responses_data,
        # "raw_token": raw_token_data,
    }
    
    dataset = Dataset.from_dict(data)
    df = dataset.to_pandas()
    unique_indices = ~df['Response'].duplicated(keep='first')
    deduped_df = df[unique_indices]
    deduped_dataset = Dataset.from_pandas(deduped_df)

    if num_selected is not None and num_selected > 0:
        if num_selected > len(deduped_dataset):
            print(f"Warning: requested {num_selected} examples but only {len(deduped_dataset)} are available")
            num_selected = len(deduped_dataset)
        deduped_dataset = deduped_dataset.select(range(num_selected))
        print(f"Selected {num_selected} examples from the dataset")

    # Print statistics
    total_rows = len(df)
    unique_rows = len(deduped_df)
    duplicates = total_rows - unique_rows

    print(f"Total rows before deduplication: {total_rows}")
    print(f"Rows after deduplication: {unique_rows}")
    print(f"Duplicates removed: {duplicates}")

    # Save dataset
    deduped_dataset.save_to_disk(output_path)

def main():
    parser = argparse.ArgumentParser(description="Generate maze dataset with configurable parameters")
    parser.add_argument("--min-length", type=int, required=True, help="Minimum path length")
    parser.add_argument("--max-length", type=int, required=True, help="Maximum path length")
    parser.add_argument("--num-mazes", type=int, required=True, help="Number of mazes to generate")
    parser.add_argument("--grid-n", type=int, required=True, help="Size of maze grid (grid_n x grid_n)")
    parser.add_argument("--output", type=str, required=True, help="Output path for dataset")
    parser.add_argument("--num-selected", type=int, help="Number of examples to select from final dataset")
    parser.add_argument("--no-parallel", action="store_false", dest="gen_parallel", 
                      help="Disable parallel generation")
    
    args = parser.parse_args()
    
    generate_mazes(
        min_length=args.min_length,
        max_length=args.max_length,
        num_mazes=args.num_mazes,
        grid_n=args.grid_n,
        output_path=args.output,
        num_selected=args.num_selected,
        gen_parallel=args.gen_parallel
    )

if __name__ == "__main__":
    main()