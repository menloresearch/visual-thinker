import itertools
import os
from typing import Callable

import numpy as np
import pyarrow as pa
from muutils.misc import shorten_numerical_to_str
from zanj import ZANJ
from pathlib import Path
import pyarrow.parquet as pq
from maze_dataset import MazeDataset, MazeDatasetConfig
from maze_dataset.generation import LatticeMazeGenerators
from maze_dataset.plotting import MazePlot
from maze_dataset.tokenization import MazeTokenizer, TokenizationMode
from maze_dataset.maze import SolvedMaze  # Import SolvedMaze
import matplotlib.pyplot as plt
import io
from tqdm import tqdm
from PIL import Image 
from utils import (
    generate_maze_tokens_cot,
    process_input,
)
from datasets import Dataset
file_name = "maze_data.parquet"
gen_parallel= True
grid_n = 5
n_mazes = 110000
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
# Generate a dataset of solved mazes
cfg = MazeDatasetConfig(
    name=f"reasoning_data_{grid_n}x{grid_n}",
    grid_n=grid_n,
    n_mazes=n_mazes,
    maze_ctor=LatticeMazeGenerators.gen_dfs,
)

dataset: MazeDataset = MazeDataset.from_config(
    cfg,
    do_download=False,
    load_local=False,
    do_generate=True,
    save_local=False,
    verbose=True,
    gen_parallel=gen_parallel,  # Use the passed value
    # zanj=ZANJ(verbose=False, no_cache=False),
)

dataset = dataset.filter_by.path_length(min_length=1)
dataset = dataset.filter_by.max_path_length(max_length=11)

tokenizer = MazeTokenizer(
        tokenization_mode=TokenizationMode.AOTP_UT_rasterized, max_grid_size=grid_n
    )
def pil_to_numpy(img: Image.Image) -> np.ndarray:
    return np.array(img)
images_data = []
prompts_data = []
cot_responses_data = []
responses_data = []
raw_token_data = []
for maze in tqdm(dataset, desc="Processing Mazes"):
    maze_tok = maze.as_tokens(maze_tokenizer=tokenizer)
    raw_tokens = " ".join(maze_tok)
    adj_list_str, origin_str, target_str, path_str = process_input(raw_tokens)
    prompt, cot_steps, instructions, golden_answer = generate_maze_tokens_cot(adj_list_str, origin_str, target_str, path_str)
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

    # img_byte_arr = io.BytesIO()
    # image.save(img_byte_arr, format='PNG')  
    # img_byte_arr = img_byte_arr.getvalue()

    images_data.append(image) 
    prompts_data.append(prompt)
    cot_responses_data.append(cot_response)
    responses_data.append(golden_answer)
    raw_token_data.append(raw_tokens)

data = {
    "Maze_image" : images_data,
    "Prompt" : prompts_data,
    "Cot_Response" : cot_responses_data,
    "Response" : responses_data,
    "raw_token": raw_token_data,
}
dataset = Dataset.from_dict(data)
df = dataset.to_pandas()
unique_indices = ~df['raw_token'].duplicated(keep='first')
deduped_df = df[unique_indices]
deduped_dataset = Dataset.from_pandas(deduped_df)
total_rows = len(df)
unique_rows = len(deduped_df)
duplicates = total_rows - unique_rows

print(f"Total rows before deduplication: {total_rows}")
print(f"Rows after deduplication: {unique_rows}")
print(f"Duplicates removed: {duplicates}")
deduped_dataset.save_to_disk("./maze_data_2/")
dataset.save_to_disk("./maze_data/")
dataset.push_to_hub("jan-hq/Maze-Reasoning", split="train")


        