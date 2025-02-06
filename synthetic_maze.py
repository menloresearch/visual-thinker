import itertools
import os
from typing import Callable

import numpy as np
import pyarrow as pa
from muutils.misc import shorten_numerical_to_str
from zanj import ZANJ
from pathlib import Path

from maze_dataset import MazeDataset, MazeDatasetConfig
from maze_dataset.generation import LatticeMazeGenerators
from maze_dataset.plotting import MazePlot
from maze_dataset.tokenization import MazeTokenizer, TokenizationMode
from maze_dataset.maze import SolvedMaze  # Import SolvedMaze
import matplotlib.pyplot as plt
from utils import get_direction, convert_path_to_directions

def generate_and_save_reasoning_data(
    n_mazes: int,
    grid_n: int,
    output_path: str,
    tokenizer: MazeTokenizer | None = None,
    save_format: str = "arrow",
    gen_parallel: bool = False,
):
    """Generates reasoning data from mazes and saves it to disk.

    Args:
        n_mazes (int): Number of mazes to generate.
        grid_n (int): The grid size for the maze (grid_n x grid_n).
        output_path (str): The path to the directory where data will be saved.
        tokenizer (MazeTokenizer, optional): Tokenizer to use for generating string representations of paths.
            Defaults to None, which uses a new default MazeTokenizer.
        save_format (str, optional): Format to save the dataset in.
            Options: "arrow", "numpy". Defaults to "arrow".
        gen_parallel (bool, optional): Whether to use multiprocessing to generate the mazes in parallel.
    """

    if tokenizer is None:
        tokenizer = MazeTokenizer(
            tokenization_mode=TokenizationMode.AOTP_UT_rasterized, max_grid_size=grid_n
        )

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

    # Set up a directory for saving images
    image_dir = Path(output_path) / "images"
    os.makedirs(image_dir, exist_ok=True)

    images_dataset = []
    instructions= []

    for maze in dataset:
        tokens = maze.as_tokens(maze_tokenizer=tokenizer)
        print(tokens)
        path_start: str = "<PATH_START>"
        path_end: str = "<PATH_END>"
        path_tokens = tokens[
            tokens.index(path_start) + 1 : tokens.index(path_end)
        ]
        

        maze_images = []
        maze_instructions = convert_path_to_directions(path_tokens)
        print(maze_instructions)
        for i in range(2, len(path_tokens)+1):
            partial_solution_tokens = path_tokens[:i]
            # Reconstruct the maze with the partial solution
            partial_maze = SolvedMaze.from_tokens(
                tokens[: tokens.index(path_start) + 1]
                + partial_solution_tokens
                + [path_end],
                tokenizer,
            )

            # Create and save the plot
            fig, ax = plt.subplots(figsize=(5, 5))  # Create a new figure and axes
            MazePlot(partial_maze).plot(fig_ax=(fig, ax), title=False)
            img_name = f"maze_3x3_step_{i-1}.png"
            img_path = os.path.join(image_dir, img_name)
            plt.savefig(img_path)
            plt.close(fig)

            maze_images.append(img_name)  # Save only the image name

            # Generate text instruction (replace this with actual instruction generation logic)

        images_dataset.append(maze_images)
        instructions.append(maze_instructions)

    # Save the data
    # if save_format == "arrow":

    #     maze_images_col = pa.array(images_dataset)
    #     instructions_col = pa.array(instructions)

    #     table = pa.Table.from_arrays(
    #         [maze_images_col, instructions_col], names=["images", "instructions"]
    #     )

    #     with pa.OSFile(output_path, "wb") as sink:
    #         with pa.RecordBatchFileWriter(sink, table.schema) as writer:
    #             writer.write_table(table)

    # elif save_format == "numpy":
    #     np.savez_compressed(
    #         output_path,
    #         maze_images=np.array(images_dataset, dtype=object),
    #         instructions=np.array(instructions, dtype=object),
    #     )
    # else:
    #     raise ValueError(
    #         f"Invalid save format '{save_format}'. Must be one of 'arrow', 'numpy'."
    #     )

    # print(f"Reasoning dataset saved to {output_path}")

generate_and_save_reasoning_data(
    n_mazes=1,
    grid_n=5,
    output_path="reasoning_data.arrow",
    gen_parallel=False,  # Set to True to use multiprocessing
)