import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Helper functions ---
def _extract_solution(text: str) -> str:
    """
    Extract the solution from the generated text.
    Cleans up the response to extract only the sequence of moves.
    
    Args:
        text (str): Raw generated text
        
    Returns:
        str: Extracted solution
    """
    # Find sequences like <|up|>, <|down|>, etc.
    import re
    move_pattern = re.compile(r'<\|(up|down|left|right)\|>')
    moves = move_pattern.findall(text)
    
    # If no moves found, return the original text
    if not moves:
        return text
    
    # Reconstruct the proper format
    solution = "".join([f"<|{move}|>" for move in moves])
    return solution
def extract_answer(text: str) -> str:
    # Parse answer from LLM with format <think> </think> {final_answer}
    try: 
        answer = text.split("</think>")[1]
        return answer.strip()
    except:
        return ""
def parse_token(token):
    """
    Given a token like "<|up_left_right_wall|>", returns the inner text.
    """
    return token.strip()[2:-2]

def parse_maze(maze_text):
    """
    Parses a maze (a multiline string) into a grid dictionary.
    """
    grid = {}
    lines = [line for line in maze_text.strip().splitlines() if line.strip()]
    token_pattern = re.compile(r"<\|([^|]+)\|>")
    
    for line in lines:
        tokens = token_pattern.findall(line)
        ncols = len(tokens) // 3
        for i in range(ncols):
            row, col = map(int, tokens[3*i].split('-'))
            walls = set(tokens[3*i + 1].replace('_wall', '').split('_')) if tokens[3*i + 1] != "no_wall" else set()
            grid[(row, col)] = {"walls": walls, "marker": tokens[3*i + 2]}
    
    if grid:
        nrows, ncols = max(r for r, c in grid) + 1, max(c for r, c in grid) + 1
    else:
        nrows = ncols = 0
    
    return grid, nrows, ncols

def move_delta(direction):
    """Return (dr, dc) for a given direction."""
    return {
        "up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)
    }.get(direction, (0, 0))

def simulate_solution(grid, start, candidate_moves, nrows, ncols):
    """
    Simulates following the candidate_moves from the start coordinate.
    """
    current = start
    for idx, move_token in enumerate(candidate_moves):
        direction = move_token 
        if direction in grid[current]["walls"]:
            return False, current, f"Move {idx+1} ({direction}) is blocked at {current}."
        dr, dc = move_delta(direction)
        new_r, new_c = current[0] + dr, current[1] + dc
        if not (0 <= new_r < nrows and 0 <= new_c < ncols):
            return False, current, f"Move {idx+1} ({direction}) goes out of bounds from {current}."
        current = (new_r, new_c)
    return True, current, ""

def benchmark_maze_solution(maze_text, candidate_solution_text):
    """
    Benchmarks the candidate solution on the maze and returns a boolean.
    """
    error_log = ""
    grid, nrows, ncols = parse_maze(maze_text)
    if candidate_solution_text=="":
        error_log = "Unable to parse solution. You may want to increase the max_tokens."
        logging.error("Unable to parse solution. You may want to increase the max_tokens.")
        return False, error_log
    
    origin = target = None
    for coord, info in grid.items():
        if info["marker"] == "origin":
            origin = coord
        if info["marker"] == "target":
            target = coord
    
    if origin is None:
        logging.error("Maze has no origin.")
        return False
    if target is None:
        logging.error("Maze has no target.")
        return False
    
    candidate_moves = re.findall(r"<\|(.*?)\|>?", candidate_solution_text)

    success, final_pos, error = simulate_solution(grid, origin, candidate_moves, nrows, ncols)
    
    if not success:
        logging.error("Candidate solution is incorrect.")
        logging.error(error)
        logging.info("Final position reached: %s", final_pos)
        return False, error
    elif final_pos == target:
        success_log = "Candidate solution is correct"
        logging.info("Candidate solution is correct. Reached target at %s", final_pos)
        return True, success_log
    else:
        error_log = f"Candidate solution did not reach the target. Ended at {final_pos} but target is at {target}"
        logging.warning("Candidate solution did not reach the target.")
        logging.info("Ended at %s but target is at %s", final_pos, target)
        return False, error_log

# --- Run the benchmark ---
if __name__ == "__main__":
    maze_text = r"""
    <|0-0|><|up_left_right_wall|><|blank|><|0-1|><|up_left_wall|><|blank|><|0-2|><|up_down_wall|><|blank|><|0-3|><|up_down_wall|><|blank|><|0-4|><|up_right_wall|><|blank|>
    <|1-0|><|down_left_wall|><|blank|><|1-1|><|down_right_wall|><|blank|><|1-2|><|up_left_wall|><|blank|><|1-3|><|up_down_wall|><|blank|><|1-4|><|down_right_wall|><|target|>
    <|2-0|><|up_left_wall|><|origin|><|2-1|><|up_right_wall|><|blank|><|2-2|><|left_right_wall|><|blank|><|2-3|><|up_left_wall|><|blank|><|2-4|><|up_right_wall|><|blank|>
    <|3-0|><|left_right_wall|><|blank|><|3-1|><|down_left_wall|><|blank|><|3-2|><|down_right_wall|><|blank|><|3-3|><|down_left_right_wall|><|blank|><|3-4|><|left_right_wall|><|blank|>
    <|4-0|><|down_left_wall|><|blank|><|4-1|><|up_down_wall|><|blank|><|4-2|><|up_down_wall|><|blank|><|4-3|><|up_down_wall|><|blank|><|4-4|><|down_right_wall|><|blank|>
    """
    candidate_solution_text = "<|right|><|down|><|right|><|right|><|up|><|up|><|right|><|left|><|right|><|right|>"
    
    result = benchmark_maze_solution(maze_text, candidate_solution_text)
    print(f"Final Result: {result}")