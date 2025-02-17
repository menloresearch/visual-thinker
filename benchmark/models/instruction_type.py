# Prompt for AlphaMaze
ALPHA_MAZE_PROMPT="""You are a helpful assistant that solves mazes.  You will be given a maze represented by a series of tokens.  
The tokens represent:
- Coordinates: <|row-col|> (e.g., <|0-0|>, <|2-4|>)
- Walls:  <|no_wall|>, <|up_wall|>, <|down_wall|>, <|left_wall|>, <|right_wall|>, <|up_down_wall|>, etc.
- Origin: <|origin|>
- Target: <|target|>
- Movement: <|up|>, <|down|>, <|left|>, <|right|>, <|blank|>

Your task is to output the sequence of movements (<|up|>, <|down|>, <|left|>, <|right|>) required to navigate from the origin to the target, based on the provided maze representation.  Think step by step. At each step, predict only the next movement token. Output only the move tokens, separated by spaces.
MAZE:
{maze_prompt}"""

prompt_templates = {
    "default" : ALPHA_MAZE_PROMPT,
    "alphamaze" : ALPHA_MAZE_PROMPT,
}