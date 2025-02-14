import random
from utils import (
    generate_maze_tokens_cot,
    generate_wrong_path_order_1,
    generate_wrong_path_order_2,
    count_walls,
    process_input,
)
from datasets import load_from_disk, load_dataset
ds = load_dataset("homebrewltd/Maze-Reasoning-v0.1", split='train')
ds = ds.remove_columns(['conversations'])
# The number of walls around original point is 2.
def filter_orginal(example):
    adj_list_str, origin, _, _ = process_input(example['raw_token'])
    num_wall = count_walls(adj_list_str, origin)
    return num_wall==2

ds_1 = ds.filter(filter_orginal, num_proc=32)
def create_cot_response(batch):
    Cot_response_data = []
    for raw_tokens, correct_cot_response in zip(batch["raw_token"], batch["Cot_Response"]):
        adj_list_str, origin_str, target_str, path_str = process_input(raw_tokens)
        for n_wrong_steps in range(3, 0, -1):
            wrong_path = generate_wrong_path_order_1(adj_list_str, origin_str, path_str, n_wrong_steps)
            if wrong_path:
                break  
        _, cot_steps, instructions, _ = generate_maze_tokens_cot(adj_list_str, origin_str, target_str, wrong_path)
        wrong_cot_response = ""
        for i, step in enumerate(cot_steps):
            if i == len(cot_steps)-1:
                wrong_cot_response += f"Step {i+1}: {instructions[i]}\n{step.strip()}"
            else: 
                wrong_cot_response += f"Step {i+1}: {instructions[i]}\n{step.strip()}\n\n"

        wall_hit_count = count_walls(adj_list_str, wrong_path.split()[-1])
        if wall_hit_count != 3:
            reset_message = "Hit a dead end. Let's reset to the original point and find another way."
        else:
            reset_message = "Heading in the wrong direction. Let's reset to the origin and try a different path."
        cot_response = f"{wrong_cot_response}\n{reset_message}\nRESET\n{correct_cot_response}"
        Cot_response_data.append(cot_response)
    return {'Cot_Response': Cot_response_data}
def create_conversations(batch):
    conversations = []
    for text, cot, answer in zip(batch["Prompt"], batch["Cot_Response"], batch['Response']):
        final_answer = f"<think>\n{cot}\n</think>\n\n{answer}"
        user_part = {"content": text.strip(), "role": "user"}
        assistant_part = {"content": final_answer, "role": "assistant"}
        conversation = [user_part, assistant_part]
        conversations.append(conversation)
    return {"conversations": conversations}
ds_1 = ds_1.map(create_cot_response, batched=True, batch_size=100, num_proc=32)
ds_1 = ds_1.map(create_conversations, batched=True, num_proc=32)
ds_1.push_to_hub("homebrewltd/Maze-Reasoning-Reset-v0.1", "order-2", split="train")

        
        

# The number of walls around original point is 1.
def filter_orginal(example):
    adj_list_str, origin, _, _ = process_input(example['raw_token'])
    num_wall = count_walls(adj_list_str, origin)
    return num_wall==1

ds_2 = ds.filter(filter_orginal, num_proc=32)
def create_cot_response(batch):
    Cot_response_data = []
    for raw_tokens, correct_cot_response in zip(batch["raw_token"], batch["Cot_Response"]):
        adj_list_str, origin_str, target_str, path_str = process_input(raw_tokens)
        wrong_paths = generate_wrong_path_order_2(adj_list_str, origin_str, path_str, max_n_steps=3)
        wrong_cot_response = ""
        for wrong_path in wrong_paths:
            wrong_response = ""
            _, cot_steps, instructions, _ = generate_maze_tokens_cot(adj_list_str, origin_str, target_str, wrong_path)
            for i, step in enumerate(cot_steps):
                if i == len(cot_steps)-1:
                    wrong_response += f"Step {i+1}: {instructions[i]}\n{step.strip()}"
                else: 
                    wrong_response += f"Step {i+1}: {instructions[i]}\n{step.strip()}\n\n"

            wall_hit_count = count_walls(adj_list_str, wrong_path.split()[-1])
            if wall_hit_count != 3:
                reset_message = "Hit a dead end."
            else:
                reset_message = "Heading in the wrong direction."
            wrong_cot_response += f"{wrong_response}\n{reset_message}\nRESET\n"
        cot_response = f"{wrong_cot_response}{correct_cot_response}"
        Cot_response_data.append(cot_response)
    return {'Cot_Response': Cot_response_data}
def create_conversations(batch):
    conversations = []
    for text, cot, answer in zip(batch["Prompt"], batch["Cot_Response"], batch['Response']):
        final_answer = f"<think>\n{cot}\n</think>\n\n{answer}"
        user_part = {"content": text.strip(), "role": "user"}
        assistant_part = {"content": final_answer, "role": "assistant"}
        conversation = [user_part, assistant_part]
        conversations.append(conversation)
    return {"conversations": conversations}
ds_2 = ds_2.map(create_cot_response, batched=True, batch_size=100, num_proc=32)
ds_2 = ds_2.map(create_conversations, batched=True, num_proc=32)
ds_2.push_to_hub("homebrewltd/Maze-Reasoning-Reset-v0.1", "order-1", split="train")