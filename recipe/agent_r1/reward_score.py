# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import ast

import difflib


# Threshold for determining if two strings are equal using
# difflib.SequenceMatcher(...).ratio().
_MIN_DIFF_SIMILARITY = 0.9


def fuzzy_match(text1: str, text2: str, ignore_case: bool = True) -> bool:
  """Compares two strings.

  Args:
    text1: The first text.
    text2: The second text.
    ignore_case: Whether to ignore case during comparison.

  Returns:
    Whether the two strings are approximately equal.
  """
  if text1 is None or text2 is None:
    return False
  text1 = str(text1)
  text2 = str(text2)

  def text_similarity(text1: str, text2: str, ignore_case: bool) -> float:
    """Computes similiarity between two texts."""
    if ignore_case:
      text1 = text1.lower()
      text2 = text2.lower()

    return difflib.SequenceMatcher(None, text1, text2).ratio()
  return (
      text_similarity(text1, text2, ignore_case=ignore_case)
      >= _MIN_DIFF_SIMILARITY
  )


def extract_reasoning_components(text):
    # Pattern to match component headings and their content
    # Captures: 1. Component name (Task, Observation, etc.)
    #           2. The content that follows until the next component
    task_start, obs_start, progress_start, exception_start, decision_start, prediction_start = text.find("1. Task"), text.find("2. Observation"), text.find("3. Progress"), text.find("4. Exception"), text.find("5. Decision"), text.find("6. Prediction")

    if task_start == -1 or obs_start == -1 or progress_start == -1 or exception_start == -1 or decision_start == -1 or prediction_start == -1 or (not task_start < obs_start < progress_start < exception_start < decision_start < prediction_start):
        return None
    
    task = text[text.find(':', task_start)+1:obs_start].strip()
    obs = text[text.find(':', obs_start)+1:progress_start].strip()
    progress = text[text.find(':', progress_start)+1:exception_start].strip()
    exception = text[text.find(':', exception_start)+1:decision_start].strip()
    decision = text[text.find(':', decision_start)+1:prediction_start].strip()
    prediction = text[text.find(':', prediction_start)+1:].strip()

    if len(task.split(" ")) < 6 or len(obs.split(" ")) < 6 or len(progress.split(" ")) < 6 or len(exception.split(" ")) < 6 or len(decision.split(" ")) < 6 or len(prediction.split(" ")) < 6:
        return None
    
    return {
        "task": task,
        "obs": obs,
        "progress": progress,
        "exception": exception,
        "decision": decision,
        "prediction": prediction
    }


"""
<thinking>
1. Task: The goal is to create a playlist titled "Road Trips Essentials" in VLC using the specified video files located in the "VLCVideos" folder within the internal memory. The user needs to navigate to the "VLCVideos" folder to access the required video files.
2. Observation: The current screen displays the contents of the "Internal memory" directory. The "VLCVideos" folder is visible at the bottom of the list, indicating that it is the target folder for accessing the video files needed for the playlist. The folder icons and names are clearly labeled, and the "VLCVideos" folder is identifiable.
3. Progress: The user has successfully navigated to the "Internal memory" directory, which is the correct location to access the "VLCVideos" folder. The next step is to open the "VLCVideos" folder to view its contents and select the required video files.
4. Exception: No exceptions are detected. The user is on the correct path, and the "VLCVideos" folder is visible and accessible.
5. Decision: The next step is to click on the "VLCVideos" folder to open it and view its contents. This action will allow the user to proceed with selecting the required video files for the playlist.
6. Prediction: After clicking on the "VLCVideos" folder, the screen will display the contents of the folder, including the video files "footage_78_export_2023_02_08.mp4" and "aXM7_moment_88_HD.mp4". The user will then be able to select these files to add them to the playlist.
</thinking>
"""
def detect_protocol(reasoning_str):
    """
    Detect if the reasoning is using the protocol.
    """
    components = extract_reasoning_components(reasoning_str)

    return components

REASON_ACTION_PATTERN = re.compile(r'<thinking>\s*(.*?)\s*</thinking>\s*<action>\s*(.*?)\s*</action>', re.DOTALL) # Can handle strings without line-breaks
def extract_reasoning_action(solution_str, use_protocol: bool = False):
    """
    E.g., <thinking>\n...\n</thinking>\n<action>\n...\n</action>
    """
    if solution_str is None:
        return None, None
        
    reasoning_action = REASON_ACTION_PATTERN.fullmatch(solution_str)
    if reasoning_action is None:
        return None, None
    reasoning = reasoning_action.group(1)
    action_str = reasoning_action.group(2)

    if reasoning.count('"action"') != 0 or action_str.count('"action"') != 1:
        return None, None

    if use_protocol:
        protocol = detect_protocol(reasoning)
        if protocol is None:
            return None, None

    try:
        action = ast.literal_eval(action_str)
        # Ensure action is a dictionary
        if not isinstance(action, dict):
            return reasoning, None
    except Exception:
        return reasoning, None

    return reasoning, action

def get_swipe_direction(start: list[int], end: list[int]):
    """
    Get the direction of the swipe. The top-left corner is the origin. X points towards to the right. Y points towards to the bottom.
    """
    if start[0] < end[0] and (start[1] - end[1]) <= 50:
        return 'right'
    elif start[0] > end[0] and (start[1] - end[1]) <= 50:
        return 'left'
    elif start[1] < end[1] and (start[0] - end[0]) <= 50:
        return 'down'
    elif start[1] > end[1] and (start[0] - end[0]) <= 50:
        return 'up'

def match_action_params(action_pred, action_gt, reward_attr):
    """
    E.g., {'action': 'click', 'coordinate': [0, 0]}
    """
    reward = 1.0

    if set(action_pred.keys()) != set(action_gt.keys()):
        reward = 0.0
    else:
        if action_pred['action'] in ['click', 'long_press', 'type']:
            # Check if coordinate exists
            if 'coordinate' not in action_pred or 'coordinate' not in action_gt:
                reward = 0.0
            else:
                x_pred, y_pred = action_pred['coordinate']
                
                # Check if bbox exists in reward_attr
                if isinstance(reward_attr, dict) and 'bbox' in reward_attr:
                    x1, y1, x2, y2 = reward_attr['bbox']
                    if not (x1 <= x_pred <= x2 and y1 <= y_pred <= y2):
                        reward = 0.0
                else:
                    # If bbox is missing, we can't validate the coordinate
                    print(f"Warning: bbox missing in reward_attr for action {action_pred['action']}")
        
        if action_pred['action'] in ['type', 'answer']:
            # Check if text exists
            if 'text' not in action_pred or 'text' not in action_gt:
                reward = 0.0
            else:
                text_pred, text_gt = action_pred['text'], action_gt['text']
                # Use fuzzy match to compute the reward for text similarity
                reward = float(fuzzy_match(text_gt, text_pred))

        elif action_pred['action'] in ['open', 'open_app']:
            # Check if text exists
            if 'text' not in action_pred or 'text' not in action_gt:
                reward = 0.0
            else:
                app_pred, app_gt = action_pred['text'], action_gt['text']
                reward = float(app_pred == app_gt)
        
        elif action_pred['action'] == 'swipe':
            # Check if start and end exist
            start_pred, end_pred = action_pred['coordinate'], action_pred['coordinate2']
            start_gt, end_gt = action_gt['coordinate'], action_gt['coordinate2']
            
            # Check if the swiping direction is correct
            pred_direction = get_swipe_direction(start_pred, end_pred)
            gt_direction = get_swipe_direction(start_gt, end_gt)
            
            if pred_direction is None or gt_direction is None:
                reward = 0.0
            else:
                reward = float(pred_direction == gt_direction)

        elif action_pred['action'] == 'terminate':
            # Check if status exists
            if 'status' not in action_pred or 'status' not in action_gt:
                reward = 0.0
            else:
                status_pred, status_gt = action_pred['status'], action_gt['status']
                reward = float(status_pred == status_gt)
        
        elif action_pred['action'] == 'system_button':
            # Check if button exists
            if 'button' not in action_pred or 'button' not in action_gt:
                reward = 0.0
            else:
                button_pred, button_gt = action_pred['button'], action_gt['button']
                reward = float(button_pred == button_gt)

    return reward

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    # 
    reasoning_pred, action_pred = extract_reasoning_action(solution_str, use_protocol=True)
    reasoning_gt, action_gt = extract_reasoning_action(ground_truth)

    # Format reward
    format_reward  = 1.0 if action_pred is not None and action_gt is not None else 0.0
    type_reward = 0.0
    act_params_reward = 0.0
    # Action type reward: The action type reward is computed by comparing the predicted action type T′ with the ground truth action type T . It assigns a reward of 1 if T′ = T and 0 otherwise
    if format_reward == 1.0:
        type_reward = 1.0 if action_pred.get('action', '1') == action_gt.get('action', '2') else 0.0

        if type_reward == 1.0:
            # Check if reward_attr exists in extra_info
            reward_attr = extra_info.get('reward_attr', {}) if extra_info else {}
            
            act_params_reward = match_action_params(action_pred, action_gt, reward_attr)

    total_reward = format_reward + type_reward + act_params_reward
    
    if total_reward == 0:
        1+1
    elif total_reward == 1:
        1+1
    elif total_reward == 2:
        1+1
    elif total_reward == 3:
        1+1
        
    # Return detailed breakdown for debugging
    return {
        'score': total_reward, # must have this key
        'format_reward': format_reward,
        'type_reward': type_reward,
        'act_params_reward': act_params_reward,
        # 'solution_str': solution_str,
        # 'ground_truth': ground_truth
    }

def permute_thoughts_components(reasoning_str):
    """
    Permute the components of the reasoning.
    """
    reasoning_action = REASON_ACTION_PATTERN.fullmatch(solution_str)
    if reasoning_action is None:
        return None, None
    reasoning = reasoning_action.group(1)
    action_str = reasoning_action.group(2)

    components = extract_reasoning_components(reasoning)

    key_values = list(components.items())
    random.shuffle(key_values)
    permuted_reasoning = '\n'.join(f"{i+1}. {k.title()}: {v}" for i, (k, v) in enumerate(key_values))

    output = f"<thinking>\n{permuted_reasoning}\n</thinking>\n<action>\n{action_str}\n</action>"
    return output

if __name__ == '__main__':
    # test the reward functions using test set
    import pandas as pd
    from tqdm import tqdm
    import numpy as np
    import json
    import os
    import random

    TEST_RANDOM = False
    PERMUTE_THOUGHTS = True

    # File path to test data
    file_path = '/mnt/jfs/copilot/lhx/ui_data/AndroidControl/0414_AW/test.parquet'

    # Load test data
    print(f"Loading test data from {file_path}")
    data = pd.read_parquet(file_path)

    # Debug: Display column names to understand the data structure
    print(f"Data columns: {data.columns.tolist()}")
    if 'reward_model' in data.columns:
        print("Sample reward_model structure:", data['reward_model'].iloc[0] if len(data) > 0 else "No data")

    results = []
    total_samples = len(data)
    failed_samples = 0
    reward_stats = {
        'total': [],
        'format': [],
        'type': [],
        'params': []
    }
    
    # Group actions by type for detailed analysis
    action_type_stats = {}
    
    print(f"Processing {total_samples} test samples...")
    history = []
    for i, item in enumerate(tqdm(data.itertuples(), total=total_samples, desc='Computing reward')):
        try:
            # Extract necessary data
            data_source = getattr(item, 'data_source', 'unknown')
            extra_info = getattr(item, 'extra_info', {})
            
            # Get the reward model dictionary
            reward_model = None
            if hasattr(item, 'reward_model'):
                reward_model = item.reward_model
            elif 'reward_model' in data.columns:
                reward_model = data.iloc[i]['reward_model']
                
            # Extract solution and ground truth
            ground_truth = reward_model['ground_truth']
            solution_str = random.choice(history) if len(history) > 0 and TEST_RANDOM else  reward_model['ground_truth'] 

            if PERMUTE_THOUGHTS:
                solution_str = permute_thoughts_components(solution_str)

            history.append(reward_model['ground_truth'])
            # Compute reward
            reward_details = compute_score(data_source, solution_str, ground_truth, extra_info)
            
            # Track stats
            reward_stats['total'].append(reward_details['score'])
            reward_stats['format'].append(reward_details['format_reward'])
            reward_stats['type'].append(reward_details['type_reward'])
            reward_stats['params'].append(reward_details['act_params_reward'])
            
            # Track by action type
            if reward_details.get('action_gt', None) is not None:
                action_type = reward_details['action_gt']['action']
                if action_type not in action_type_stats:
                    action_type_stats[action_type] = {'count': 0, 'success': 0, 'rewards': []}
                
                action_type_stats[action_type]['count'] += 1
                if reward_details['total_reward'] >= 2.5:  # Successful if got most of the points
                    action_type_stats[action_type]['success'] += 1
                action_type_stats[action_type]['rewards'].append(reward_details['total_reward'])
            
            # Add detailed result for this sample
            results.append({
                'index': i,
                'rewards': reward_details,
                'solution': solution_str,
                'ground_truth': ground_truth
            })
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            failed_samples += 1
    
    # Calculate and print overall statistics
    print("\n==== Reward Function Debugging Results ====")
    print(f"Total samples: {total_samples}")
    print(f"Failed to process: {failed_samples}")
    
    if reward_stats['total']:
        print("\n==== Overall Reward Statistics ====")
        print(f"Average total reward: {np.mean(reward_stats['total']):.4f}")
        print(f"Average format reward: {np.mean(reward_stats['format']):.4f}")
        print(f"Average type reward: {np.mean(reward_stats['type']):.4f}")
        print(f"Average params reward: {np.mean(reward_stats['params']):.4f}")
    
    # Print per action type statistics
    print("\n==== Reward by Action Type ====")
    for action_type, stats in action_type_stats.items():
        success_rate = stats['success'] / stats['count'] if stats['count'] > 0 else 0
        avg_reward = np.mean(stats['rewards']) if stats['rewards'] else 0
        
        print(f"Action: {action_type}")
        print(f"  Count: {stats['count']}")
        print(f"  Success rate: {success_rate:.2%}")
        print(f"  Average reward: {avg_reward:.4f}")
    
    # Save detailed results to JSON for further analysis
    output_dir = os.path.dirname(file_path)
    output_file = os.path.join(output_dir, 'reward_debug_results.json')
    
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'total_samples': total_samples,
                'failed_samples': failed_samples,
                'average_rewards': {
                    'total': float(np.mean(reward_stats['total'])) if reward_stats['total'] else 0,
                    'format': float(np.mean(reward_stats['format'])) if reward_stats['format'] else 0,
                    'type': float(np.mean(reward_stats['type'])) if reward_stats['type'] else 0,
                    'params': float(np.mean(reward_stats['params'])) if reward_stats['params'] else 0
                },
                'action_type_stats': action_type_stats
            },
            'detailed_results': results[:100]  # Limiting to first 100 for file size
        }, f, indent=2)
    
    print(f"\nDetailed results saved to {output_file}")
