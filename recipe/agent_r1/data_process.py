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
"""
Preprocess the GSM8k dataset to parquet format
"""

import json, base64
import os
import pandas as pd
from verl.utils.hdfs_io import copy, makedirs
import argparse
from PIL import Image
import io

def extract_solution(solution_str):
    solution = solution_str.split("<action>")[1].split("</action>")[0].strip()
    assert solution is not None
    return solution


def make_map_fn(split):
    def process_fn(example, idx):
        messages = example.pop('messages')
        question = messages[0]['content']
        answer_raw = messages[-1]['content'].replace('\n\n','\n')
        
        # Convert PIL Image to bytes for serialization
        img = Image.open(example['images'][0]).convert('RGB') # the images field should be a list of {"bytes": bytes (e.g., b'\x89PNG\r\n\x1a\n\x00\x00'), "path": None}
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_base64 = f'data:image/jpeg;base64,{base64.b64encode(img_buffer.getvalue()).decode("utf-8")}'
        data = {
            "data_source": data_source,
            "prompt": [{
                    "role": "user",
                    "content": question,
                }],
            "images": [img_base64], #[{"bytes": img_bytes, "path": None}],  # Store as bytes instead of PIL Image
            "ability": "agent",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer_raw
            },
            "extra_info": {
                'split': split,
                'index': idx,
                'reward_attr': example.get('reward_attr', None),
                "question": question,
            }
        }
        return data
    return process_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/mnt/jfs/copilot/lhx/ui_data/AndroidControl/0414_AW/')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()
    data_source = 'HongxinLi/0414_AW_v1'

    # ROOT
    ROOT = args.local_dir

    # Load JSON data
    with open(os.path.join(ROOT, 'train.json')) as f:
        train_data = json.load(f)
        print(f"Loading {len(train_data)} train samples from {ROOT}/train.json")
    with open(os.path.join(ROOT, 'test.json')) as f:
        test_data = json.load(f)
        print(f"Loading {len(test_data)} test samples from {ROOT}/test.json")

    # Process data
    train_processed = [make_map_fn('train')(example, idx) for idx, example in enumerate(train_data)]
    test_processed = [make_map_fn('test')(example, idx) for idx, example in enumerate(test_data)]

    # Convert to DataFrame
    train_df = pd.DataFrame(train_processed)
    test_df = pd.DataFrame(test_processed)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Ensure directory exists
    os.makedirs(local_dir, exist_ok=True)

    # Save to parquet
    train_df.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_df.to_parquet(os.path.join(local_dir, 'test.parquet'))

    # Debug: Load and print parquet file to verify content
    print("Debugging: Loading saved parquet files to verify content")
    try:
        debug_train_df = pd.read_parquet(os.path.join(local_dir, 'train.parquet'))
        debug_test_df = pd.read_parquet(os.path.join(local_dir, 'test.parquet'))
        
        print(f"Train dataset shape: {debug_train_df.shape}")
        print(f"Test dataset shape: {debug_test_df.shape}")
        
        # Print sample data (first row)
        if not debug_train_df.empty:
            print("\nSample from train dataset:")
            print(debug_train_df.iloc[0])
        
        if not debug_test_df.empty:
            print("\nSample from test dataset:")
            print(debug_test_df.iloc[0])
    except Exception as e:
        print(f"Error loading parquet files for debugging: {e}")
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
