# The following codes are based on OSWorld(https://github.com/xlang-ai/OSWorld/tree/main).
# For codes from mm_agents, please refer to the original OSWorld repository.

import base64
import json
import logging
import os
import re
import tempfile
import time
import xml.etree.ElementTree as ET
import io
from io import BytesIO
from typing import Dict, List
import yaml
# import backoff
import pandas as pd
from datetime import datetime
import copy
import random
# import google.generativeai as genai
# import openai
import requests
from PIL import Image
import argparse
from openai import AzureOpenAI, OpenAI



prompt = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 
## Output Format
```
Action: ...
```
## Action Space
{action_space}

## User Instruction
{instruction}
"""


Android_control_Action_Space5 = """
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
long_press(start_box='<|box_start|>(x1,y1)<|box_end|>', time='')
type(content='')
open_app(app_name='')
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
press_back()
wait() 
"""

def parse_action_android_control(action, factor, image_height, image_width):
    action_type = action["action_type"]
    if action_type == "scroll":

        action_str = f"scroll(direction='{action['direction']}')"
        
    elif action_type == "click":
        x = round(action["x"] / image_width * 1000)
        y = round(action["y"] / image_height * 1000)
        # print(action["x"], action["y"], x, y, image_height, image_width, type(image_height))
        action_str = f"click(start_box='({x},{y})')"
        
    elif action_type == "long_press":
        x = round(action["x"] / image_width * 1000)
        y = round(action["y"] / image_height * 1000)
        action_str = f"long_press(start_box='({x},{y})')"
        
    elif action_type == "navigate_back":  
        action_str = "press_back()"
    
    elif action_type == "input_text":
        
        action_str = f"type(content='{action['text']}')"
        
    elif action_type == "wait":
        
        action_str = "wait()"
    elif action_type == "open_app":
        action_str = f"open_app(app_name='{action['app_name']}')"

import math
import numpy as np

import httpx

logger = logging.getLogger("desktopenv.agent")


# Function to encode the image
def encode_image(image_content):
    return base64.b64encode(image_content).decode("utf-8")


def build_processor(processor_path: str) -> "ProcessorMixin":
    """
    Builds the processor.
    """
    return AutoProcessor.from_pretrained(processor_path, padding_side="right", trust_remote_code=True)

def encoded_img_to_pil_img(data_str):
    base64_str = data_str.replace("data:image/png;base64,", "")
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))

    return image

def pil_img_2_bytes(image):
    # 创建一个BytesIO对象
    byte_io = io.BytesIO()

    # 将图像保存到BytesIO对象中，指定格式
    image.save(byte_io, format='PNG')  # 格式可以是'JPEG', 'PNG'等
    # 获取图像的字节表示
    image_bytes = byte_io.getvalue()
    return image_bytes

def pil_to_base64(image):
    with open(image, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_image

def save_to_tmp_img_file(data_str):
    base64_str = data_str.replace("data:image/png;base64,", "")
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))

    tmp_img_path = os.path.join(tempfile.mkdtemp(), "tmp_img.png")
    image.save(tmp_img_path)

    return tmp_img_path



def parse_actions_from_string(input_string):
    if input_string.strip() in ["WAIT", "DONE", "FAIL"]:
        return [input_string.strip()]
    # Search for a JSON string within the input string
    actions = []
    matches = re.findall(r"```json\s+(.*?)\s+```", input_string, re.DOTALL)
    if matches:
        # Assuming there's only one match, parse the JSON string into a dictionary
        try:
            for match in matches:
                action_dict = json.loads(match)
                actions.append(action_dict)
            return actions
        except json.JSONDecodeError as e:
            return f"Failed to parse JSON: {e}"
    else:
        matches = re.findall(r"```\s+(.*?)\s+```", input_string, re.DOTALL)
        if matches:
            # Assuming there's only one match, parse the JSON string into a dictionary
            try:
                for match in matches:
                    action_dict = json.loads(match)
                    actions.append(action_dict)
                return actions
            except json.JSONDecodeError as e:
                return f"Failed to parse JSON: {e}"
        else:
            try:
                action_dict = json.loads(input_string)
                return [action_dict]
            except json.JSONDecodeError:
                raise ValueError("Invalid response format: " + input_string)


def parse_code_from_string(input_string):
    input_string = "\n".join(
        [line.strip() for line in input_string.split(";") if line.strip()]
    )
    if input_string.strip() in ["WAIT", "DONE", "FAIL"]:
        return [input_string.strip()]

    # This regular expression will match both ```code``` and ```python code```
    # and capture the `code` part. It uses a non-greedy match for the content inside.
    pattern = r"```(?:\w+\s+)?(.*?)```"
    # Find all non-overlapping matches in the string
    matches = re.findall(pattern, input_string, re.DOTALL)

    # The regex above captures the content inside the triple backticks.
    # The `re.DOTALL` flag allows the dot `.` to match newline characters as well,
    # so the code inside backticks can span multiple lines.

    # matches now contains all the captured code snippets

    codes = []

    for match in matches:
        match = match.strip()
        commands = [
            "WAIT",
            "DONE",
            "FAIL",
        ]  # fixme: updates this part when we have more commands

        if match in commands:
            codes.append(match.strip())
        elif match.split("\n")[-1] in commands:
            if len(match.split("\n")) > 1:
                codes.append("\n".join(match.split("\n")[:-1]))
            codes.append(match.split("\n")[-1])
        else:
            codes.append(match)

    return codes


def parse_code_from_som_string(input_string, masks):
    # parse the output string by masks
    tag_vars = ""
    for i, mask in enumerate(masks):
        x, y, w, h = mask
        tag_vars += (
            "tag_"
            + str(i + 1)
            + "="
            + "({}, {})".format(int(x + w // 2), int(y + h // 2))
        )
        tag_vars += "\n"

    actions = parse_code_from_string(input_string)

    for i, action in enumerate(actions):
        if action.strip() in ["WAIT", "DONE", "FAIL"]:
            pass
        else:
            action = tag_vars + action
            actions[i] = action

    return actions


def trim_accessibility_tree(linearized_accessibility_tree, max_tokens):
    # enc = tiktoken.encoding_for_model("gpt-4")
    # tokens = enc.encode(linearized_accessibility_tree)
    # if len(tokens) > max_tokens:
    #     linearized_accessibility_tree = enc.decode(tokens[:max_tokens])
    #     linearized_accessibility_tree += "[...]\n"
    return linearized_accessibility_tree

class PromptAgent:
    def __init__(
        self,
        platform="ubuntu",
        max_tokens=1000,
        top_p=0.9,
        top_k=1.0,
        temperature=0.0,
        action_space="computer_13",
        observation_type="screenshot",
        # observation_type can be in ["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"]
        max_trajectory_length=50,
        a11y_tree_max_tokens=10000,
        runtime_conf: dict = {}
    ):
        self.platform = platform
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature
        self.action_space = action_space
        self.observation_type = observation_type
        self.max_trajectory_length = max_trajectory_length
        self.a11y_tree_max_tokens = a11y_tree_max_tokens
        self.runtime_conf = runtime_conf
        self.model = None # should replace with your UI-TARS server api
        self.infer_mode = 'qwen2vl_user'
        self.prompt_style = 'qwen2vl_no_thought'

        self.language = 'English'
        self.max_steps = 50

        self.thoughts = []
        self.actions = []
        self.observations = []
        self.history_images = []
        self.history_responses = []
        self.history_ins = []
        

        self.customize_action_parser = parse_action_android_control
        self.action_parse_res_factor = 1000

            

        self.prompt_action_space = Android_control_Action_Space5   

        


        self.prompt_template = prompt

        
        self.history_n = 5

        self.deployment_name = "ui-tars"
        
        api_base = "http://0.0.0.0:8000/v1"
        api_key= "empty",
        deployment_name="ui-tars"
        
        
        self.client = OpenAI(
            base_url=api_base,
            api_key=api_key,
        )
        
        
    def model_infer_client(self, messages):
        # print("messages", messages)
        res = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
            frequency_penalty=1,
            max_tokens=2048,
            temperature=0.0,
            top_p=self.top_p
        )
        res = res.choices[0].message.content
        
        return res
    def predict(
        self, row
    ) -> List:
        """
        Predict the next action(s) based on the current observation.
        """

        # Append trajectory
        # print(len(self.observations), len(self.actions), len(self.actions))
        #
        
        results = []
        for i in range(len(row['step_instructions'])):
            instruction = row['step_instructions'][i]
            obs = {"screenshot":row['screenshots'][i]}
            # print("self.observations",self.observations)
            # print("self.actions",self.actions)
            # print("self.thoughts",self.thoughts)
            assert len(self.observations) == len(self.actions) and len(self.actions) == len(
                self.thoughts
            ), "The number of observations and actions should be the same."

            if len(self.observations) > self.max_trajectory_length:
                if self.max_trajectory_length == 0:
                    _observations = []
                    _actions = []
                    _thoughts = []
                else:
                    _observations = self.observations[-self.max_trajectory_length :]
                    _actions = self.actions[-self.max_trajectory_length :]
                    _thoughts = self.thoughts[-self.max_trajectory_length :]
            else:
                _observations = self.observations
                _actions = self.actions
                _thoughts = self.thoughts


            self.history_images.append(obs["screenshot"])
            self.history_ins.append(instruction)

            if self.observation_type in ["screenshot", "screenshot_a11y_tree"]:
                base64_image = obs["screenshot"]
                # logger.debug("LINEAR AT: %s", linearized_accessibility_tree)

                self.observations.append(
                    {"screenshot": base64_image}
                )

            else:
                raise ValueError(
                    "Invalid observation_type type: " + self.observation_type
                )  # 1}}}
            
            if self.infer_mode == "qwen2vl_user":
                user_prompt = self.prompt_template.format(
                    instruction=row['goal'],
                    action_space=self.prompt_action_space,
                )
            elif self.infer_mode == "qwen2vl_no_thought":
                user_prompt = self.prompt_template.format(
                    instruction=row['goal'],
                    action_space= self.prompt_action_space
                )
            user_prompt = self.prompt_template.format(
                instruction=row['goal'],
                action_space=self.prompt_action_space,
            )

            if len(self.history_images) > self.history_n:
                self.history_images = self.history_images[-self.history_n:]
                self.history_ins = self.history_ins[-self.history_n:]
                
            if len(self.history_responses) > self.history_n - 1:
                self.history_responses = self.history_responses[-(self.history_n-1):]
            
            max_pixels = 1350 * 28 * 28
            min_pixels = 100 * 28 * 28
            messages, images, messages2, instructions = [], [], [], []
            
            if isinstance(self.history_images, bytes):
                self.history_images = [self.history_images]
            elif isinstance(self.history_images, np.ndarray):
                self.history_images = list(self.history_images)
            elif isinstance(self.history_images, list):
                pass
            else:
                raise TypeError(f"Unidentified images type: {type(self.history_images)}")
            max_image_nums_under_32k = int(32768*0.75/max_pixels*28*28)
            
            if len(self.history_images) > max_image_nums_under_32k:
                num_of_images = min(5, len(self.history_images))
                max_pixels = int(32768*0.75) // num_of_images

            for turn, image in enumerate(self.history_images):
                if len(images) >= 5:
                    break


                images.append(image)
                instructions.append(self.history_ins[turn])
            
            # 先添加第一个prompt
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_prompt}]
                }
            ]
            
            messages2 = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_prompt}]
                }
            ]
            
            image_num = 0
            if len(self.history_responses) > 0:
                for history_idx, history_response in enumerate(self.history_responses):
                    # send at most history_n images to the model
                    # if history_idx + self.history_n > len(self.history_responses):

                    cur_image = images[image_num]
                    cur_ins = instructions[image_num]
                    
                    encoded_string = pil_to_base64(cur_image)
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url", 
                                "image_url": 
                                    {
                                    "url": f"data:image/png;base64,{encoded_string}"
                                    }
                            },
                            {"type": "text", "text": cur_ins}
                        ]
                    })
                    
                    messages2.append({
                        "role": "user",
                        "content": [{
                            "type": "image", 
                            "image": cur_image
                        },
                        {"type": "text", "text": cur_ins}]
                    })
                    
                    image_num += 1
                        
                    messages.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": history_response}]
                    })
                    
                    messages2.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": history_response}]
                    })
                    ## "content": [history_response]

                

                cur_image = images[image_num]
                cur_ins = instructions[image_num]
                
                encoded_string = pil_to_base64(cur_image)
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{encoded_string}"
                            }},
                            {"type": "text", "text": cur_ins}]
                })
                
                messages2.append({
                    "role": "user",
                    "content": [{"type": "image", "image": cur_image},
                            {"type": "text", "text": cur_ins}]
                })
                
                image_num += 1
            
            else:
                cur_image = images[image_num]
                cur_ins = instructions[image_num]
                
                encoded_string = pil_to_base64(cur_image)
                messages.append({
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{encoded_string}"
                            }},
                            {"type": "text", "text": cur_ins}]
                })
                
                messages2.append({
                    "role": "user",
                    "content": [{"type": "image", "image": cur_image},
                            {"type": "text", "text": cur_ins}]
                })
                
                image_num += 1
            
            print(messages2, flush=True)
            prediction = [self.model_infer_client(messages)]
            # prediction = ['test']
            action_gt = row['actions'][i]
            parsed_responses = self.customize_action_parser(
                action_gt,
                self.action_parse_res_factor,
                row['screenshot_heights'][i],
                row['screenshot_widths'][i]
            )

            
            self.history_responses.append(parsed_responses)
            self.thoughts.append(prediction)

            
            self.actions.append(row['actions'][i])
            
            if len(self.history_responses) >= self.max_trajectory_length:
                # Default to FAIL if exceed max steps
                actions = ["FAIL"]

            results.append(prediction)
            # return prediction, actions, parsed_responses
            print("prediction:", prediction)
        # raise
        self.reset()
        
        return results
    
    def parse_actions(self, response: str, masks=None):

        if self.observation_type in ["screenshot", "a11y_tree", "screenshot_a11y_tree"]:
            # parse from the response
            if self.action_space == "computer_13":
                actions = parse_actions_from_string(response)
            elif self.action_space == "pyautogui":
                actions = parse_code_from_string(response)
            else:
                raise ValueError("Invalid action space: " + self.action_space)

            self.actions.append(actions)

            return actions
        elif self.observation_type in ["som"]:
            # parse from the response
            if self.action_space == "computer_13":
                raise ValueError("Invalid action space: " + self.action_space)
            elif self.action_space == "pyautogui":
                actions = parse_code_from_som_string(response, masks)
            else:
                raise ValueError("Invalid action space: " + self.action_space)

            self.actions.append(actions)

            return actions

    def reset(self):
        self.thoughts = []
        self.actions = []
        self.observations = []
        self.history_images = []
        self.history_responses = []
        self.history_ins = []



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, help='The directory where to write output')
    parser.add_argument('--base_model_path', type=str, help='The directory where to write output')
    
    args = parser.parse_args()
    
    if args.base_model_path == None:
        base_model_path = '/model/UI-TARS-2B-SFT'
    else:
        base_model_path = args.base_model_path
    
    
    

    json_file = "./android_control_test.json"

    # 使用 pandas 读取 JSON 文件
    data = pd.read_json(json_file,lines=True)


    agent = PromptAgent()

    
    model_name = base_model_path.split('/')[-1]

    json_file_name = json_file.split('/')[-1].split('.')[0]
    

    suffix = '_'.join([model_name, json_file_name, '.json'])
    outputfile = './output' + suffix
    
    
    directory = os.path.dirname(outputfile)

    # 判断目录是否存在，如果不存在则创建
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"目录 {directory} 已创建。")
        

    for idx, row in data.iterrows():
        # try:

        formatted_result = agent.predict(row)
        # .
        
        
        # 转换为单行JSON
        row_json = json.dumps(
            {**row.to_dict(), 'result': formatted_result},
            ensure_ascii=False
        )
        
        # 追加写入
        with open(outputfile, 'a', encoding='utf-8') as f:
            f.write(row_json + '\n')
            
