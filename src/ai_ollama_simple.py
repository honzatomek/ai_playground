#!/usr/bin/env python3.12

import os
import sys

import json
import inspect
from typing import get_type_hints
from typing import Callable, Type

import rich
from rich import print_json

import ollama
from ollama import Client

AVAILABLE_MODELS = ["llama3"]


def get_type_name(t: Type):
    name = str(t)
    if "list" in name or "dict" in name:
        return name
    else:
        return t.__name__


def function_to_json(func: Callable):
    signature = inspect.signature(func)
    type_hints = get_type_hints(func)

    function_info = {
        "name": func.__name__,
        "description": func.__doc__,
        "parameters": {"type": "object", "properties": {}},
        "returns": type_hints.get("return", "void").__name__,
    }

    for name, _ in signature.parameters.items():
        param_type = get_type_name(type_hints.get(name, type(None)))
        function_info["parameters"]["properties"][name] = {"type": param_type}

    return json.dumps(function_info, indent=2)


def print_tools(tools: list) -> str:
    return "\n".join([function_to_json(t) for t in tools])


def read_file(filename: str) -> str:
    """Read a file from disk

    Args:
        filename: file name with path of a file to read

    Returns:
        filename preceded by #, empty line and then text contents of a file
    """
    with open(filename, "rt", encoding="utf-8") as file:
        text = file.read()
    return "# " + filename + "\n\n" + text


def directory_contents(dirpath: str) -> str:
    """List the contents of a directory

    Args:
        dirpath: path to the directory to list

    Returns:
        text with contents of a directory, one entry on each line
    """
    filenames = os.listdir(dirpath)
    return "\n".join(filenames)


def write_file(filename: str, text: str) -> str:
    """Write a file to the disk

    Args:
        filename: file name with path of a file to write
        text:     contents of the file to write

    Returns:
        True on success, contents of error on failure
    """
    try:
        with open(filename, "wt", encoding="utf-8") as file:
            file.write(text)
    except Exception as e:
        return str(e)


class Message:
    def __init__(self, role: str, content: str = None):
        self.role = role
        self.content = content

    @property
    def role(self) -> str:
        return self._role

    @role.setter
    def role(self, role: str = "user"):
        if role.lower() in ("system", "assistant", "user", "tool"):
            self._role = role.lower()
        else:
            raise ValueError(f"Attempt to assign a wrong role to {type(self).__str__:s} ({str(role):s}).")

    @property
    def content(self) -> str:
        return self._content

    @content.setter
    def content(self, content: str = None):
        self._content = str(content) if content is not None else None

    def __getitem__(self, key: str) -> str:
        if key.lower() == "role":
            return self.role
        elif key.lower() == "content":
            return self.content
        else:
            raise ValueError(f"Key {str(key):s} not found in {type(self).__str__:s}.")

    def __setitem__(self, key: str, value: str):
        if key.lower() == "role":
            self.role = key
        elif key.lower() == "content":
            return self.setattr(key, value)
        else:
            raise ValueError(f"Key {str(key):s} not found in {type(self).__str__:s}.")

    def todict(self) -> dict:
        return {"role": self.role, "content": self.content}


class ChatBot:
    tools = [read_file,
             write_file,
             directory_contents]

    system_prompt = f"""You are a helpfull assistant.

You have access to the following tools:
{print_tools(tools):s}

You must follow these instructions:
You can select one or more of the above tools based on the user query.
You must respond in the JSON format matching the following schema:
{{
   "reply": "<your reply if there is no need for a tool>",
   "tools": [{{
        "tool": "<name of the selected tool>",
        "tool_input": "<parameters for the selected tool, matching the tool's JSON schema>",
   }},]
}}
If there are multiple tools required, make sure a list of tools are returned in a JSON array.
If there is no tool that matches the user request or you do not need to use any,
you will respond with empty JSON array in the "tools" part.
Do not add additional notes or explanations.

User Query:
"""
    @staticmethod
    def process_response(response: str) -> list:
        try:
            processed = json.loads(response["message"]["content"])
            assistant = Message(role=response["message"]["role"], content=processed["reply"])
            if assistant.content == "":
                assistant.content = None
            tools = processed["tools"]
            return assistant, tools
        except Exception as e:
            return Message(role=response["message"]["role"], content=response["message"]["content"]), []


    def __init__(self,
                 host: str   = "http://localhost:11434",
                 model: str  = "llama3"):
        self.model = model
        self.client = Client(host="http://localhost:11434")
        self.messages = [Message("system", self.system_prompt)]

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, model: str = "llama3") -> str:
        if model in AVAILABLE_MODELS:
            self._model = model
        else:
            raise ValueError(f"LLM Model not found ({str(model):s}.")

    def tolist(self) -> list:
        res = []
        for m in self.messages:
            if m.role == "tool":
                res.append({"role": "user", "content": m.content})
            else:
                res.append(m.todict())
        return res

    def __getitem__(self, i: int) -> Message:
        return self.messages[i]

    def prompt(self, prompt: str):
        self.messages.append(Message("user", prompt))
        response = self.client.chat(model=self.model, messages=self.tolist())
        self.messages.append(Message("assistant", response["message"]["content"]))
        return self.messages[-1]

    def communicate(self):
        raw_response = self.client.chat(model=self.model, messages=self.tolist())
        response, tools = self.process_response(raw_response)

        if response.role == "assistant" and response.content is not None:
            self.messages.append(response)

        for tool in tools:
            print(f"Calling {tool['tool']:s}({', '.join([f'{k:s}="{v:s}"' for k, v in tool['tool_input'].items()]):s})")
            response = str(eval(tool["tool"])(**tool["tool_input"]))
            self.messages.append(Message(role="tool", content=response))

        return self.messages[-1]


if __name__ == "__main__":

    chatbot = ChatBot()
    print(f"{chatbot[-1].role:<9s} > {chatbot[-1].content:s}")
    while True:
        if chatbot[-1].role in ("system", "assistant"):
            prompt = input(f"{'user':<9s} > ")
            if prompt.lower() in ("", "q", "quit", "e", "exit"):
                break
            chatbot.messages.append(Message("user", prompt))
        response = chatbot.communicate()
        if response.role == "assistant":
            print(f"{chatbot[-1].role:<9s} > {chatbot[-1].content:s}")


