#!/usr/bin/env python3.12

import os
import sys

import json
import inspect
from typing import get_type_hints
from typing import Callable, Type, List

import rich
from rich import print_json

import ollama
from ollama import Client

LLM_MODELS = ["llama3"]


def get_type_name(t: Type):
    name = str(t)
    if "list" in name or "dict" in name:
        return name
    else:
        return t.__name__



def docstring_to_dict(docstring: str, parameters: list) -> dict:
    lines = docstring.split("\n")
    docdict = {
        "short description": [],
        "long description": [],
        "args": [],
        "retruns": [],
        "raises": []
        }

    key = "short description"
    for i in range(len(lines)):
        if lines[i].strip() == "":
            break
        if key not in docdict.keys():
            docdict.setdefault(key, [])
        docdict[key].append(lines[i].strip())

    key = "long description"
    for i in range(i, len(lines)):
        if lines[i].strip().lower().startswith("args"):
            key = "args"
            continue
        elif lines[i].strip().lower().startswith("returns"):
            key = "returns"
            continue
        elif lines[i].strip().lower().startswith("raises"):
            key = "raises"
            continue

        if key not in docdict.keys():
            docdict.setdefault(key, [])

        docdict[key].append(lines[i].strip())

    kwargs = {}
    key = ""
    for line in docdict["args"]:
        if line.split(":")[0] in parameters:
            key = line.split(":", 1)[0].strip()
            line = line.split(":", 1)[1].strip()
        if key != "":
            if key not in kwargs.keys():
                kwargs.setdefault(key, [])
            kwargs[key].append(line)

    for key in docdict.keys():
        docdict[key] = "\n".join(docdict[key]).strip()

    for key in kwargs.keys():
        kwargs[key] = "\n".join(kwargs[key]).strip()

    docdict["args"] = kwargs

    return docdict




def read_file(filename: str) -> str:
    """Read a file from disk

    Args:
        filename: file name with path of a file to read

    Returns:
        filename preceded by #, empty line and then text contents of a file
    """
    try:
        with open(filename, "rt", encoding="utf-8") as file:
            text = file.read()
        return "# " + filename + "\n\n" + text
    except Exception as e:
        return str(e)



def directory_contents(dirpath: str) -> str:
    """List the contents of a directory

    Args:
        dirpath: path to the directory

    Returns:
        text with contents of a directory, one entry on each line
    """
    try:
        filenames = os.listdir(dirpath)
        return "\n".join(filenames)
    except Exception as e:
        return str(e)



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



class Tool:
    def __init__(self, func: Callable):
        """AI Chatbot Tool constructor

        Args:
            func: a reference to a function definition
        """
        self.function = func

    @property
    def name(self) -> str:
        return self.function.__name__

    # def __str__(self):
    #     """returns a function definition in a json format"""
    #     signature  = inspect.signature(self.function)
    #     type_hints = get_type_hints(self.function)

    #     docdict = docstring_to_dict(self.function.__doc__, list(signature.parameters.keys()))

    #     # "description": self.function.__doc__,
    #     function_info = {
    #         "name":        self.function.__name__,
    #         "parameters":  {"type": "object", "properties": {}},
    #         "returns":     type_hints.get("return", "void").__name__,
    #     }
    #     if "short description" in docdict.keys():
    #         function_info.setdefault("short description", docdict["short description"])
    #     else:
    #         function_info.setdefault("description", self.function.__doc__)

    #     if "long description" in docdict.keys():
    #         function_info.setdefault("long description", docdict["long description"])

    #     breakpoint()
    #     for name, _ in signature.parameters.items():
    #         param_type = get_type_name(type_hints.get(name, type(None)))
    #         if name in docdict["args"].keys():
    #             function_info["parameters"]["properties"][name] = {"type": param_type,
    #                                                                "description": docdict["args"][name],}
    #         else:
    #             function_info["parameters"]["properties"][name] = {"type": param_type}

    #     return json.dumps(function_info, indent=2)

    def __str__(self):
        """returns a function definition in a json format"""
        signature  = inspect.signature(self.function)
        type_hints = get_type_hints(self.function)

        function_info = {
            "name":        self.function.__name__,
            "description": self.function.__doc__,
            "parameters":  {"type": "object", "properties": {}},
            "returns":     type_hints.get("return", "void").__name__,
        }

        for name, _ in signature.parameters.items():
            param_type = get_type_name(type_hints.get(name, type(None)))
            function_info["parameters"]["properties"][name] = {"type": param_type}

        return json.dumps(function_info, indent=2)

    def show(self, **kwargs):
        return f"{self.name:s}({', '.join([f'{k:s}="{v:s}"' for k, v in kwargs.items()]):s})"

    def call(self, **kwargs):
        return f"{self.name:s} reply:" + "\n" + str(self.function(**kwargs))



class Message:
    def __init__(self, role: str, content: str = None):
        self.role = role
        self.content = content

    def print(self, tool_call: str = None):
        if tool_call:
            print(f"> {self.role:s}: {tool_call:s}" + "\n  " + "\n  ".join(self.content.split("\n")) + "\n")
        else:
            print(f"> {self.role:s}:" + "\n  " + "\n  ".join(self.content.split("\n")) + "\n")

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

    @staticmethod
    def print_tools(tools: List[Callable]) -> str:
        """prints tools defined as a list of references to function in a json format"""
        return "\n".join([function_to_json(t) for t in tools])

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
                 model: str  = "llama3",
                 system: str = "You are a helpful assistant"):
        self.tools = {}
        self.model = model
        self.client = Client(host=host)
        self.system = system
        self.messages = [Message("system", self.system_prompt)]

    @property
    def system_prompt(self):
        if len(self.tools.keys()) == 0:
            return self.system
        else:
            str_tools = "\n".join([str(t) for n, t in self.tools.items()])
            return f"""{self.system:s}

You have access to the following tools:
{str_tools:s}

You must follow these instructions:
You can select one or more of the above tools based on the user query.
You must respond in the JSON format matching the following schema:
{{
   "reply": "<your reply if there is no need for a tool>",
   "tools":
   [
     {{
        "tool": "<name of the selected tool>",
        "tool_input": "<parameters for the selected tool, matching the tool's JSON schema>",
     }},
   ],
}}

If there are multiple tools required, make sure a list of tools is returned as a JSON array.
If there is no tool that matches the user request or you do not need to use any,
you will respond with empty JSON array in the "tools" part.
Do not add additional notes or explanations.

User Query:
"""

    def register_tool(self, func: Callable):
        tool = Tool(func)
        self.tools[tool.name] = tool
        self.messages[0] = Message("system", self.system_prompt)

    def unregister_tools(self, func: Callable):
        if func.__name__ in self.tools.keys():
            self.tools.pop(func.__name__)
        self.messages[0] = Message("system", self.system_prompt)

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, model: str = "llama3") -> str:
        if model in LLM_MODELS:
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

        # print(tools)
        for tool in tools:
            tool_call = self.tools[tool["tool"]].show(**tool["tool_input"])
            response = self.tools[tool["tool"]].call(**tool["tool_input"])
            self.messages.append(Message(role="tool", content=response))
            # print(f"{self[-1].role:s} {tool_call:s} >" + "\n" + f"{self[-1].content:s}" + "\n")
            self[-1].print(tool_call)

        return self.messages[-1]

    def chat(self):
        # print(f"> {self[-1].role:s}:" + "\n" + f"{self[-1].content:s}")
        self[0].print()
        while True:
            if self[-1].role in ("system", "assistant"):
                prompt = input(f"> {'user (q for quit)':s}:" + "\n")
                if prompt.lower() in ("", "q", "quit", "e", "exit"):
                    break
                print()
                self.messages.append(Message("user", prompt))

            response = self.communicate()
            if response.role == "assistant":
                # contents = ("\n" + " " * 12).join(self[-1].content.split("\n"))
                # print(f"{self[-1].role:<9s} > {contents:s}")
                # print(f"> {self[-1].role:s}: " + "\n" + "\n ".join(self[-1].content.split("\n") + "\n")
                self[-1].print()



if __name__ == "__main__":
    docstring_to_dict(directory_contents.__doc__, ["dirpath"])

    chatbot = ChatBot()
    chatbot.register_tool(read_file)
    chatbot.register_tool(directory_contents)
    chatbot.register_tool(write_file)
    chatbot.chat()


