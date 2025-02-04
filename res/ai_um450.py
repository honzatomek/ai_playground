#!/usr/bin/env python3.12

import os
import sys
import re
import json
from typing import List


with open("./um450_edu_v19_trial02_poppler_layout.json", "rt", encoding="utf-8") as f:
    UM450 = json.loads(f.read())

def get_chapters() -> str:
    return "\n".join(list([v["_name"] for k, v in UM450.items() if not k.startswith("_")]))


def get_subchapters(chapter: str) -> str:
    if chapter not in set([v["_name"] for k, v in UM450.items()]):
        return f"{chapter:s} was not found, mut be one of {', '.join([v["_name"] for k, v in UM450.items() if not k.startswith("_")]):s}."
    else:
        chapter = [UM450[k] for k, v in UM450.items() if v["_name"] == chapter if not k.startswith("_")][0]
        return "\n".join([v["_name"] for k, v in chapter.items() if not k.startswith("_")])


def get_uci_sections() -> str:
    sections = []
    for k, v in [(k, v) for k, v in UM450["2"].items() if not k.startswith("_")]:
        if "section" in v["_name"].lower():
            sections.append(v["_name"].split(" ")[-2])
        elif "commands" in v["_name"].lower():
            sections.append(" ".join(v["_name"].split(" ")[-2:]))
    return "\n".join(sections)

def get_uci_keywords(section: str) -> str:
    sections = [v for k, v in UM450["2"].items() if not k.startswith("_") and section in v["_name"]]
    if len(sections) == 0:
        sections = get_uci_sections.split("\n")
        return f"Section {section:s} not found, must be one of: {', '.join(sections):s}."
    else:
        return "\n".join([k for k, v in sections[0].items() if not k.startswith("_")])

def get_uci_keyword(section: str, keyword: str) -> str:
    for k, v in [(k, v) for k, v in UM450["2"].items() if not k.startswith("_")]:
        if section in v["_name"]:
            if keyword in v.keys():
                return v[keyword]
            else:
                keywords = get_dat_keywords(section).split("\n")
                return (f"Keyword {keyword:s} not found in section {section:s}, " +
                        f"keyword must be one of: {', '.join(keywords):s}.")
    else:
        sections = get_uci_sections.split("\n")
        return f"Section {section:s} not found, must be one of: {', '.join(sections):s}."

def get_dat_blocks() -> str:
    ret = ["COMPONENT", "MATERIAL", "FUNCTION", "GEOMETRY", "MISC"]
    return "\n".join(ret)

def get_dat_block_variants(block: str) -> str:
    blocks = get_dat_blocks().split("\n")
    if block not in blocks:
        return f"Block {block:s} not found, must be one of: {', '.join(blocks):s}."
    transl = {"COMPONENT": "Model Data",
              "MATERIAL":  "Material Data",
              "FUNCTION":  "Function Data",
              "GEOMETRY":  "Geometry Data",
              "MISC":      "Miscellaneus",}
    breakpoint()
    p = [v for k, v in UM450["3"].items() if trans[block] in v["_name"]][0]



def get_dat_variants(toplevel: str) -> str:
    if toplevel not in get_dat_toplevel().split("\n"):
        return f"toplevel must be one of {', '.join(get_dat_toplevel()):s}"

# print(get_chapters())
# print(get_subchapters("2 User Control Interface (UCI)"))
# print(get_subchapters("3 Data Input (DAT)"))
# print(get_uci_sections())
# print(get_uci_keywords("Global Commands"))
# print(get_uci_keywords("EXPORT"))
# print(get_uci_keyword("EXEC", "STATIC"))
# print(get_dat_block_variants("COMPONENT"))



