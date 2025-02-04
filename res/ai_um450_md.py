#!/usr/bin/env python3.12

import os
import sys
import re
import json
from typing import List

UM450 = {}
with open("./um450_edu_v19_trial02_poppler_layout.md", "rt", encoding="utf-8") as f:
    for line in f:
        if line.startswith("#"):
            keyword = line.rstrip()
        if keyword not in UM450.keys():
            UM450.setdefault(keyword, line)
        else:
            UM450[keyword] += line

for k, v in UM450.items():
    lines = v.split("\n")
    print(f"{k = }, lines = {len(lines):d}")

