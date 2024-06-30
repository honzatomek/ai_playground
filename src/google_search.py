#!/usr/bin/env python3.12

import os
import sys

from googlesearch import search as gsearch

query = "Current temperature at Zakynthos"

print(f"{query = }")
for url in gsearch(query, num_results=10):
    print(f"{url = }")

