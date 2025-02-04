#!/usr/bin/env python3.12

# https://ollama.com/blog/embedding-models

import version

__doc__ = f"""Script to convert markdown file to LLM embeddings
author:  {version.__author__:s}
version: {version.__version__:s}
date:    {version.__date__:s}

description:
Script to read markdown file to a flattened dict and export it to a LLM embeddings
index stored as a json format.

"""


import os
import sys
import re
import json
import numpy as np
import argparse

from modules import ai_embed

_RE_MD_LEVEL = re.compile(r"^(?P<level>#+)\s+(?P<title>.*)$")


class CheckFile(argparse.Action):
    """Argparse action to check whether the file exists"""
    def __call__(self, parser, namespace, fname, option_string=None):
        if not os.path.isfile(fname):
            parser.error(f"{fname:s} does not exist.")
        else:
            setattr(namespace, self.dest, fname)


def md_to_dict(filename: str, flattened: bool = True) -> dict:
    filedict = {}
    fd = {}
    title = "root"
    level = 0
    nlevel = None
    levelstr = ""
    path = ["root"]
    with open(os.path.realpath(filename), "rt", encoding="utf-8") as md:
        while True:
            last_pos = md.tell()
            line = md.readline()
            if not line: # EOF
                break


            if line.startswith("#"):
                m = re.match(_RE_MD_LEVEL, line)
                if m:
                    title  = m["title"]
                    nlevel = len(m["level"]) - 1

                    if nlevel < level:
                        while level > nlevel:
                            path.pop()
                            level -= 1

                    elif nlevel > level:
                        while level < nlevel:
                            path.append("__dummy__")
                            level += 1

                    path[level] = title

                    if flattened:
                        title = "#".join(path)
                        if title not in filedict.keys():
                            filedict.setdefault(title, "")
                        fd = filedict

                    else:
                        fd = filedict
                        for title in path:
                            if title not in fd.keys():
                                fd.setdefault(title, {})
                            fd = fd[title]

            if flattened:
                fd[title] += line

            else:
                if "_contents" not in fd.keys():
                    fd.setdefault("_contents", "")
                fd["_contents"] += line

    return filedict



def main(md_filename: str, json_filename: str):
    md_dict = md_to_dict(md_filename, flattened = True)

    embeddings = ai_embed.embed([v for k, v in md_dict.items()])

    for k, e in zip(list(md_dict.keys()), embeddings):
        md_dict[k] = {"_contents": md_dict[k], "_embedding": e.tolist()}

    with open(json_filename, "wt", encoding="utf-8") as jf:
        jf.write(json.dumps(md_dict, indent=2))



if __name__ == "__main__":
    # Create options parser object.
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)

    # Add arguments
    parser.add_argument("-v", "--verbose", dest="verbose", action="count", default=0,
                        help="""Increase verbosity.""")

    parser.add_argument("markdown_file", type=str, action=CheckFile,
                        help="""Markdown file to process.""")

    # Parse command-line arguments.
    args = parser.parse_args()

    # Prepare output filename
    json_outfile = os.path.splitext(os.path.realpath(args.markdown_file))[0] + ".json"

    # Run the script
    main(args.markdown_file, json_outfile)

