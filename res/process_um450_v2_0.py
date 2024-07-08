#!/usr/bin/env python3.12

import os
import sys
import re
import io
import json

UM450 = "./um450_edu_v19_trial02_poppler_layout.txt"

_RE_FOOTER_L = re.compile(r"^\s*(?P<version>V \d+\.\d+\.\d+)\s+(?P<permas>([a-zA-Z0-9'’()]+ )+)\s+(?P<page>[a-zA-Z0-9\-.]+)\s*$")
_RE_FOOTER_R = re.compile(r"^\s*(?P<page>[a-zA-Z0-9\-.]+)\s+(?P<permas>([a-zA-Z0-9'’()]+ )+)\s+(?P<version>V \d+\.\d+\.\d+)\s*$")

_RE_KEYWORD = re.compile(r"\s{2,}")
_RE_TOC_SPACE = re.compile(r"\s\s+")
_RE_TOC_DOTS  = re.compile(r"\s+\.( \.)+\s+")


def dict_to_md(d: dict, level: int = 0) -> str:
    ret = ""
    if "_name"  in d.keys():
        ret = "#" * level + " " + d["_name"] + "\n\n"
    if "_contents"  in d.keys() and len(d["_contents"]) > 0:
        while "\n\n\n" in d["_contents"]:
            d["_contents"] = d["_contents"].replace("\n\n\n", "\n\n")
        ret += d["_contents"].lstrip("\n").rstrip("\n") + "\n\n\n"

    keys = list([k for k in d.keys() if not k.startswith("_")])
    for k in keys:
        if isinstance(d[k], str):
            ret += "#" * (level + 1) + " " + k + "\n\n"
            while "\n\n\n" in d[k]:
                d[k] = d[k].replace("\n\n\n", "\n\n")
            ret += d[k].lstrip("\n").rstrip("\n") + "\n\n\n"
        else:
            ret += dict_to_md(d[k], level + 1)

    return ret.rstrip("\n") + "\n\n\n"


def stripline(line: str) -> str:
    while "  " in line:
        line = line.replace("  ", " ")
    return line.strip()


def get_offset(file: io.TextIOWrapper) -> int:
    last_pos = file.tell()
    offset = []

    while True:
        line = file.readline()
        if not line:
            break
        elif re.match(_RE_FOOTER_L, line):
            break
        elif re.match(_RE_FOOTER_R, line):
            break
        elif line.strip("").strip() == "":
            continue
        elif stripline(line) in skip:
            continue
        _o = len(line) - len(line.lstrip(" "))
        # print(f"{_o = }, {line = }")
        offset.append(_o)

    file.seek(last_pos)

    if len(offset) < 5:
        return 0
    else:
        return min(offset[2:])


def is_subtitle(suptitle: str, subtitle: str) -> bool:
    supnum = suptitle.split(" ")[0].split(".")
    subnum = subtitle.split(" ")[0].split(".")
    if len(subnum) < len(supnum):
        return False
    for i in range(len(supnum)):
        if subnum[i] != supnum[i]:
            return False
    return True


def readline(file: io.TextIOWrapper) -> (int, str, str):
    last_pos = file.tell()
    line = file.readline()

    if not line: # EOF
        return last_pos, "", line

    line = line.lstrip("")
    if "" in line:
        line = line.replace("", "")

    if "" in line:
        line = line.replace("", "")

    if line.strip() == "":
        line = "\n"

    keyword = ""
    offset = -1

    if re.match(_RE_FOOTER_L, line):
        last_pos = file.tell()
        line = file.readline()
        line = line.lstrip("").strip()
        keyword = re.split(_RE_KEYWORD, line)
        if len(keyword) > 1:
            keyword = keyword[0].upper()
            # print(f"{keyword = }")
        else:
            keyword = ""
        offset = get_offset(file)

    elif re.match(_RE_FOOTER_R, line):
        last_pos = file.tell()
        line = file.readline()
        line = line.lstrip("").strip()
        keyword = re.split(_RE_KEYWORD, line)
        if len(keyword) > 1:
            keyword = keyword[-1].upper()
            # print(f"{keyword = }")
        else:
            keyword = ""
        offset = get_offset(file)

    return last_pos, keyword, offset, line


TOC = set()
parsed = {"Introduction": {"_name": "Introduction", "_contents": ""}}
p = parsed["Introduction"]
title = "Introduction"
keyword = ""
last_pos = 0
offset = 0

skip = set(["How-to",
            "Intro",
            "UCI",
            "Global",
            "DAT",
            "Element",
            "Atlas",
            "Design",
            "Elements",
            "Results",
            "Export",
            "Export to",
            "Material",
            "STARTUP",
            "INPUT",
            "EXEC",
            "SELECT",
            "PRINT",
            "EXPORT",
            "USER",
            "LICENSE",
            "Situation",
            "Structure",
            "Constraints",
            "System",
            "Loading",
            "Modification",
            "Function",
            "Data",
            "Homogen.",
            "Sandwich",
            "Laminated",
            "Gasket",
            "Fluid",
            "Geometry",
            "Shape",
            "Misc",
            "Flange",
            "Membrane",
            "Solid",
            "Beam",
            "Shell",
            "Gasket",
            "Spring",
            "Mass",
            "Damper",
            "Viscoelastic Solid",
            "Load Carrying",
            "Membrane Elements",
            "Special",
            "Special CA",
            "Plot",
            "Convection",
            "Fluid",
            "Scalar",
            "2D-Infinite",
            "3D-Infinite",
            "3D-RBC (B-T)",
            "3D-RBC (E-M)",
            "axisymmetric",
            "Flanges",
            "Membranes",
            "Solids",
            "Shells",
            "Springs",
            "Fluids",
            "Infinite",
            "PERMAS",
            "MEDINA",
            "HyperView",
            "I-DEAS",
            "PATRAN",
            "DADS",
            "VAO",
            "Virtual.Lab",
            "ADAMS",
            "EXCITE",
            "FIRST",
            "MOTIONSOLVE",
            "SIMDRIVE3D",
            "SIMPACK",
            "Appendix",
            "Index"])


do_break = False


with open(UM450, "rt", encoding="utf-8") as um450:
    while True:
        last_pos, _k, _o, line = readline(um450)

        if not line: # EOF
            break

        if _o != -1:
            # print(f"{offset = }")
            offset = _o

        # print(line[offset:])

        if _k != "":
            keyword = _k
            # print(f"{keyword = }")
            continue

        _line = stripline(line)
        if _line in ("Contents", "Table of Contents"):
            title = _line
            TOC.add(title)
            if title not in parsed.keys():
                parsed.setdefault(title, {"_name": title, "_contents": ""})
            p = parsed[title]
            continue

        elif _line in skip:
            continue

        elif _line in TOC:
            if is_subtitle(_line, title):
                continue
            elif title == _line:
                continue
            title = _line
            print(f"{title = }")
            # if title.startswith("2 "):
            #     breakpoint()
            keyword = ""
            if title in ("UCI-Index", "DAT-Index", "Global Index"):
                number = [title]
            else:
                number = title.split(" ")[0].split(".")
            p = parsed
            for k in number:
                p = p[k]
            continue

        elif title == "Table of Contents":
            if _line != "":
                if ". ." in line:
                    _line = re.split(_RE_TOC_DOTS, line.strip())
                else:
                    _line = re.split(_RE_TOC_SPACE, line.strip())
                # print(f"{stripline(_line[0]) = }")
                _title = stripline(_line[0])
                TOC.add(_title)

                if _title in ("UCI-Index", "DAT-Index", "Global Index"):
                    number = [_title]
                else:
                    number = _title.split(" ")[0].split(".")

                pp = parsed
                for k in number[:-1]:
                    pp = pp[k]
                if number[-1] not in pp.keys():
                    pp.setdefault(number[-1], {"_name": _title, "_contents": ""})

        if keyword != "":
            if keyword not in p.keys():
                p.setdefault(keyword, "")
            p[keyword] += line[offset:]
        else:
            p["_contents"] += line[offset:]


for key in ("Contents", "Table of Contents", "UCI-Index", "DAT-Index", "Global Index"):
    parsed.pop(key)

ret = dict_to_md(parsed)

with open(os.path.splitext(UM450)[0] + ".md", "wt", encoding="utf-8") as um450:
    um450.write(ret)

with open(os.path.splitext(UM450)[0] + ".json", "wt", encoding="utf-8") as um450:
    um450.write(json.dumps(parsed, indent=2))


