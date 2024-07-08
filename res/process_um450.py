#!/usr/bin/env python3.12

import os
import sys
import re
import json

_RE_TOC = re.compile(r"^\s*(?P<number>[A-Z0-9]+(\.\d+)*)\s+(?P<title>([a-zA-Z0-9\-\/()]+[ .])+)\s+(?P<delim>(\. )+\.)?\s+(?P<page>[a-zA-Z0-9\-]+)\s*$")
_RE_TITLE = re.compile(r"^\s*(?P<number>[A-Z0-9]([.\-]\d+)*)\s+(?P<title>([a-zA-Z0-9()\-\/]+ ?)+)\s*$")
_RE_FOOTER_L = re.compile(r"^\s*(?P<version>V \d+\.\d+\.\d+)\s+(?P<permas>([a-zA-Z0-9'’()]+ )+)\s+(?P<page>[a-zA-Z0-9\-.]+)\s*$")
_RE_FOOTER_R = re.compile(r"^\s*(?P<page>[a-zA-Z0-9\-.]+)\s+(?P<permas>([a-zA-Z0-9'’()]+ )+)\s+(?P<version>V \d+\.\d+\.\d+)\s*$")

UM450 = "./um450_edu_v19_trial02_poppler_layout.txt"

def dict_to_md(d: dict, level: int = 0) -> str:
    ret = ""
    if "_name"  in d.keys():
        ret = "#" * level + " " + d["_name"] + "\n\n"
    if "_contents"  in d.keys():
        while "\n\n\n" in d["_contents"]:
            d["_contents"] = d["_contents"].replace("\n\n\n", "\n\n")
        ret += d["_contents"]

    keys = list([k for k in d.keys() if not k.startswith("_")])
    for k in keys:
        if isinstance(d[k], str):
            ret += "#" * (level + 1) + " " + k + "\n\n"
            while "\n\n\n" in d[k]:
                d[k] = d[k].replace("\n\n\n", "\n\n")
            ret += d[k] + "\n"
        else:
            ret += dict_to_md(d[k], level + 1)

    return ret + "\n"

TOC = set()
parsed = {"Introduction": {"_name": "Introduction", "_contents": ""}}
p = parsed["Introduction"]
title = "Introduction"
uci = False
keyword = None
_number = ""
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
        line = um450.readline()

        # if line.strip() == "Please refer to examples DEVX6V2, DEVX6V3 and DRA16 of [um550].":
        #     do_break = True

        if not line:
            break

        _line = line
        while "  " in _line:
            _line = _line.replace("  ", " ")
        _line = _line.strip()

        # if "global index" in _line.lower():
        #     do_break = True

        if line.startswith(""):
            print(f"{line = }")
            do_break = True
            # continue

        if do_break:
            breakpoint()

        if line.strip() in skip:
            continue

        if _number == "Global Index":
            title = [k for k in parsed.keys() if not k.startswith("_")][0]

        if _line == "":
            pass

        elif re.match(_RE_FOOTER_L, line) or re.match(_RE_FOOTER_R, line):
            line = um450.readline()
            for t in TOC:
                if t in line:
                    keyword = line.replace(t, "").strip().upper()
                    if keyword == "":
                        keyword = None
                    elif keyword in ("STARTUP", "INPUT", "EXEC", "SELECT", "PRINT", "EXPORT", "USER", "LICENSE"):
                        keyword = None
                    break
            else:
                keyword = None

            if do_break:
                breakpoint()

            last_pos = um450.tell()
            offset = 20
            for i in range(10):
                line = um450.readline()
                if line[:150].strip() == "" and line.strip() != "":
                    skip.add(line.strip())
                    continue
                elif line.strip() in skip:
                    continue
                elif line.lstrip() != "":
                    offset = min(offset, len(line) - len(line.lstrip()))
            # if offset < 8:
            #     offset = 0
            # else:
            #     offset = 8

            um450.seek(last_pos)

            continue

        elif _line in ("Contents", "Table of Contents"):
            title = _line
            parsed.setdefault(title, {"_name": title, "_contents": ""})
            p = parsed[title]
            continue

        elif _line in TOC:
            if title.startswith(_line.split(" ")[0]):
                continue
            title = _line
            if title in ("UCI-Index", "DAT-Index", "Global Index"):
                number = [title]
            else:
                number = title.split(" ")[0].split(".")
            uci = number[0] == "2"
            p = parsed
            for k in number:
                p = p[k]
            continue

        elif title == "Table of Contents":
            while " . " in _line:
                _line = _line.replace(" . ", " ")
            _line = _line.split(" ")
            if "Global Index" in line:
                _number = "Global Index"
                _title = ""
            else:
                _number = _line[0]
                _title = " ".join(_line[1:-1])
            _page = _line[-1]
            if _title == "":
                TOC.add(_number)
                parsed.setdefault(_number, {"_name": _number, "_contents": ""})
            else:
                print(f"TOC: " + _number + " " + _title)
                TOC.add(_number + " " + _title)
                number = _number.split(".")
                pp = parsed
                for k in number[:-1]:
                    pp = pp[k]
                pp.setdefault(number[-1], {"_name": _number + " " + _title, "_contents": ""})

        if keyword is not None:
            if keyword.upper() not in p.keys():
                p.setdefault(keyword.upper(), "")
            if line[:offset].strip() == "":
                p[keyword.upper()] += line[offset:].rstrip() + "\n"
            else:
                p[keyword.upper()] += line.rstrip() + "\n"

        else:
            if line[:offset].strip() == "":
                p["_contents"] += line[offset:].rstrip() + "\n"
            else:
                p["_contents"] += line.rstrip() + "\n"

for key in ("Contents", "Table of Contents", "UCI-Index", "DAT-Index", "Global Index"):
    parsed.pop(key)

ret = dict_to_md(parsed)

with open(os.path.splitext(UM450)[0] + ".md", "wt", encoding="utf-8") as um450:
    um450.write(ret)

with open(os.path.splitext(UM450)[0] + ".json", "wt", encoding="utf-8") as um450:
    um450.write(json.dumps(parsed, indent=2))



