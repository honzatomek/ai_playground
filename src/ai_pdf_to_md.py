#!/usr/bin/env python3.12

import version

__doc__ = f"""Script to convert PDF page by page to markdown
author:  {version.__author__:s}
version: {version.__version__:s}
date:    {version.__date__:s}

description:
Script to use AI to extract data from pdf page by page using images of the pages.
The AI then should convert these images to a markdown format.
"""

import os
import sys
import io
import argparse
import base64
import pdfplumber
import pytesseract
import PIL
from PIL import Image

import ollama
from ollama import Client


# TODO:
# try llava-llama3
AI_MODEL = "llama3"


class CheckFile(argparse.Action):
    """Argparse action to check whether the file exists"""
    def __call__(self, parser, namespace, fname, option_string=None):
        if not os.path.isfile(fname):
            parser.error(f"{fname:s} does not exist.")
        else:
            setattr(namespace, self.dest, fname)


def pdf_pages(pdffilename: str) -> Image:
    """Generator to split pdf file into pages and return them as images"""
    with pdfplumber.open(os.path.realpath(pdffilename)) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                yield text
            else:
                print(f"[-] No text found on page {page_num:d}, attempting OCR.")
                page_image = page.to_image(resolution=300).original
                text = pytesseract.image_to_string(page_image)
                yield text


def encode_image(image: Image) -> str:
    """Function to take a Image object and encode it as string in base64"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return str(img_str)


def convert_pdf_to_md(pdffilename: str, outfilename: str):
    client = Client(host="http://localhost:11434")
    system = {"role": "system",
              "content": """You are a helpful assistant. You will be supplied with
an image encoded as bytes64 srting. Extract the text that you will find in the image
and return it in github markdown format. Return only the extracted text.
If you find a mathematic equation, convert it to a block in latex format.
Do not add additional notes or explanations.

The PNG Image:
"""}

    # with open(outfilename, "wt", encoding="utf-8") as md:
    breakpoint()
    for page in pdf_pages(pdffilename):
        user = {"role": "user",
                "content": encode_image(page)}
        response = client.chat(model=AI_MODEL, messages=[system, user])
        print(response["message"]["content"])


if __name__ == "__main__":
    # Create options parser object.
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)

    # Add arguments
    parser.add_argument("-v", "--verbose", dest="verbose", action="count", default=0,
                        help="""Increase verbosity.""")

    parser.add_argument("pdffile", type=str, action=CheckFile,
                        help="""PDF file to convert to *.md file""")

    # Parse command-line arguments.
    args = parser.parse_args()

    # Prepare output filename
    outfile = os.path.splitext(args.pdffile)[0] + ".md"

    # Run the script
    convert_pdf_to_md(args.pdffile, outfile)


