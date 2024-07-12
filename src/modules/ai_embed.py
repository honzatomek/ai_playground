#!/usr/bin/env python3.12

# https://ollama.com/blog/embedding-models

import os
import sys
import re
import json
import numpy as np
from typing import List, Union
from numpy.typing import ArrayLike as Array

import ollama
from ollama import Client
import faiss

_RE_MD_LEVEL = re.compile(r"^(?P<level>#+)\s+(?P<title>.*)$")

MODEL_AI = "llama3"
MODEL_EMBEDDINGS = "mxbai-embed-large"



class Embeddings:
    """class for AI embeddings"""

    @staticmethod
    def _create_index(embeddings: Array,
                      index_dtype: str = "IP",
                      embed_dtype: type = np.float32) -> faiss.IndexFlat:
        """create a searchable vector database using faiss module

        Args:
            embeddings (Array): an array of vectors
            index_dtype (str):  the database type to use (IP/L2, default = IP)
                                IP = Inner Product
                                L2 = Euclidean Norm
            embed_dtype (type): vector type to use (default = np.float32)

        Raises:
            ValueError: if the index_dtype is neither IP nor L2

        Returns:
            (faiss.IndexFlat) a searchable vector database
        """
        embeddings = np.array(embeddings, dtype=embed_dtype)
        if index_dtype == "IP":
            index = faiss.IndexFlatIP(embeddings.shape[1])
        elif index_dtype == "L2":
            index = faiss.IndexFlatL2(embeddings.shape[1])
        else:
            raise ValueError(f"Index type {index_dtype:s} not recognised, must be either L2 or IP")
        index.add(embeddings)
        return index

    @staticmethod
    def distance_l2(v1: Union[List[float] | Array],
                    v2: Union[List[float] | Array]) -> Union[float | Array]:
        """Euclidean distance between two vectors

        d = ( Σ (v_2_i - v_1_i) ^ 2) ^ 0.5

        Args:
            v1 (list): first vector
            v2 (list): second vector, can be a list of vectors

        Raises:
            ValueError: if len(v1) != len(v2)

        Returns:
            (float | list): distance between the two vectors, list of distances if v2 is array
        """
        v1 = np.array(v1).flatten()
        v2 = np.array(v2)

        if v1.shape[0] != v2.shape[-1]:
            raise ValueError(f"Length of vectors must be equal ({v1.shape[0]:d} != {v2.shape[-1]:d}).")

        if len(v2.shape) == 2:
            return np.linalg.norm(v2 - v1, axis=1)
        else:
            return np.linalg.norm(v2 - v1)

    @staticmethod
    def distance_cos(v1: Union[List[float] | Array],
                     v2: Union[List[float] | Array]) -> Union[float | Array]:
        """Cosine distance between two vectors (1 - cosine similarity) => 0. means similar

        d = 1. - (v1 . v2) / (|v1| |v2|)

        Args:
            v1 (list): first vector
            v2 (list): second vector, can be a list of vectors

        Raises:
            ValueError: if len(v1) != len(v2)

        Returns:
            (float): distance between the two vectors, list of distances if v2 is array
        """
        v1 = np.array(v1).flatten()
        v2 = np.array(v2)

        if v1.shape[0] != v2.shape[-1]:
            raise ValueError(f"Length of vectors must be equal ({v1.shape[0]:d} != {v2.shape[-1]:d}).")

        if len(v2.shape) == 2:
            return 1. - np.dot(v2, v1) / (np.linalg.norm(v1) * np.linalg.norm(v2, axis=1))
        else:
            return 1. - np.dot(v2, v1) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    @staticmethod
    def distance_mac(v1: Union[List[float] | Array],
                     v2: Union[List[float] | Array]) -> Union[float | Array]:
        """MAC value between two vectors (1 - MAC) => 0. means similar

        d = 1. - (v1 . v2)^2 / ((v1 . v1) * (v2 . v2))

        Args:
            v1 (list): first vector
            v2 (list): second vector, can be a list of vectors

        Raises:
            ValueError: if len(v1) != len(v2)

        Returns:
            (float): distance between the two vectors, list of distances if v2 is array
        """
        v1 = np.array(v1).flatten()
        v2 = np.array(v2)

        if v1.shape[0] != v2.shape[-1]:
            raise ValueError(f"Length of vectors must be equal ({v1.shape[0]:d} != {v2.shape[-1]:d}).")

        if len(v2.shape) == 2:
            return 1. - np.dot(v2, v1) ** 2. / (np.dot(v1, v1) * np.dot(v2, v2).diagonal())
        else:
            return 1. - np.dot(v2, v1) ** 2. / (np.dot(v1, v1) * np.dot(v2, v2))

    @staticmethod
    def distance_ip(v1: Union[List[float] | Array],
                    v2: Union[List[float] | Array]) -> Union[float | Array]:
        """Inner Product between two vectors, the component of v1 in direction of v2

        d = (v2 . v1)

        Args:
            v1 (list): first vector
            v2 (list): second vector, can be a list of vectors

        Raises:
            ValueError: if len(v1) != len(v2)

        Returns:
            (float): distance between the two vectors, list of distances if v2 is array
        """
        v1 = np.array(v1).flatten()
        v2 = np.array(v2)

        if v1.shape[0] != v2.shape[-1]:
            raise ValueError(f"Length of vectors must be equal ({v1.shape[0]:d} != {v2.shape[-1]:d}).")

        return np.dot(v2, v1)

    @classmethod
    def closest_l2(cls,
                   vec: Union[List[float] | Array],
                   arr: Union[List[List[float]], Array],
                   number: int = 1) -> (Array, Array):
        """Return the index of n closest vectors of arr to vec using L2 norm

        Args:
            vec (list):   vector to look for closest vector to
            arr (list):   list of vector to search the closest vectors from
            number (int): number of closest indexes to return

        Raises:
            ValueError: if length of vectors is not the same

        Returns:
            (list, list): a list of distances, a list of indexes
        """
        v = np.array(vec).flatten()
        a = np.array(arr)
        if v.shape[0] != arr.shape[1]:
            raise ValueError(f"Length of vectors must be equal ({v.shape[0]:d} != {a.shape[1]:d}).")
        # compute distances between vectors
        d   = cls.distance_l2(v, a)
        idx = np.argsort(d)[:number]
        return d[idx], idx

    @classmethod
    def closest_cos(cls,
                    vec: Union[List[float] | Array],
                    arr: Union[List[List[float]], Array],
                    number: int = 1) -> (Array, Array):
        """Return the index of n closest vectors of arr to vec using (1 - cosine similarity)

        Args:
            vec (list):   vector to look for closest vector to
            arr (list):   list of vector to search the closest vectors from
            number (int): number of closest indexes to return

        Raises:
            ValueError: if length of vectors is not the same

        Returns:
            (list, list): a list of distances, a list of indexes
        """
        v = np.array(vec).flatten()
        a = np.array(arr)
        if v.shape[0] != arr.shape[1]:
            raise ValueError(f"Length of vectors must be equal ({v.shape[0]:d} != {a.shape[1]:d}).")
        # compute distances between vectors
        d   = cls.distance_cos(v, a)
        idx = np.argsort(d)[:number]
        return d[idx], idx

    @classmethod
    def closest_mac(cls,
                    vec: Union[List[float] | Array],
                    arr: Union[List[List[float]], Array],
                    number: int = 1) -> (Array, Array):
        """Return the index of n closest vectors of arr to vec using MAC value

        Args:
            vec (list):   vector to look for closest vector to
            arr (list):   list of vector to search the closest vectors from
            number (int): number of closest indexes to return

        Raises:
            ValueError: if length of vectors is not the same

        Returns:
            (list, list): a list of distances, a list of indexes
        """
        v = np.array(vec).flatten()
        a = np.array(arr)
        if v.shape[0] != arr.shape[1]:
            raise ValueError(f"Length of vectors must be equal ({v.shape[0]:d} != {a.shape[1]:d}).")
        # compute distances between vectors
        d   = cls.distance_mac(v, a)
        idx = np.argsort(d)[-number:][::-1]
        return d[idx], idx

    @classmethod
    def closest_ip(cls,
                   vec: Union[List[float] | Array],
                   arr: Union[List[List[float]], Array],
                   number: int = 1) -> (Array, Array):
        """Return the index of n closest vectors of arr to vec using Inner Product value

        Args:
            vec (list):   vector to look for closest vector to
            arr (list):   list of vector to search the closest vectors from
            number (int): number of closest indexes to return

        Raises:
            ValueError: if length of vectors is not the same

        Returns:
            (list, list): a list of distances, a list of indexes
        """
        v = np.array(vec).flatten()
        a = np.array(arr)
        if v.shape[0] != arr.shape[1]:
            raise ValueError(f"Length of vectors must be equal ({v.shape[0]:d} != {a.shape[1]:d}).")
        # compute distances between vectors
        d   = cls.distance_ip(v, a)
        idx = np.argsort(d)[-number:][::-1]
        return d[idx], idx

    @classmethod
    def search_closest(cls,
                       vec: Union[List[float] | Array],
                       arr: Union[List[List[float]], Array],
                       number: int = 1) -> (Array, Array):
        """Return the index of n closest vectors of arr to vec using a combination
        of L2 norm and (1 - cosine similarity)

        Args:
            vec (list):   vector to look for closest vector to
            arr (list):   list of vector to search the closest vectors from
            number (int): number of closest indexes to return

        Raises:
            ValueError: if length of vectors is not the same

        Returns:
            (list, list): a list of distances, a list of indexes
        """
        d_l2,  idx_l2  = cls.closest_l2(vec, arr, number)
        d_cos, idx_cos = cls.closest_cos(vec, arr, number)
        d_mac, idx_mac = cls.closest_mac(vec, arr, number)
        d_ip,  idx_ip  = cls.closest_ip(vec, arr, number)

        d   = []
        idx = []
        for i in range(number):
            for dd, iidx in zip([d_l2, d_cos, d_mac, d_ip], [idx_l2, idx_cos, idx_mac, idx_ip]):
                if iidx[i] not in idx:
                    d.append(dd[i])
                    idx.append(iidx[i])
                if len(idx) >= number:
                    break
        return np.array(d[:number]), np.array(idx[:number])

    @staticmethod
    def embed_prompt(prompt: str,
                     model: str = MODEL_EMBEDDINGS,
                     dtype = np.float32) -> np.ndarray:
        response = ollama.embeddings(model=model, prompt=prompt)
        return np.array(response["embedding"], dtype=dtype).flatten()

    @classmethod
    def from_json(cls,
                  filename: str,
                  embed_model: str = MODEL_EMBEDDINGS,
                  index_dtype: str = "IP",
                  embed_dtype: type = np.float32) -> object:
        print(f"[+] reading embeddings model from json file: {filename:s}.")
        dtb = {}
        with open(os.path.realpath(filename), "rt", encoding="utf-8") as txt:
            dtb = json.loads(txt.read())
        return cls(dtb, embed_model, index_dtype, embed_dtype)

    @classmethod
    def from_md(cls,
                filename: str,
                embed_model: str = MODEL_EMBEDDINGS,
                index_dtype: str = "IP",
                embed_dtype: type = np.float32) -> object:
        print(f"[+] reading markdown file: {filename:s}.")

        # read and process the file to a flattened dictionary
        dtb = {}
        title = "root"
        level = 0
        nlevel = None
        levelstr = ""
        path = ["root"]
        mdpath = ["root"]
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
                                mdpath.pop()
                                level -= 1

                        elif nlevel > level:
                            while level < nlevel:
                                path.append("__dummy__")
                                mdpath.append("__dummy__")
                                level += 1

                        path[level] = title
                        mdpath[level] = line.strip()

                        title = "#".join(path)
                        if title not in dtb.keys():
                            dtb.setdefault(title, "\n".join(mdpath) + "\n\n")
                        continue

                dtb[title] += line

        # cleanup a bit
        for title in dtb.keys():
            while "\n\n\n" in dtb[title]:
                dtb[title] = dtb[title].replace("\n\n\n", "\n\n")

        return cls(dtb, embed_model, index_dtype, embed_dtype)

    def __init__(self,
                 str_dict: dict,
                 embed_model: str = MODEL_EMBEDDINGS,
                 index_dtype: str = "IP",
                 embed_dtype: type = np.float32):
        """constructor

        Args:
            str_dict (dict):    a flat dictionary of strings to be search using embedding
            embed_model (str):  the LLM to be used for embedding both the search strings and the queries
            index_dtype (str):  the method to use for searching the closest strings to query
                                L2  - Euclidean distance (faiss module can be used by creating index)
                                IP  - Inner Product (faiss module can be used by creating index)
                                cos - Cosine distance (1 - cosine similarity)
                                MAC - MAC value
            embed_dtype (type): the type to use for storing the embeddings (default = np.float32)
        """
        self.keys       = list(str_dict.keys())
        self.embed_model = embed_model
        self.index_dtype = index_dtype
        self.embed_dtype = embed_dtype
        self.index       = None

        if isinstance(str_dict[self.keys[0]], dict):
            self.dict  = {k: str_dict[k]["_contents"]  for k in self.keys}
            keys       = []
            embeddings = []
            for k, v in self.dict.items():
                if "_embedding" in v.keys():
                    keys.append(k)
                    embeddings.append(v["_embedding"])
            self.keys       = keys
            self.embeddings = np.array(embeddings, dtype=self.embed_dtype)

        else:
            self.dict       = dict(str_dict)
            self.embeddings = None
            # self.embed_all()

    def create_index(self):
        """create a searchable database of embedded vectors"""
        self.index = self._create_index(self.embeddings, self.index_dtype, self.embed_dtype)

    def embed_all(self):
        """create an embedding of the flat dictionary, one key each"""
        embeddings = []
        num = len(str(len(self.keys)))
        fmt = "{0:" + str(num) + "d}"
        keys = []
        for i, (k, v) in enumerate(self.dict.items()):
            print("[i] Embedding " + fmt.format(i+1) + f"/{len(self.keys):d} ({k:s})")

            _v = "\n".join([vv for vv in v.split("\n") if not vv.startswith("#")]).strip()

            if _v == "":
                print("    Empty string, skipping...")
                continue

            keys.append(k)

            embeddings.append(self.embed_prompt(v, self.embed_model, self.embed_dtype))

        self.keys = keys
        self.embeddings = np.array(embeddings, dtype=self.embed_dtype)

    def save_json(self, filename: str):
        """save the embeddings including the flat dictionary to a file in a json format

        Args:
            filename (str): the filename to save the json to (will be overwritten)
        """
        print(f"[+] writing embeddings model to json file: {filename:s}.")
        out_dict = {}
        for k in self.dict.keys():
            try:
                i = self.keys.index(k)
                out_dict.setdefault(k, {
                                        "_contents":  self.dict[k],
                                        "_embedding": self.embeddings[i].flatten().tolist(),
                                        })
            # key is not in self.keys
            except ValueError as e:
                out_dict.setdefault(k, {"_contents":  self.dict[k]})

        with open(os.path.realpath(filename), "wt", encoding="utf-8") as txt:
            txt.write(json.dumps(out_dict, indent=2))

    def search(self, query: str, number: int = 1) -> List[str]:
        """search the embeddings for n closest vectors and return their respective strings

        If an index was not created using the create_index() method, the closest
        vectors will be found using numpy.

        Args:
            query (str):  the string to search for
            number (int): the number n of the closest to return

        Raises:
            ValueError: if model has not been embedded yet

        Returns:
            (list): a list of n closest strings to the query
        """
        equery = self.embed_prompt(query, self.embed_model, self.embed_dtype).reshape(1, -1)
        if self.index is not None:
            D, I = self.index.search(equery, number)
            D, I = D[0], I[0]
        else:
            if self.embeddings is None:
                raise ValueError("Model needs to be embedded first.")
            if self.index_dtype == "L2":
                D, I = self.closest_l2(equery, self.embeddings, number)
            elif self.index_dtype == "IP":
                D, I = self.closest_ip(equery, self.embeddings, number)
            elif self.index_dtype == "MAC":
                D, I = self.closest_mac(equery, self.embeddings, number)
            elif self.index_dtype == "cos":
                D, I = self.closest_cos(equery, self.embeddings, number)
            else:
                D, I = self.search_closest(equery, self.embeddings, number)
        return [self.dict[self.keys[i]] for i in I]


if __name__ == "__main__":
    # e = Embeddings.from_json("../../res/um450_edu_v19.json", index_dtype="IP")
    e = Embeddings.from_md("../../res/um450_edu_v19.md", index_dtype="IP")
    e.embed_all()
    e.save_json("./tmp.json")
    # e.create_index()
    while True:
        prompt = input("Ask PERMAS manual (q for quit) > ")
        if prompt.lower() in ("", "q", "quit", "e", "exit"):
            break
        for i, response in enumerate(e.search(prompt, 10)):
            print(f"[+] Response {i+1:d} " + "=" * 100)
            print(response)


