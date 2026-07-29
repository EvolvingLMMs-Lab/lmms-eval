# Taken and modified from https://github.com/harvardnlp/im2markup.
# Copyright (c) 2016 Harvard NLP; licensed under MIT.
# See LICENSE-IM2MARKUP in this directory.
# tokenize latex formulas
import argparse
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from threading import Timer


def run_cmd(cmd, timeout_sec=30):
    proc = subprocess.Popen(cmd, shell=True)
    kill_proc = lambda p: p.kill()
    timer = Timer(timeout_sec, kill_proc, [proc])
    try:
        timer.start()
        stdout, stderr = proc.communicate()
    finally:
        timer.cancel()


def tokenize_latex(latex_code, latex_type="", middle_file=""):
    if not latex_code:
        return False, latex_code
    if not latex_type:
        latex_type = "tabular" if "tabular" in latex_code else "formula"
    if not middle_file:
        middle_file = "out-" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".txt"
    temp_file = middle_file + ".tmp"

    if latex_type == "formula":
        with open(temp_file, "w") as f:
            prepre = latex_code
            # replace split, align with aligned
            prepre = re.sub(r"\\begin{(split|align|alignedat|alignat|eqnarray)\*?}(.+?)\\end{\1\*?}", r"\\begin{aligned}\2\\end{aligned}", prepre, flags=re.S)
            prepre = re.sub(r"\\begin{(smallmatrix)\*?}(.+?)\\end{\1\*?}", r"\\begin{matrix}\2\\end{matrix}", prepre, flags=re.S)
            f.write(prepre)

        cmd = r"cat %s | node %s %s > %s " % (temp_file, os.path.join(os.path.dirname(__file__), "preprocess_formula.js"), "normalize", middle_file)
        try:
            ret = subprocess.call(cmd, shell=True, timeout=60)
        except subprocess.TimeoutExpired:
            ret = 1
        if os.path.exists(temp_file):
            os.remove(temp_file)
        if ret != 0:
            return False, latex_code

        operators = "\s?".join(
            "|".join(
                [
                    "arccos",
                    "arcsin",
                    "arctan",
                    "arg",
                    "cos",
                    "cosh",
                    "cot",
                    "coth",
                    "csc",
                    "deg",
                    "det",
                    "dim",
                    "exp",
                    "gcd",
                    "hom",
                    "inf",
                    "injlim",
                    "ker",
                    "lg",
                    "lim",
                    "liminf",
                    "limsup",
                    "ln",
                    "log",
                    "max",
                    "min",
                    "Pr",
                    "projlim",
                    "sec",
                    "sin",
                    "sinh",
                    "sup",
                    "tan",
                    "tanh",
                ]
            )
        )
        ops = re.compile(r"\\operatorname {(%s)}" % operators)
        with open(middle_file, "r") as fin:
            for line in fin:
                tokens = line.strip().split()
                tokens_out = []
                for token in tokens:
                    tokens_out.append(token)
                post = " ".join(tokens_out)
                # use \sin instead of \operatorname{sin}
                names = ["\\" + x.replace(" ", "") for x in re.findall(ops, post)]
                post = re.sub(ops, lambda match: str(names.pop(0)), post).replace(r"\\ \end{array}", r"\end{array}")
        os.remove(middle_file)
        return True, post

    elif latex_type == "tabular":
        latex_code = latex_code.replace("\\\\%", "\\\\ %")
        latex_code = latex_code.replace("\%", "<PERCENTAGE_TOKEN>")
        latex_code = latex_code.split("%")[0]
        latex_code = latex_code.replace("<PERCENTAGE_TOKEN>", "\%")
        if not "\\end{tabular}" in latex_code:
            latex_code += "\\end{tabular}"
        with open(middle_file, "w") as f:
            f.write(latex_code.replace("\r", " ").replace("\n", " "))
        cmd = "perl -pe 's|hskip(.*?)(cm\\|in\\|pt\\|mm\\|em)|hspace{\\1\\2}|g' %s > %s" % (middle_file, temp_file)
        try:
            ret = subprocess.call(cmd, shell=True, timeout=60)
        except subprocess.TimeoutExpired:
            ret = 1
        if ret != 0:
            return False, latex_code
        os.remove(middle_file)
        cmd = r"cat %s | node %s %s > %s " % (temp_file, os.path.join(os.path.dirname(__file__), "preprocess_tabular.js"), "tokenize", middle_file)
        try:
            ret = subprocess.call(cmd, shell=True, timeout=60)
        except subprocess.TimeoutExpired:
            ret = 1
        if os.path.exists(temp_file):
            os.remove(temp_file)
        if ret != 0:
            return False, latex_code
        with open(middle_file, "r") as fin:
            for line in fin:
                tokens = line.strip().split()
                tokens_out = []
                for token in tokens:
                    tokens_out.append(token)
                post = " ".join(tokens_out)
        os.remove(middle_file)
        return True, post
    else:
        print(f"latex type{latex_type} unrecognized.")
        return False, latex_code


if __name__ == "__main__":
    latex_code = open("2.txt", "r").read().replace("\r", " ")
    print("=>", latex_code)
    new_code = tokenize_latex(latex_code)
    print("=>", new_code)
