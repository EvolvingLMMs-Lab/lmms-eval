import json
import logging
import os
import re
import shutil

import numpy as np
from PIL import Image

SKIP_PATTERNS = [r"\{", r"\}", r"[\[\]]", r"\\begin\{.*?\}", r"\\end\{.*?\}", r"\^", r"\_", r"\\.*rule.*", r"\\.*line.*", r"\[[\-.0-9]+[epm][xtm]\]"]
SKIP_Tokens = [
    "\\",
    "\\\\",
    "\\index",
    "\\a",
    "&",
    "$",
    "\\multirow",
    "\\def",
    "\\raggedright",
    "\\url",
    "\\cr",
    "\\ensuremath",
    "\\left",
    "\\right",
    "\\mathchoice",
    "\\scriptstyle",
    "\\displaystyle",
    "\\qquad",
    "\\quad",
    "\\,",
    "\\!",
    "~",
    "\\boldmath",
]
PHANTOM_Tokens = ["\\fontfamily", "\\vphantom", "\\phantom", "\\rowcolor", "\\ref"]
TWO_Tail_Tokens = ["\\frac", "\\binom"]
AB_Tail_Tokens = ["\\xrightarrow", "\\xleftarrow", "\\sqrt"]  # special token \xxx [] {}
TWO_Tail_Invisb_Tokens = ["\\overset", "\\underset", "\\stackrel"]
ONE_Tail_Tokens = ["\\widetilde", "\\overline", "\\hat", "\\widehat", "\\tilde", "\\Tilde", "\\dot", "\\bar", "\\vec", "\\underline", "\\underbrace", "\\check", "\\breve", "\\Bar", "\\Vec", "\\mathring", "\\ddot"]
ONE_Tail_Invisb_Tokens = [
    "\\boldsymbol",
    "\\pmb",
    "\\textbf",
    "\\mathrm",
    "\\mathbf",
    "\\mathbb",
    "\\mathcal",
    "\\textmd",
    "\\texttt",
    "\\textnormal",
    "\\text",
    "\\textit",
    "\\textup",
    "\\mathop",
    "\\mathbin",
    "\\smash",
    "\\operatorname",
    "\\textrm",
    "\\mathfrak",
    "\\emph",
    "\\textsf",
    "\\textsc",
]


def flatten_multiline(latex):
    brace_map = {
        "\\left(": "\\right)",
        "\\left[": "\\right]",
        "\\left{": "\\right}",
    }
    l_split = latex.split(" ")
    if l_split[0] == "\\begin{array}":
        if l_split[-1] == "\\end{array}":
            l_split = l_split[2:-1]
        else:
            l_split = l_split[2:]

    idx = 0
    while idx < len(l_split):
        token = l_split[idx]
        if token.startswith("\\left") and token in brace_map.keys():
            end_idx = find_matching_brace(l_split, idx, brace=[token, brace_map[token]])
            if end_idx != -1:
                idx = end_idx
        elif token in ["\\\\", "~", "\\qquad"]:
            l_split = l_split[0:idx] + l_split[idx + 1 :]
            idx -= 1
        idx += 1
    latex = " ".join(l_split)
    return "$ " + latex + " $"


def clean_latex(text):
    # TODO 让GPT写的去空格函数, 初步测了是没问题的, 不确定是否完全没有bug
    cleaned_text = re.sub(r"(?<=[^\\])\s+(?=[^\\])", "", text)
    # TODO 有一些不能去掉的空格给补充回来
    for item in ["\\hline", "\\midrule", "\\times", "\\bf", "\\footnotesize", "\\cr", "\\log"]:
        cleaned_text = cleaned_text.replace(item, item + " ")
    cleaned_text = cleaned_text.replace(" \\mathcolor{black}", "\\mathcolor{black}")
    return cleaned_text


def remove_trailing_latex(formula):
    pattern = r"(\\(hspace\*?\{[^{}]*?\}|vspace\*?\{[^{}]*?\}|smallskip|medskip|quad|qquad|bigskip|[;,])|\~|\.)*$"
    # Replace the matched pattern with an empty string
    cleaned_formula = re.sub(pattern, "", formula, count=1)
    return cleaned_formula


def find_matching_brace(sequence, start_index, brace=["{", "}"]):
    # Finds the index of the matching brace for the one at start_index
    left_brace, right_brace = brace
    depth = 0
    for i, char in enumerate(sequence[start_index:], start=start_index):
        if char == left_brace:
            depth += 1
        elif char == right_brace:
            depth -= 1
            if depth == 0:
                return i
    if depth > 0:
        error_info = "Warning! found no matching brace in sequence !"
        raise ValueError(error_info)
    return -1


def normalize_latex(l, rm_trail=False):
    if "tabular" in l:
        latex_type = "tabular"
    else:
        latex_type = "formula"

    if rm_trail:
        l = remove_trailing_latex(l)
    l = l.strip().replace(r"\pmatrix", r"\mypmatrix").replace(r"\matrix", r"\mymatrix")

    # TODO \raggedright \arraybackslash, these align method, difficult to handle, remove it.
    for item in ["\\raggedright", "\\arraybackslash"]:
        l = l.replace(item, "")

    for item in ["\\lowercase", "\\uppercase"]:
        l = l.replace(item, "")

    # TODO \hspace {1 . 5 cm}, for formula, change to \hspace{1.5cm}, for table, remove it.
    pattern = r"\\[hv]space { [.0-9a-z ]+ }"
    old_token = re.findall(pattern, l, re.DOTALL)
    if latex_type == "tabular":
        new_token = ["" for item in old_token]
    else:
        new_token = [item.replace(" ", "") for item in old_token]
    for bef, aft in zip(old_token, new_token):
        l = l.replace(bef, aft)

    # TODO take \begin {tabular} {} as one token
    # TODO there are \begin{array} in table too，so the process should run in both formula and table.
    if latex_type == "tabular":
        l = l.replace("\\begin {tabular}", "\\begin{tabular}")
        l = l.replace("\\end {tabular}", "\\end{tabular}")
        l = l.replace("\\begin {array}", "\\begin{array}")
        l = l.replace("\\end {array}", "\\end{array}")
        l_split = l.split(" ")
        idx = 0
        while idx < len(l_split):
            token = l_split[idx]
            if token == "\\begin{tabular}":
                sub_idx = idx + 1
                end_idx = find_matching_brace(l_split, sub_idx)
                new_token = "".join(l_split[idx : end_idx + 1])
                l_split = l_split[0:idx] + [new_token] + l_split[end_idx + 1 :]
                break
            idx += 1
        l = " ".join(l_split)

        # TODO some complex format, hart to deal with re.match, so using brace match, such as：\cmidrule ( l { 3 p t } r { 3 p t } ) { 1 - 1 }
        l_split = l.split(" ")
        idx = 0
        while idx < len(l_split):
            token = l_split[idx]
            if token in ["\\cmidrule", "\\cline"]:
                sub_idx = idx + 1
                if l_split[sub_idx] == "(":
                    mid_end = find_matching_brace(l_split, sub_idx, brace=["(", ")"])
                    end_idx = find_matching_brace(l_split, mid_end + 1)
                else:
                    end_idx = find_matching_brace(l_split, sub_idx)
                new_token = "".join(l_split[idx : end_idx + 1])
                l_split = l_split[0:idx] + [new_token] + l_split[end_idx + 1 :]
            idx += 1
        l = " ".join(l_split)

    pattern = r"\\begin{array} { [lrc ]+ }"
    old_token = re.findall(pattern, l, re.DOTALL)
    new_token = [item.replace("\\begin{array} ", "<s>").replace(" ", "").replace("<s>", "\\begin{array} ") for item in old_token]
    for bef, aft in zip(old_token, new_token):
        l = l.replace(bef, aft)

    # TODO token such \not= should be one token
    pattern = r"\\not [<>+=\-]"
    old_token = re.findall(pattern, l, re.DOTALL)
    new_token = [item.replace(" ", "") for item in old_token]
    for bef, aft in zip(old_token, new_token):
        l = l.replace(bef, aft)

    # TODO tokens such as \dots \exp \sinh, split them to parts, so the bbox match will be easier.

    l = " " + l + " "
    l = l.replace(" \\ldots ", " . . . ")
    l = l.replace(" \\cdots ", " . . . ")
    l = l.replace(" \\dots ", " . . . ")
    l = l.replace(" \\dotsb ", " . . . ")
    l = l.replace(" \\log ", " \\mathrm { l o g } ")
    l = l.replace(" \\exp ", " \\mathrm { e x p } ")
    l = l.replace(" \\sin ", " \\mathrm { s i n } ")
    l = l.replace(" \\cos ", " \\mathrm { c o s } ")
    l = l.replace(" \\tan ", " \\mathrm { t a n } ")
    l = l.replace(" \\tanh ", " \\mathrm { t a n h } ")
    l = l.replace(" \\cosh ", " \\mathrm { c o s h } ")
    l = l.replace(" \\sinh ", " \\mathrm { s i n h } ")

    # ** token such as \big( should be one token
    pattern = r"\\[Bb]ig[g]?[glrm]? [(){}|\[\]] "
    old_token = re.findall(pattern, l, re.DOTALL)
    new_token = [item.replace(" ", "") for item in old_token]
    for bef, aft in zip(old_token, new_token):
        l = l.replace(bef, aft + " ")

    pattern = r"\\[Bb]ig[g]?[glrm]? \\.*? "
    old_token = re.findall(pattern, l, re.DOTALL)
    new_token = [item.replace(" ", "") for item in old_token]
    for bef, aft in zip(old_token, new_token):
        l = l.replace(bef, aft + " ")

    # TODO when \operatorname * meets mathcolor it comes error, yet the * is useless, so we simply remove it bynow.
    pattern = r"\\operatorname \*"
    old_token = re.findall(pattern, l, re.DOTALL)
    new_token = ["\\operatorname" for item in old_token]
    for bef, aft in zip(old_token, new_token):
        l = l.replace(bef, aft)

    # TODO \lefteqn will lead to letter overlap, it's harmfull for render, so simply remove it.
    l = l.replace("\\lefteqn", "")

    # TODO \footnote can not seem as ONE_Tail_Invisb_Tokens(usually this type token add color by \mathrm {\color(x)}, yet \footnode should be \color{\footnote{x}}), so we simple change it to "^".
    l = l.replace("\\footnote ", "^ ")

    # TODO \' can not be rendered separately(cause to different visulize performence), so we take these tokens as one token such as \' e -> \'e, on the other hand, if { after \' then render them separately.
    pattern = r"\\\' [^{] "
    old_token = re.findall(pattern, l, re.DOTALL)
    new_token = [item.replace(" ", "") for item in old_token]
    for bef, aft in zip(old_token, new_token):
        l = l.replace(bef, aft + " ")

    # TODO [ -1.5ex ] [ 1.5pt ] [ 3 mm ] some layout adjustment, no need to render. combine them as one token.
    if latex_type == "tabular":
        pattern = r"\[ [\-.0-9 ]+[exptcm ]+ \]"
        old_token = re.findall(pattern, l, re.DOTALL)
        new_token = [item.replace(" ", "") for item in old_token]
        for bef, aft in zip(old_token, new_token):
            l = l.replace(bef, aft)

    # ** \parbox { 3cm } {} shoudle be combined as one token
    pattern = r"\\parbox {[^{]+}"
    old_token = re.findall(pattern, l, re.DOTALL)
    new_token = [item.replace(" ", "") for item in old_token]
    for bef, aft in zip(old_token, new_token):
        l = l.replace(bef, aft)

    # ** \raisebox{<lift>}[<height>][<depth>] {} shoudle be combined as one token, \raisebox{-1.5ex}[0pt]
    pattern = r"\\raisebox {[^{]+} [\[\]0-9 exptcm]+{"
    old_token = re.findall(pattern, l, re.DOTALL)
    new_token = [item.replace(" ", "") for item in old_token]
    for bef, aft in zip(old_token, new_token):
        l = l.replace(bef, aft[0:-1] + " {")

    # ** \char shoudle be combined as one token
    pattern = r"{ \\char[0-9\' ]+}"
    old_token = re.findall(pattern, l, re.DOTALL)
    new_token = [item.replace(" ", "") for item in old_token]
    for bef, aft in zip(old_token, new_token):
        l = l.replace(bef, "{ " + aft[1:-1] + " }")

    # ** \not xx shoudle be combined as one token
    pattern = r"\\not [\\=\<\>][^ ]+ "
    old_token = re.findall(pattern, l, re.DOTALL)
    new_token = [item.replace(" ", "") for item in old_token]
    for bef, aft in zip(old_token, new_token):
        l = l.replace(bef, aft + " ")

    # ** \specialrule{1pt}{2pt}{2pt}, special lines, shoudle be combined as one token
    pattern = r"\\specialrule {[ .0-9a-z]+} {[ .0-9a-z]+} {[ .0-9a-z]+}"
    old_token = re.findall(pattern, l, re.DOTALL)
    new_token = [item.replace(" ", "") for item in old_token]
    for bef, aft in zip(old_token, new_token):
        l = l.replace(bef, aft)

    # ** for easier add color, the original color should be removed, there are two type of color for now: \color[rgb]{0, 1, 0} and \color{red}
    pattern = r"\\colorbox[ \[\]RGBrgb]+{ [A-Za-z 0-9,!]+ } |\\color[ \[\]RGBrgb]+{ [A-Za-z 0-9,!]+ } |\\textcolor[ \[\]RGBrgb]+{ [A-Za-z 0-9,!]+ } |\\cellcolor[ \[\]RGBrgb]+{ [A-Za-z 0-9,!]+ } "
    old_token = re.findall(pattern, l, re.DOTALL)
    for bef in old_token:
        l = l.replace(bef, "")

    # ** filling the missing brace [] and {} according to token.
    l_split = l.split(" ")
    idx = 0
    while idx < len(l_split):
        token = l_split[idx]
        if token in ONE_Tail_Tokens + ONE_Tail_Invisb_Tokens:
            # ** normalize tokens such as \hat, fill missing the {}, such as \hat \lambda -> \hat {\lambda}
            sub_idx = idx + 1
            while sub_idx < len(l_split) and l_split[sub_idx] in ONE_Tail_Tokens + ONE_Tail_Invisb_Tokens:
                sub_idx += 1
            new_split = l_split[0:idx]
            for ii in range(idx, sub_idx):
                new_split = new_split + [l_split[ii], "{"]
            if l_split[sub_idx] != "{":
                new_split = new_split + [l_split[sub_idx]] + ["}"] * (sub_idx - idx)
                l_split = new_split + l_split[sub_idx + 1 :]
            else:
                end_idx = find_matching_brace(l_split, sub_idx)
                new_split = new_split + l_split[sub_idx + 1 : end_idx] + ["}"] * (sub_idx - idx)
                l_split = new_split + l_split[end_idx + 1 :]
        elif token in AB_Tail_Tokens:
            # ** normalize special tokens such as \sqrt, fill the missing [] {} in \sqrt [] {}, yet the [] is optional, for example: \sqrt A B -> \sqrt {A} B and \sqrt [A] B -> \sqrt [A] {B}
            if l_split[idx + 1] != "[" and l_split[idx + 1] != "{":
                l_split = l_split[0 : idx + 1] + ["{"] + [l_split[idx + 1]] + ["}"] + l_split[idx + 2 :]
            else:
                if l_split[idx + 1] == "[":
                    end1 = find_matching_brace(l_split, idx + 1, brace=["[", "]"])
                else:
                    end1 = idx
                if l_split[end1 + 1] != "{":
                    l_split = l_split[0 : end1 + 1] + ["{"] + [l_split[end1 + 1]] + ["}"] + l_split[end1 + 2 :]
        elif token in TWO_Tail_Tokens + TWO_Tail_Invisb_Tokens:
            # ** normalize special tokens such as \frac, add missing brace in \frac {A} {B} for example: \frac {\lambda} 2 -> \frac {\lambda} {2}
            if l_split[idx + 1] != "{":
                l_split = l_split[0 : idx + 1] + ["{"] + [l_split[idx + 1]] + ["}"] + l_split[idx + 2 :]
            end1 = find_matching_brace(l_split, idx + 1)
            if l_split[end1 + 1] != "{":
                l_split = l_split[0 : end1 + 1] + ["{"] + [l_split[end1 + 1]] + ["}"] + l_split[end1 + 2 :]

        idx += 1
    l = " ".join(l_split)

    return l


def token_add_color(l_split, idx, render_dict):
    token = l_split[idx]
    if token in PHANTOM_Tokens:
        # ** special tokens that do not need render, skip it
        if l_split[idx + 1] == "{":
            brace_end = find_matching_brace(l_split, idx + 1)
        else:
            brace_end = idx + 1
        next_idx = brace_end + 1
    elif token in TWO_Tail_Tokens:
        # ** tokens such as \frac A B, and the token needs render too.
        num_start = idx + 1
        num_end = find_matching_brace(l_split, num_start)
        den_start = num_end + 1
        den_end = find_matching_brace(l_split, den_start)
        l_split_copy = (
            l_split[:idx]
            + [r"\mathcolor{black}{" + token + "{"]
            + [r"\mathcolor{gray}{"]
            + l_split[num_start + 1 : num_end]
            + ["}"]
            + [r"}{"]
            + [r"\mathcolor{gray}{"]
            + l_split[den_start + 1 : den_end]
            + ["}"]
            + ["}"]
            + ["}"]
            + l_split[den_end + 1 :]
        )

        l_new = " ".join(l_split_copy)
        l_new = r"\mathcolor{gray}{ " + l_new + " }"
        render_dict[str(idx)] = l_new, token
        next_idx = idx + 1
    elif token in ONE_Tail_Tokens:
        # ** tokens such as \hat A, and the token needs render too.
        num_start = idx + 1
        num_end = find_matching_brace(l_split, num_start)
        l_split_copy = l_split[:idx] + [r"\mathcolor{black}{"] + l_split[idx : num_start + 1] + [r"\mathcolor{gray}{"] + l_split[num_start + 1 : num_end] + ["}"] + l_split[num_end : num_end + 1] + ["}"] + l_split[num_end + 1 :]
        l_new = " ".join(l_split_copy)
        l_new = r"\mathcolor{gray}{ " + l_new + " }"
        render_dict[str(idx)] = l_new, token
        next_idx = idx + 1
    elif token in ONE_Tail_Invisb_Tokens:
        # ** tokens such as \text A B, and the token does not need render.
        num_start = idx + 1
        num_end = find_matching_brace(l_split, num_start)
        sub_idx = num_start + 1
        if num_end - num_start == 2:
            l_split_copy = l_split.copy()
            l_split_copy[sub_idx] = r"{\mathcolor{black}{" + l_split_copy[sub_idx] + "}}"
            l_new = " ".join(l_split_copy)
            l_new = r"\mathcolor{gray}{ " + l_new + " }"
            render_dict[str(idx)] = l_new, l_split[sub_idx]
            next_idx = num_end
        else:
            while sub_idx < num_end:
                l_split, sub_idx, render_dict = token_add_color(l_split, sub_idx, render_dict)
        next_idx = num_end + 1
    elif token in AB_Tail_Tokens:
        # ** special token \xrightarrow, could be \xrightarrow [] {} or \xrightarrow {}, process method are different.
        if l_split[idx + 1] == "{":
            num_start = idx + 1
            num_end = find_matching_brace(l_split, num_start)
            l_split_copy = l_split[:idx] + [r"\mathcolor{black}{"] + l_split[idx : idx + 2] + [r"\mathcolor{gray}{"] + l_split[num_start + 1 : num_end] + ["}}"] + l_split[num_end:]
            l_new = " ".join(l_split_copy)
            l_new = r"\mathcolor{gray}{ " + l_new + " }"
            render_dict[str(idx)] = l_new, token
            sub_idx = num_start + 1
            while sub_idx < num_end:
                l_split, sub_idx, render_dict = token_add_color(l_split, sub_idx, render_dict)
            next_idx = num_end + 1
        elif l_split[idx + 1] == "[":
            num_start = idx + 1
            num_end = find_matching_brace(l_split, num_start, brace=["[", "]"])
            den_start = num_end + 1
            den_end = find_matching_brace(l_split, den_start)
            l_split_copy = (
                l_split[:idx]
                + [r"{\mathcolor{black}{"]
                + l_split[idx : idx + 2]
                + [r"\mathcolor{gray}{"]
                + l_split[idx + 2 : num_end]
                + ["}"]
                + l_split[num_end : den_start + 1]
                + [r"\mathcolor{gray}{"]
                + l_split[den_start + 1 : den_end]
                + ["}"]
                + l_split[den_end : den_end + 1]
                + ["}}"]
                + l_split[den_end + 1 :]
            )
            l_new = " ".join(l_split_copy)
            l_new = r"\mathcolor{gray}{ " + l_new + " }"
            render_dict[str(idx)] = l_new, token
            sub_idx = num_start + 1
            while sub_idx < num_end:
                l_split, sub_idx, render_dict = token_add_color(l_split, sub_idx, render_dict)
            sub_idx = den_start + 1
            while sub_idx < den_end:
                l_split, sub_idx, render_dict = token_add_color(l_split, sub_idx, render_dict)
            next_idx = den_end + 1
    elif token in ["\\multicolumn", "\\multirow"]:
        # ** tokens with three {}, such as \multicolumn {} {} {}, the text in third {} need be rendered.
        first_start = idx + 1
        first_end = find_matching_brace(l_split, first_start)
        second_start = first_end + 1
        second_end = find_matching_brace(l_split, second_start)
        third_start = second_end + 1
        third_end = find_matching_brace(l_split, third_start)

        sub_idx = third_start + 1
        while sub_idx < third_end:
            l_split, sub_idx, render_dict = token_add_color(l_split, sub_idx, render_dict)
        next_idx = third_end + 1
    elif token in SKIP_Tokens + TWO_Tail_Invisb_Tokens or any(re.match(pattern, token) for pattern in SKIP_PATTERNS):
        # ** tokens no need render, just skip
        # print('skip', idx, token)
        # TODO special case :[], could be single, or in \sqrt[]{}.
        if (token == "[" and l_split[idx - 1] != "\\sqrt") or (token == "]" and idx >= 3 and l_split[idx - 3] != "\\sqrt"):
            l_split_copy = l_split.copy()
            l_split_copy[idx] = r"\mathcolor{black}{ " + l_split_copy[idx] + " }"
            l_new = " ".join(l_split_copy)
            l_new = r"\mathcolor{gray}{ " + l_new + " }"
            render_dict[str(idx)] = l_new, token
            next_idx = idx + 1
        else:
            next_idx = idx + 1
    else:
        # ** nomal token
        l_split_copy = l_split.copy()
        # TODO sometimes there is translation after add color, the exp prove that \mathcolor{black}{ A } is better than \mathcolor{black}{A}
        l_split_copy[idx] = r"\mathcolor{black}{ " + l_split_copy[idx] + " }"

        l_new = " ".join(l_split_copy)
        l_new = r"\mathcolor{gray}{ " + l_new + " }"
        render_dict[str(idx)] = l_new, token
        next_idx = idx + 1

    return l_split, next_idx, render_dict


def token_add_color_RGB(l_split, idx, token_list, brace_color=False):
    """using \mathcolor[RGB]{r,g,b} to render latex."""
    token = l_split[idx]
    if not token:
        next_idx = idx + 1
    elif token in PHANTOM_Tokens:
        # ** special tokens that do not need render, skip it
        if l_split[idx + 1] == "{":
            brace_end = find_matching_brace(l_split, idx + 1)
        else:
            brace_end = idx + 1
        next_idx = brace_end + 1
    elif token in TWO_Tail_Tokens:
        # ** tokens such as \frac A B, and the token needs render too.
        num_start = idx + 1
        num_end = find_matching_brace(l_split, num_start)
        den_start = num_end + 1
        den_end = find_matching_brace(l_split, den_start)
        color_token = "\\color[RGB]{<color_<idx>>}{".replace("<idx>", str(len(token_list)))
        l_split = l_split[:idx] + [color_token + token] + l_split[idx + 1 : den_end + 1] + ["}"] + l_split[den_end + 1 :]
        token_list.append(token)
        next_idx = idx + 1
    elif token in ONE_Tail_Tokens:
        # ** tokens such as \hat A, and the token needs render too.
        num_start = idx + 1
        num_end = find_matching_brace(l_split, num_start)
        color_token = "\\color[RGB]{<color_<idx>>}{".replace("<idx>", str(len(token_list)))
        if token != "\\underbrace" and num_end + 1 < len(l_split) and l_split[num_end + 1] == "_":
            l_split = l_split[:idx] + ["{" + color_token + token] + l_split[idx + 1 : num_end + 1] + ["}}"] + l_split[num_end + 1 :]
        else:
            l_split = l_split[:idx] + [color_token + token] + l_split[idx + 1 : num_end + 1] + ["}"] + l_split[num_end + 1 :]
        token_list.append(token)
        next_idx = idx + 1
    elif token in ONE_Tail_Invisb_Tokens:
        # ** tokens such as \text A B, and the token does not need render.
        num_start = idx + 1
        num_end = find_matching_brace(l_split, num_start)
        sub_idx = num_start + 1
        if num_end - num_start == 2:
            color_token = "\\color[RGB]{<color_<idx>>}{".replace("<idx>", str(len(token_list)))
            token_list.append(l_split[num_start + 1])
            l_split = l_split[: num_start + 1] + [color_token + l_split[num_start + 1] + "}"] + l_split[num_end:]
        else:
            while sub_idx < num_end:
                l_split, sub_idx, token_list = token_add_color_RGB(l_split, sub_idx, token_list)
        next_idx = num_end + 1
    elif token in AB_Tail_Tokens:
        # ** special token \xrightarrow, could be \xrightarrow [] {} or \xrightarrow {}, process method are different.
        if l_split[idx + 1] == "{":
            num_start = idx + 1
            num_end = find_matching_brace(l_split, num_start)
            color_token = "\\color[RGB]{<color_<idx>>}{".replace("<idx>", str(len(token_list)))
            l_split = l_split[:idx] + [color_token + token] + l_split[idx + 1 : num_end + 1] + ["}"] + l_split[num_end + 1 :]
            token_list.append(token)
            sub_idx = num_start + 1
            while sub_idx < num_end:
                l_split, sub_idx, token_list = token_add_color_RGB(l_split, sub_idx, token_list)
            next_idx = num_end + 1
        elif l_split[idx + 1] == "[":
            num_start = idx + 1
            num_end = find_matching_brace(l_split, num_start, brace=["[", "]"])
            den_start = num_end + 1
            den_end = find_matching_brace(l_split, den_start)
            color_token = "\\color[RGB]{<color_<idx>>}{".replace("<idx>", str(len(token_list)))
            l_split = l_split[:idx] + [color_token + token] + l_split[idx + 1 : den_end + 1] + ["}"] + l_split[den_end + 1 :]
            token_list.append(token)
            sub_idx = num_start + 1
            while sub_idx < num_end:
                l_split, sub_idx, token_list = token_add_color_RGB(l_split, sub_idx, token_list, brace_color=True)
            sub_idx = den_start + 1
            while sub_idx < den_end:
                l_split, sub_idx, token_list = token_add_color_RGB(l_split, sub_idx, token_list)
            next_idx = den_end + 1
    elif token in ["\\multicolumn", "\\multirow"]:
        # ** tokens with three {}, such as \multicolumn {} {} {}, the text in third {} need be rendered.
        first_start = idx + 1
        first_end = find_matching_brace(l_split, first_start)
        second_start = first_end + 1
        second_end = find_matching_brace(l_split, second_start)
        third_start = second_end + 1
        third_end = find_matching_brace(l_split, third_start)

        sub_idx = third_start + 1
        while sub_idx < third_end:
            l_split, sub_idx, token_list = token_add_color_RGB(l_split, sub_idx, token_list)
        next_idx = third_end + 1
    elif token in SKIP_Tokens + TWO_Tail_Invisb_Tokens or any(re.match(pattern, token) for pattern in SKIP_PATTERNS):
        # ** tokens no need render, just skip
        # print('skip', idx, token)
        # TODO special case :[], could be single, or in \sqrt[]{}.
        if (token == "[" and l_split[idx - 1] != "\\sqrt") or (token == "]" and idx >= 3 and l_split[idx - 3] != "\\sqrt"):
            color_token = "\\color[RGB]{<color_<idx>>}{".replace("<idx>", str(len(token_list)))
            l_split = l_split[:idx] + [color_token + l_split[idx] + "}"] + l_split[idx + 1 :]
            token_list.append(token)
            next_idx = idx + 1
        else:
            next_idx = idx + 1
    else:
        # ** nomal token
        if brace_color or (idx > 1 and l_split[idx - 1] == "_"):
            color_token = "\\color[RGB]{<color_<idx>>}{".replace("<idx>", str(len(token_list)))
            l_split = l_split[:idx] + ["{" + color_token + l_split[idx] + "}}"] + l_split[idx + 1 :]
            token_list.append(token)
            next_idx = idx + 1
        else:
            color_token = "\\color[RGB]{<color_<idx>>}{".replace("<idx>", str(len(token_list)))
            l_split = l_split[:idx] + [color_token + l_split[idx] + "}"] + l_split[idx + 1 :]
            token_list.append(token)
            next_idx = idx + 1
    return l_split, next_idx, token_list
