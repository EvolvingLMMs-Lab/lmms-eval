import json
import logging
import os
import subprocess
from threading import Timer

import numpy as np
from PIL import Image, ImageDraw

from .latex_processor import normalize_latex, token_add_color_RGB
from .tokenize_latex.tokenize_latex import tokenize_latex

tabular_template = r"""
\documentclass[12pt]{article}
\usepackage[landscape]{geometry}
\usepackage{geometry}
\geometry{a<PaperSize>paper,scale=0.98}
\pagestyle{empty}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{amssymb}
\usepackage{upgreek}
\usepackage{amsmath}
\usepackage{xcolor}
\begin{document}
\makeatletter
\renewcommand*{\@textcolor}[3]{%%
  \protect\leavevmode
  \begingroup
    \color#1{#2}#3%%
  \endgroup
}
\makeatother
\begin{displaymath}
%s
\end{displaymath}
\end{document}
"""

# 需要配置Source Han Sans SC或其他中文字体
formular_template = r"""
\documentclass[12pt]{article}
\usepackage[landscape]{geometry}
\usepackage{geometry}
\geometry{a<PaperSize>paper,scale=0.98}
\pagestyle{empty}
\usepackage{amsmath}
\usepackage{upgreek}
\usepackage{amssymb}
\usepackage{xcolor}
\usepackage{xeCJK}
\setCJKmainfont{Source Han Sans SC}
\setCJKsansfont{Source Han Sans SC}
\setCJKmonofont{Source Han Sans SC}
\xeCJKsetup{CJKmath=true}
\begin{document}
\makeatletter
\renewcommand*{\@textcolor}[3]{%%
  \protect\leavevmode
  \begingroup
    \color#1{#2}#3%%
  \endgroup
}
\makeatother
\begin{displaymath}
%s
\end{displaymath}
\end{document}
"""


def run_cmd(cmd, timeout_sec=30, temp_dir=None):
    # 设置进程独立的环境变量
    env = os.environ.copy()
    if temp_dir:
        env["TMPDIR"] = temp_dir
        env["TMP"] = temp_dir
        env["TEMP"] = temp_dir
        env["MAGICK_TMPDIR"] = temp_dir
        env["TEXMFCACHE"] = temp_dir
        env["TEXMFVAR"] = temp_dir

    proc = subprocess.Popen(cmd, shell=True, env=env)

    def kill_proc(p):
        return p.kill()

    timer = Timer(timeout_sec, kill_proc, [proc])
    try:
        timer.start()
        stdout, stderr = proc.communicate()
    finally:
        timer.cancel()


def convert_pdf2img(pdf_filename, png_filename, temp_dir=None):
    cmd = 'magick -density 200 -quality 100 "%s" "%s"' % (pdf_filename, png_filename)
    run_cmd(cmd, temp_dir=temp_dir)


def crop_image(image_path, pad=8):
    img = Image.open(image_path).convert("L")
    img_data = np.asarray(img, dtype=np.uint8)
    nnz_inds = np.where(img_data != 255)
    if len(nnz_inds[0]) == 0:
        y_min = 0
        y_max = 10
        x_min = 0
        x_max = 10
    else:
        y_min = np.min(nnz_inds[0])
        y_max = np.max(nnz_inds[0])
        x_min = np.min(nnz_inds[1])
        x_max = np.max(nnz_inds[1])

    img = Image.open(image_path).convert("RGB").crop((x_min - pad, y_min - pad, x_max + pad, y_max + pad))
    img.save(image_path)


def extrac_bbox_from_color_image(image_path, color_list):
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    pixels = list(img.getdata())

    bbox_list = []
    for target_color in color_list:
        target_pixels = [i for i, pixel in enumerate(pixels) if pixel == target_color]
        x_list = []
        y_list = []
        for idx in target_pixels:
            x_list.append(idx % W)
            y_list.append(idx // W)
        try:
            y_min, y_max, x_min, x_max = min(y_list), max(y_list), min(x_list), max(x_list)
            bbox_list.append([x_min - 1, y_min - 1, x_max + 1, y_max + 1])

        except Exception:
            bbox_list.append([])
            continue

    img = img.convert("L")
    img_bw = img.point(lambda x: 255 if x == 255 else 0, "1")
    img_bw.convert("RGB").save(image_path)
    return bbox_list


def latex2bbox_color(input_arg):
    latex, basename, output_path, temp_dir, total_color_list = input_arg
    _template = tabular_template if "tabular" in latex else formular_template
    basename = basename.replace(".jpg", "")  # *****
    output_bbox_path = os.path.join(output_path, "bbox", basename + ".jsonl")
    output_vis_path = os.path.join(output_path, "vis", basename + ".png")
    output_base_path = os.path.join(output_path, "vis", basename + "_base.png")

    if os.path.exists(output_bbox_path) and os.path.exists(output_vis_path) and os.path.exists(output_base_path):
        return

    try:
        latex = latex.replace("\n", " ")
        latex = latex.replace("\%", "<PERCENTAGETOKEN>")
        ret, new_latex = tokenize_latex(latex, middle_file=os.path.join(temp_dir, basename + ".txt"))
        if not (ret and new_latex):
            log = f"ERROR, Tokenize latex failed: {basename}."
            logging.info(log)
            new_latex = latex
        new_latex = new_latex.replace("< P E R C E N T A G E T O K E N >", "\%")
        latex = normalize_latex(new_latex)
        token_list = []
        l_split = latex.strip().split(" ")
        color_list = total_color_list[0 : len(l_split)]
        idx = 0
        while idx < len(l_split):
            l_split, idx, token_list = token_add_color_RGB(l_split, idx, token_list)

        rgb_latex = " ".join(l_split)
        for idx, color in enumerate(color_list):
            R, G, B = color
            rgb_latex = rgb_latex.replace(f"<color_{idx}>", f"{R},{G},{B}")

        if len(token_list) > 1300:
            paper_size = 3
        elif len(token_list) > 600:
            paper_size = 4
        else:
            paper_size = 5
        final_latex = formular_template.replace("<PaperSize>", str(paper_size)) % rgb_latex

    except Exception as e:
        log = f"ERROR, Preprocess latex failed: {basename}; {e}."
        logging.info(log)
        return

    pre_name = output_path.replace("/", "_").replace(".", "_") + "_" + basename
    tex_filename = os.path.join(temp_dir, pre_name + ".tex")
    log_filename = os.path.join(temp_dir, pre_name + ".log")
    aux_filename = os.path.join(temp_dir, pre_name + ".aux")

    # print(os.path.exists(tex_filename), tex_filename)
    with open(tex_filename, "w") as w:
        # print(final_latex, file=w)
        w.write(final_latex)
    # print(os.path.exists(tex_filename), tex_filename)
    # run_cmd(f"pdflatex -interaction=nonstopmode -output-directory={temp_dir} {tex_filename} >/dev/null")
    run_cmd(f'xelatex -interaction=nonstopmode -output-directory={temp_dir} "{tex_filename}" >/dev/null', temp_dir=temp_dir)
    try:
        os.remove(tex_filename)
        os.remove(log_filename)
        os.remove(aux_filename)
    except Exception:
        pass
    pdf_filename = tex_filename[:-4] + ".pdf"
    if not os.path.exists(pdf_filename):
        log = f"ERROR, Compile pdf failed: {pdf_filename}"
        logging.info(log)
    else:
        convert_pdf2img(pdf_filename, output_base_path)
        os.remove(pdf_filename)

        crop_image(output_base_path)
        bbox_list = extrac_bbox_from_color_image(output_base_path, color_list)
        vis = Image.open(output_base_path)
        draw = ImageDraw.Draw(vis)

        with open(output_bbox_path, "w", encoding="utf-8") as f:
            for token, box in zip(token_list, bbox_list):
                item = {"bbox": box, "token": token}
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

                if not box:
                    continue
                x_min, y_min, x_max, y_max = box
                draw.rectangle([x_min, y_min, x_max, y_max], fill=None, outline=(0, 250, 0), width=1)
                try:
                    draw.text((x_min, y_min), token, (250, 0, 0))
                except Exception:
                    pass

        vis.save(output_vis_path)
