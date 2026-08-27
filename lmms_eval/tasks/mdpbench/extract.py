"""Vendored prediction parser from the reference MDPBench repository.

Only import paths are adapted for lmms-eval; parsing behavior is unchanged.
"""

import re
from collections import defaultdict

from pylatexenc.latexwalker import (
    LatexCharsNode,
    LatexEnvironmentNode,
    LatexGroupNode,
    LatexMacroNode,
    LatexSpecialsNode,
)

from lmms_eval.tasks.mdpbench.data_preprocess import (
    remove_markdown_fences,
    replace_repeated_chars,
)

# from  modules.table_utils import convert_markdown_to_html #end
from lmms_eval.tasks.mdpbench.table_utils import convert_markdown_to_html


def extract_tabular(text):
    begin_pattern = r"\\begin{tabular}"
    end_pattern = r"\\end{tabular}"

    tabulars = []
    positions = []
    current_pos = 0
    stack = []

    while current_pos < len(text):
        begin_match = re.search(begin_pattern, text[current_pos:])
        end_match = re.search(end_pattern, text[current_pos:])

        if not begin_match and not end_match:
            break

        if begin_match and (not end_match or begin_match.start() < end_match.start()):
            stack.append(current_pos + begin_match.start())
            current_pos += begin_match.start() + len(end_pattern)
        elif end_match:
            if stack:
                start_pos = stack.pop()
                if not stack:
                    end_pos = current_pos + end_match.start() + len(end_pattern)
                    tabular_code = text[start_pos:end_pos]
                    tabulars.append(tabular_code)
                    positions.append((start_pos, end_pos))
            current_pos += end_match.start() + len(end_pattern)
        else:
            current_pos += 1

    if stack:
        new_start = stack[0] + len(begin_pattern)
        new_tabulars, new_positions = extract_tabular(text[new_start:])
        new_positions = [(start + new_start, end + new_start) for start, end in new_positions]
        tabulars.extend(new_tabulars)
        positions.extend(new_positions)

    return tabulars, positions


# math reg
# r'\\begin{equation\*?}(.*?)\\end{equation\*?}|'
# r'\\begin{align\*?}(.*?)\\end{align\*?}|'
# r'\\begin{gather\*?}(.*?)\\end{gather\*?}|'
display_reg = re.compile(
    # r'\\begin{equation\*?}(.*?)\\end{equation\*?}|'
    # r'\\begin{align\*?}(.*?)\\end{align\*?}|'
    # r'\\begin{gather\*?}(.*?)\\end{gather\*?}|'
    # r'\\begin{array\*?}(.*?)\\end{array\*?}|'
    r"\$\$(.*?)\$\$|" r"\\\[(.*?)\\\]|" r"\$(.*?)\$|" r"\\\((.*?)\\\)",
    re.DOTALL,
)

# inline_reg = re.compile(
#     r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)|'
#     r'\\\((.*?)\\\)',
# )
inline_reg = re.compile(
    r"\$(.*?)\$|" r"\\\((.*?)\\\)",
)

# table
table_reg = re.compile(r"\\begin{table\*?}(.*?)\\end{table\*?}|" r"\\begin{tabular\*?}(.*?)\\end{tabular\*?}", re.DOTALL)
md_table_reg = re.compile(r"\|\s*.*?\s*\|\n", re.DOTALL)
html_table_reg = re.compile(r"(<table.*?</table>)", re.DOTALL)

# title
title_reg = re.compile(r"^\s*#.*$", re.MULTILINE)

# img
img_pattern = r"!\[.*?\]\(.*?\)"

# code block
code_block_reg = re.compile(r"```(\w+)\n(.*?)```", re.DOTALL)


def md_tex_filter(content):
    """
    Input: 1 page md or tex content - String
    Output: text, display, inline, table, title, code - list
    """
    content = re.sub(img_pattern, "", content)  # remove image
    content = remove_markdown_fences(content)  # remove markdown fences
    content = replace_repeated_chars(content)  # replace all consecutive characters
    content = content.replace("<html>", "").replace("</html>", "").replace("<body>", "").replace("</body>", "")

    # # 使用正则表达式对unicode进行替换
    # special_unicode = ''.join(unicode_replacements.keys())
    # content = re.sub(f'[{special_unicode}]', replace_unicode, content)

    # content = fullwidth_to_halfwidth(content)  # fullwidth to halfwidth, TODO: GT also needs this operation

    # # pylatexenc's unicode to latex
    # content = unicode_to_latex(content, unknown_char_warning=False)
    # markdown_table_content[i, j] = LatexNodes2Text().latex_to_text(content_str)
    # content_ori = copy.deepcopy(content)

    # print('--------------After pre_process: \n', content)

    pred_all = []
    # deal with inline formula
    # content_new, inline_array = inline_filter_unicode(content)
    # #print('------------inline_array----------------',inline_array)
    # for inline_item in inline_array:
    #     inline_item['content'] = inline_to_unicode(inline_item['content'])
    #     #print('------------inline_array_unicode----------------',inline_item['content'])
    #     pred_all.append({
    #         'category_type': 'text_all',
    #         'position': inline_item['position'],
    #         'content': inline_item['content'],
    #         'fine_category_type': 'equation_inline'
    #     })

    # extract latex table
    latex_table_array, table_positions = extract_tex_table(content)
    for latex_table, position in zip(latex_table_array, table_positions):
        position = [position[0], position[0] + len(latex_table)]  # !!!
        pred_all.append({"category_type": "latex_table", "position": position, "content": latex_table})
        content = content[: position[0]] + " " * (position[1] - position[0]) + content[position[1] :]  # replace latex table with space

    # print('--------After latex table: \n', content)
    # print('-------latex_table_array: \n', latex_table_array)

    # extract html table
    html_table_array, table_positions = extract_html_table(content)
    for html_table, position in zip(html_table_array, table_positions):
        position = [position[0], position[0] + len(html_table)]
        pred_all.append({"category_type": "html_table", "position": position, "content": html_table})
        content = content[: position[0]] + " " * (position[1] - position[0]) + content[position[1] :]  # replace html table with space
    # html_table_array = []
    # html_table_matches = html_table_reg.finditer(content)
    # if html_table_matches:
    #     for match in html_table_matches:
    #         matched = match.group(0)
    #         position = [match.start(), match.end()]
    #         html_table_array.append(matched.strip())
    #         # content = content.replace(matched, ' '*len(matched)) # replace html table with space
    #         content = content[:position[0]] + ' '*(position[1]-position[0]) + content[position[1]:]  # replace html table with space
    #         pred_all.append({
    #             'category_type': 'html_table',
    #             'position': position,
    #             'content': matched.strip()
    #         })

    # print('--------------After html table: \n', content)
    # # extract tables in latex and html
    # table_array = []
    # table_matches = table_reg.finditer(content)
    # tables = ""
    # for match in table_matches:
    #     matched = match.group(0)
    #     if matched:
    #         tables += matched
    #         tables += "\n\n"
    #         table_array.append(matched)
    #         content = content.replace(matched, '')

    # extract interline formula
    display_matches = display_reg.finditer(content)
    _content_copy = content
    for match in display_matches:
        matched = match.group(0)
        if matched:
            # single_line = ''.join(matched.split())
            single_line = " ".join(matched.strip().split("\n"))
            position = [match.start(), match.end()]
            # replace $$ with \[\]
            dollar_pattern = re.compile(r"\$\$(.*?)\$\$|\$(.*?)\$|\\\((.*?)\\\)", re.DOTALL)
            sub_match = dollar_pattern.search(single_line)
            if sub_match is None:
                # pass
                content = content[: position[0]] + " " * (position[1] - position[0]) + content[position[1] :]
                pred_all.append({"category_type": "equation_isolated", "position": position, "content": single_line})
            elif sub_match.group(1):
                single_line = re.sub(dollar_pattern, r"\\[\1\\]", single_line)
                content = content[: position[0]] + " " * (position[1] - position[0]) + content[position[1] :]  # replace equation with space
                pred_all.append({"category_type": "equation_isolated", "position": position, "content": single_line})
            else:
                # start, end = match.span()
                # char_before = content_copy[start-1] if start > 0           else '\n'
                # char_after  = content_copy[end]   if end   < len(content_copy) else '\n'
                # if char_before == '\n' or char_after == '\n':
                #     single_line = re.sub(dollar_pattern, r'\\[\2\3\\]', single_line)
                #     pred_all.append({
                #         'category_type': 'equation_isolated',
                #         'position': position,
                #         'content': single_line,
                #         'fine_category_type': 'equation_inline'
                #     })
                single_line = re.sub(dollar_pattern, r"\\[\2\3\\]", single_line)
                pred_all.append({"category_type": "equation_isolated", "position": position, "content": single_line, "fine_category_type": "equation_inline"})
            # single_line = re.sub(dollar_pattern, r'\\[\1\2\3\\]', single_line)
            # print('single_line: ', single_line)
            # content = content.replace(matched, ' '*len(matched))
            # pred_all.append({
            #     'category_type': 'equation_isolated',
            #     'position': position,
            #     'content': single_line
            # })
            # print('-----Found display formula: ', matched)

    # print('-------------After display: \n', content)
    # extract md table with ||
    md_table_mathces = md_table_reg.findall(content + "\n")
    if len(md_table_mathces) >= 2:
        # print("md table found!")
        # print("content:", content)
        content = convert_markdown_to_html(content)
        # print('----------content after converting md table to html:', content)
        html_table_matches = html_table_reg.finditer(content)
        if html_table_matches:
            for match in html_table_matches:
                matched = match.group(0)
                position = [match.start(), match.end()]
                # content = content.replace(match, '')
                # print('content after removing the md table:', content)
                content = content[: position[0]] + " " * (position[1] - position[0]) + content[position[1] :]  # replace md table with space
                pred_all.append({"category_type": "html_table", "position": position, "content": matched.strip(), "fine_category_type": "md2html_table"})
    # print('---------After md table: \n', content)

    # extract code blocks
    code_matches = code_block_reg.finditer(content)
    if code_matches:
        for match in code_matches:
            position = [match.start(), match.end()]
            language = match.group(1)
            code = match.group(2).strip()
            # content = content.replace(match.group(0), '')
            content = content[: position[0]] + " " * (position[1] - position[0]) + content[position[1] :]  # replace code block with space
            pred_all.append({"category_type": "text_all", "position": position, "content": code, "language": language, "fine_category_type": "code"})

    # print('-------After code block: \n', content)

    # # Extract titles: Do not extract titles, as some models do not wrap code blocks, causing all comments to be treated as titles
    # title_matches = title_reg.finditer(content)
    # if title_matches:
    #     for match in title_matches:
    #         position = [match.start(), match.end()]
    #         matched = match.group(0)
    #         matched = matched.replace("#", "").strip()
    #         # content = content.replace(match, '')
    #         # print('content after removing the titles:', content)
    #         if matched:
    #             # print('Add title: ', matched)
    #             content = content[:position[0]] + ' '*(position[1]-position[0]) + content[position[1]:]
    #             pred_all.append({
    #                 'category_type': 'text_all',
    #                 'position': position,
    #                 'content': matched,
    #                 'fine_category_type': 'title'
    #             })

    # print('----------After title: \n', content)

    # # Delete extracted content
    # extracted_position = [_['position'] for _ in pred_all]
    # for start, end in sorted(extracted_position, reverse=True):
    #     content = content[:start] + content[end:]

    # print('----------After delete extracted: \n', content)

    # Remove latex style
    content = re.sub(r"\\title\{(.*?)\}", r"\1", content)
    content = re.sub(r"\\title\s*\{\s*(.*?)\s*\}", r"\1", content, flags=re.DOTALL)
    content = re.sub(r"\\text\s*\{\s*(.*?)\s*\}", r"\1", content, flags=re.DOTALL)
    content = re.sub(r"\\section\*?\{(.*?)\}", r"\1", content)
    content = re.sub(r"\\section\*?\{\s*(.*?)\s*\}", r"\1", content, flags=re.DOTALL)

    # extract texts
    res = content.split("\n\n")
    if len(res) == 1:
        res = content.split("\n")  # some models do not use double newlines, so use single newlines to split

    content_position = 0
    for text in res:
        position = [content_position, content_position + len(text)]
        content_position += len(text)
        text = text.strip()
        text = text.strip("\n")
        # print('ori_text: ', text)
        text = "\n".join([_.strip() for _ in text.split("\n") if _.strip()])  # avoid some single newline content with many spaces
        # print('after strip text: ', text)

        if text:  # Check if the stripped text is not empty
            if text.startswith("<table") and text.endswith("</table>"):
                pred_all.append(
                    {
                        "category_type": "html_table",
                        "position": position,
                        "content": text,
                    }
                )
            # elif text.startswith('#') and '\n' not in text:
            #     text = text.replace('#', '').strip()
            #     if text:
            #         # print('Add title: ', matched)
            #         pred_all.append({
            #             'category_type': 'text_all',
            #             'position': position,
            #             'content': text,
            #             'fine_category_type': 'title'
            #         })
            elif text.startswith("$") and text.endswith("$"):
                if text.replace("$", "").strip():
                    pred_all.append(
                        {
                            "category_type": "equation_isolated",
                            "position": position,
                            "content": text.strip(),
                        }
                    )
            else:
                text = text.strip()
                if text:
                    pred_all.append({"category_type": "text_all", "position": position, "content": text, "fine_category_type": "text_block"})
                # if '$' in text:
                #     for formula in re.findall(r'\$(.*?)\$', text):
                #         formula_array.append(formula)

    pred_dataset = defaultdict(list)
    pred_all = sorted(pred_all, key=lambda x: x["position"][0])
    for item in pred_all:
        pred_dataset[item["category_type"]].append(item)
    # pdb.set_trace()
    return pred_dataset


# def replace_or_extract(match):
#     content = match.group(1) if match.group(1) is not None else match.group(2)

#     if any(char in content for char in r'\^_'):
#         inline_array.append(match.group(0))
#         return ''
#     else:
#         return content

# extract inline math equations in text
# def inline_filter(text):

#     inline_array = []
#     inline_matches = inline_reg.finditer(text)
#     for match in inline_matches:
#         content = match.group(1) if match.group(1) is not None else match.group(2)

#         # remove \\, \_, \&, \%, \^
#         clean_content = re.sub(r'\\([\\_&%^])', '', content)

#         if any(char in clean_content for char in r'\^_'):
#             inline_array.append(match.group(0))
#             text = text.replace(match.group(0), '')
#         else:
#             text = text.replace(match.group(0), content)

#     return text, inline_array

# def extract_tex_table(content):
#     tables = []
#     positions = []

#     walker = LatexWalker(content)
#     nodes, _, _ = walker.get_latex_nodes()
#     if nodes is None:
#         return tables, positions

#     for node in nodes:
#         if isinstance(node, LatexEnvironmentNode) and (
#             node.environmentname == 'tabular' or node.environmentname == 'table'):
#             # table_latex = extract_node_content(node)
#             table_latex = content[node.pos:node.pos_end]
#             tables.append(table_latex)
#             start_pos = node.pos
#             end_pos = get_node_end_pos(node)
#             positions.append((start_pos, end_pos))

#     return tables, positions


def extract_tex_table(content):
    tables = []
    tables_positions = []

    pattern = r"\\begin{table}(.*?)\\end{table}"
    for match in re.finditer(pattern, content, re.DOTALL):
        start_pos = match.start()
        end_pos = match.end()
        table_content = match.group(0)
        tables.append(table_content)
        tables_positions.append((start_pos, end_pos))
        content = content[:start_pos] + " " * (end_pos - start_pos) + content[end_pos:]

    tabulars, tabular_positions = extract_tabular(content)
    all_tables = tables + tabulars
    all_positions = tables_positions + tabular_positions

    all_result = sorted([[pos, table] for pos, table in zip(all_positions, all_tables)], key=lambda x: x[0][0])
    all_tables = [x[1] for x in all_result]
    all_positions = [x[0] for x in all_result]

    return all_tables, all_positions


# def extract_html_table(content):
#     soup = BeautifulSoup(content, 'html.parser')
#     all_tables = soup.find_all('table')
#     tables = []
#     positions = []

#     for table in all_tables:
#         if table.find_parent('table') is None:
#             table_str = str(table)
#             start_pos = content.find(table_str)
#             end_pos = start_pos + len(table_str)

#             tables.append(table_str)
#             positions.append((start_pos, end_pos))
#     return tables, positions


def extract_html_table(text):
    begin_pattern = r"<table(?:[^>]*)>"
    end_pattern = r"</table>"

    tabulars = []
    positions = []
    current_pos = 0
    stack = []

    while current_pos < len(text):
        begin_match = re.search(begin_pattern, text[current_pos:])
        end_match = re.search(end_pattern, text[current_pos:])

        if not begin_match and not end_match:
            break

        if begin_match and (not end_match or begin_match.start() < end_match.start()):
            stack.append(current_pos + begin_match.start())
            current_pos += begin_match.start() + len(end_pattern)
        elif end_match:
            if stack:
                start_pos = stack.pop()
                if not stack:
                    end_pos = current_pos + end_match.start() + len(end_pattern)
                    tabular_code = text[start_pos:end_pos]
                    tabulars.append(tabular_code)
                    positions.append((start_pos, end_pos))
            current_pos += end_match.start() + len(end_pattern)
        else:
            current_pos += 1

    if stack:
        new_start = stack[0] + len(begin_pattern)
        new_tabulars, new_positions = extract_html_table(text[new_start:])
        new_positions = [(start + new_start, end + new_start) for start, end in new_positions]
        tabulars.extend(new_tabulars)
        positions.extend(new_positions)

    return tabulars, positions


def extract_node_content(node):
    """Recursively extract content from LatexEnvironmentNode and rebuild LaTeX table representation"""
    if isinstance(node, LatexCharsNode):
        return node.chars  # Use chars attribute
    elif isinstance(node, LatexGroupNode):
        return "{" + "".join(extract_node_content(n) for n in node.nodelist) + "}"
    elif isinstance(node, LatexMacroNode):
        # Extract macro command and its arguments
        macro_content = "\\" + node.macroname
        if node.nodeargs:
            macro_content += "".join([extract_node_content(arg) for arg in node.nodeargs])
        return macro_content
    elif isinstance(node, LatexEnvironmentNode):
        # Extract environment, preserve environment name and arguments
        content = "\\begin{" + node.environmentname + "}"
        if node.nodeargd and node.nodeargd.argnlist:
            # content += "".join("{" + extract_node_content(arg) + "}" for arg in node.nodeargd)
            # content += "".join("{" + extract_node_content(node.nodeargd) + "}")
            content += "{" + extract_node_content(node.nodeargd.argnlist[0]) + "}"
        if node.nodelist:
            content += "".join(extract_node_content(n) for n in node.nodelist)
        content += "\\end{" + node.environmentname + "}"
        return content
    elif isinstance(node, LatexSpecialsNode):  # Changed to LatexSpecialsNode
        return node.specials_chars
    else:
        return ""


def get_node_end_pos(node):
    """Recursively determine the end position of a node"""
    if hasattr(node, "nodelist") and node.nodelist:
        # If the node has child nodes, recursively find the end position of the last child node
        return get_node_end_pos(node.nodelist[-1])
    elif hasattr(node, "pos_end"):
        # If the node has pos_end attribute, return it directly
        return node.pos_end
    else:
        # If there are no child nodes, assume the node ends at the last character of its content
        return node.pos + len(str(node))


def remove_tex_table(content):
    tables, positions = extract_tex_table(content)

    # Delete in reverse order by position to avoid affecting unprocessed start positions
    for start, end in sorted(positions, reverse=True):
        content = content[:start] + content[end:]  # Remove table content

    return content
