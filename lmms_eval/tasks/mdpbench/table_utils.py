"""Vendored table helpers from the reference MDPBench repository.

The parser only uses ``convert_markdown_to_html``. Plotting imports are omitted
so prediction parsing does not require matplotlib at runtime.
"""

import os
import re


def print_aligned_dict(data):
    # Find the maximum length of all keys
    max_key_length = max(len(key) for key in data["testcase1"])

    # Print header
    print(f"{' ' * (max_key_length + 4)}", end="")
    for key in data:
        print(f"{key:>{max_key_length}}", end="")
    print()

    # Print dictionary content
    for subkey in data["testcase1"]:
        print(f"{subkey:<{max_key_length + 4}}", end="")
        for key in data:
            print(f"{data[key][subkey]:>{max_key_length}}", end="")
        print()


def create_dict_from_folders(directory):
    body = {}
    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)
        if os.path.isdir(folder_path):
            body[folder_name] = {}
    return body


# The function is from https://github.com/intsig-textin/markdown_tester
def markdown_to_html(markdown_table):
    rows = [row.strip() for row in markdown_table.strip().split("\n")]
    _num_columns = len(rows[0].split("|")) - 2

    html_table = "<table>\n  <thead>\n    <tr>\n"

    header_cells = [cell.strip() for cell in rows[0].split("|")[1:-1]]
    for cell in header_cells:
        html_table += f"      <th>{cell}</th>\n"
    html_table += "    </tr>\n  </thead>\n  <tbody>\n"

    for row in rows[2:]:
        cells = [cell.strip() for cell in row.split("|")[1:-1]]
        html_table += "    <tr>\n"
        for cell in cells:
            html_table += f"      <td>{cell}</td>\n"
        html_table += "    </tr>\n"

    html_table += "  </tbody>\n</table>\n"
    return html_table


def convert_table_str(s):
    s = re.sub(r"<table.*?>", "<table>", s)
    s = re.sub(r"<th", "<td", s)
    s = re.sub(r"</th>", "</td>", s)
    # s = re.sub(r'<td rowspan="(.)">',lambda x:f'<td colspan="1" rowspan="{x.group(1)}">',s)
    # s = re.sub(r'<td colspan="(.)">',lambda x:f'<td colspan="{x.group(1)}" rowspan="1">',s)
    res = ""
    res += "\n\n"
    temp_item = ""
    for c in s:
        temp_item += c
        if c == ">" and not re.search(r"<td.*?>\$", temp_item):
            res += temp_item + "\n"
            temp_item = ""
    return res + "\n"


def merge_table(md):
    table_temp = ""
    for line in md:
        table_temp += line
    return convert_table_str(table_temp)


def find_md_table_mode(line):
    if re.search(r"-*?:", line) or re.search(r"---", line) or re.search(r":-*?", line):
        return True
    return False


def delete_table_and_body(input_list):
    res = []
    for line in input_list:
        if not re.search(r"</?t(able|head|body)>", line):
            res.append(line)
    return res


def merge_tables(input_str):
    # Delete HTML comments
    input_str = re.sub(r"<!--[\s\S]*?-->", "", input_str)

    # Use regex to find each <table> block
    table_blocks = re.findall(r"<table>[\s\S]*?</table>", input_str)

    # Process each <table> block, replace <th> with <td>
    output_lines = []
    for block in table_blocks:
        block_lines = block.split("\n")
        for i, line in enumerate(block_lines):
            if "<th>" in line:
                block_lines[i] = line.replace("<th>", "<td>").replace("</th>", "</td>")
        final_tr = delete_table_and_body(block_lines)
        if len(final_tr) > 2:
            output_lines.extend(final_tr)  # Ignore <table> and </table> tags, keep only table content

    # Rejoin the processed strings
    merged_output = "<table>\n{}\n</table>".format("\n".join(output_lines))

    return "\n\n" + merged_output + "\n\n"


def replace_table_with_placeholder(input_string):
    lines = input_string.split("\n")
    output_lines = []

    in_table_block = False
    temp_block = ""
    last_line = ""

    _org_table_list = []
    _in_org_table = False

    for idx, line in enumerate(lines):
        # if not in_org_table:
        # if "<table>" not in last_line and in_table_block == False and temp_block != "":
        #     output_lines.append(merge_tables(temp_block))
        #     temp_block = ""
        if "<table>" in line:
            # if "<table><tr" in line:
            #     org_table_list.append(line)
            #     in_org_table = True
            #     output_lines.append(last_line)
            #     continue
            # else:
            in_table_block = True
            temp_block += last_line
        elif in_table_block:
            if not find_md_table_mode(last_line) and "</thead>" not in last_line:
                temp_block += "\n" + last_line
            if "</table>" in last_line:
                if "<table>" not in line:
                    in_table_block = False
                    output_lines.append(merge_tables(temp_block))
                    temp_block = ""
        else:
            output_lines.append(last_line)

        last_line = line
        # else:
        #     org_table_list.append(line)
        #     if "</table" in line:
        #         in_org_table = False
        #         last_line = merge_table(org_table_list)
        #         org_table_list = []

    if last_line:
        if in_table_block or "</table>" in last_line:
            temp_block += "\n" + last_line
            output_lines.append(merge_tables(temp_block))
        else:
            output_lines.append(last_line)
    # if "</table>" in last_line:
    #     output_lines.append(merge_tables(temp_block))

    return "\n".join(output_lines)


def convert_table(input_str):
    # Replace <table>
    output_str = input_str.replace("<table>", '<table border="1" >')

    # Replace <td>
    output_str = output_str.replace("<td>", '<td colspan="1" rowspan="1">')

    return output_str


def convert_markdown_to_html(markdown_content):
    # Define a regex pattern to find Markdown tables with newlines
    markdown_content = markdown_content.replace("\r", "") + "\n"
    pattern = re.compile(r"\|\s*.*?\s*\|\n", re.DOTALL)

    # Find all matches in the Markdown content
    matches = pattern.findall(markdown_content)

    for match in matches:
        html_table = markdown_to_html(match)
        markdown_content = markdown_content.replace(match, html_table, 1)  # Only replace the first occurrence

    res_html = convert_table(replace_table_with_placeholder(markdown_content))

    return res_html
