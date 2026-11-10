# Adapted from allenai/olmocr at f7cfe4c22098b154c76b6ec950d1c0a464eecf8d.
# Copyright Allen Institute for Artificial Intelligence.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Official table cell and relation parser."""

from collections import defaultdict
from dataclasses import dataclass

from lmms_eval.imports import require_package

require_package("bs4", extras="olmocr_bench", feature="olmOCR-Bench table scoring")
from bs4 import BeautifulSoup  # noqa: E402 - validate the optional dependency first


@dataclass(frozen=True)
class TableData:
    """Table cells and their directional and heading relationships.

    Spanning cells occur once in cell_text, at their top-left coordinate. A cell
    can have multiple neighbors in each direction because of row/column spans.
    Heading lookup walks the connection graph to cells marked as headings.
    """

    cell_text: dict[tuple[int, int], str]  # Stores map from row, col to cell text
    heading_cells: set[tuple[int, int]]  # Contains the row, col pairs which are headings

    is_rectangular: bool  # False if any cells are missing after accounting for spans.

    up_relations: dict[tuple[int, int], set[tuple[int, int]]]
    down_relations: dict[tuple[int, int], set[tuple[int, int]]]
    left_relations: dict[tuple[int, int], set[tuple[int, int]]]
    right_relations: dict[tuple[int, int], set[tuple[int, int]]]

    def _walk_heading_relations(
        self,
        start: tuple[int, int],
        relation: dict[tuple[int, int], set[tuple[int, int]]],
    ) -> set[tuple[int, int]]:
        resulting_heading_cells = set()
        resulting_end_cells = set()

        visited = set()
        to_visit = {start}

        while len(to_visit) > 0:
            cur = to_visit.pop()
            visited.add(cur)

            if cur in self.heading_cells:
                resulting_heading_cells.add(cur)

            if cur not in relation or len(relation[cur]) == 0:
                resulting_end_cells.add(cur)
            else:
                to_visit |= relation[cur]

        resulting_heading_cells -= {start}
        resulting_end_cells -= {start}

        if resulting_heading_cells:
            return resulting_heading_cells
        else:
            return resulting_end_cells

    def top_heading_relations(self, start_row: int, start_col: int) -> set[tuple[int, int]]:
        return self._walk_heading_relations((start_row, start_col), self.up_relations)

    def left_heading_relations(self, start_row: int, start_col: int) -> set[tuple[int, int]]:
        return self._walk_heading_relations((start_row, start_col), self.left_relations)


def _safe_span_int(value: str | int | None, default: int = 1) -> int:
    """Convert rowspan/colspan attributes to positive integers."""
    if value in (None, "", 0):
        return default
    try:
        span = int(value)
    except (TypeError, ValueError):
        return default
    if span <= 0:
        return default
    return span


def _build_table_data_from_specs(row_specs: list[list[dict[str, str | int | bool]]]) -> TableData | None:
    """
    Build a TableData object from a list of row specifications.

    Each row specification is a list of dictionaries with keys:
        - text: cell text content
        - rowspan: integer rowspan (>= 1)
        - colspan: integer colspan (>= 1)
        - is_heading: bool indicating if the cell should be treated as a heading
    """
    if not row_specs:
        return None

    cell_text: dict[tuple[int, int], str] = {}
    heading_cells: set[tuple[int, int]] = set()
    cell_meta: dict[tuple[int, int], dict[str, int | bool]] = {}
    occupancy: list[list[tuple[int, int] | None]] = []
    active_rowspans: list[tuple[tuple[int, int], int] | None] = []

    for row_idx, cells in enumerate(row_specs):
        row_entries: list[tuple[int, int] | None] = []
        col_index = 0
        spec_idx = 0
        total_specs = len(cells)

        while spec_idx < total_specs or col_index < len(active_rowspans):
            if col_index < len(active_rowspans) and active_rowspans[col_index] is not None:
                cell_id, remaining = active_rowspans[col_index]
                row_entries.append(cell_id)
                remaining -= 1
                active_rowspans[col_index] = (cell_id, remaining) if remaining > 0 else None
                col_index += 1
                continue

            if spec_idx >= total_specs:
                if col_index < len(active_rowspans):
                    row_entries.append(None)
                    col_index += 1
                    continue
                break

            spec = cells[spec_idx]
            spec_idx += 1

            text = spec.get("text", "") or ""
            rowspan = spec.get("rowspan", 1)
            colspan = spec.get("colspan", 1)
            is_heading = bool(spec.get("is_heading", False))

            rowspan = rowspan if isinstance(rowspan, int) else _safe_span_int(rowspan)
            colspan = colspan if isinstance(colspan, int) else _safe_span_int(colspan)
            rowspan = max(1, rowspan)
            colspan = max(1, colspan)

            cell_id = (row_idx, col_index)
            cell_text[cell_id] = text
            if is_heading:
                heading_cells.add(cell_id)

            cell_meta[cell_id] = {
                "row": row_idx,
                "col": col_index,
                "rowspan": rowspan,
                "colspan": colspan,
            }

            required_len = col_index + colspan
            if len(active_rowspans) < required_len:
                active_rowspans.extend([None] * (required_len - len(active_rowspans)))

            for offset in range(colspan):
                current_col = col_index + offset
                row_entries.append(cell_id)
                if rowspan > 1:
                    active_rowspans[current_col] = (cell_id, rowspan - 1)
                else:
                    active_rowspans[current_col] = None

            col_index += colspan

        occupancy.append(row_entries)

    # Flush any remaining active rowspans into additional rows
    while any(entry is not None for entry in active_rowspans):
        row_entries: list[tuple[int, int] | None] = []
        for col_index, span_entry in enumerate(active_rowspans):
            if span_entry is None:
                row_entries.append(None)
                continue
            cell_id, remaining = span_entry
            row_entries.append(cell_id)
            remaining -= 1
            active_rowspans[col_index] = (cell_id, remaining) if remaining > 0 else None
        occupancy.append(row_entries)

    if not cell_text:
        return None

    # Normalize occupancy to a consistent width based on populated columns
    valid_columns = {idx for row in occupancy for idx, value in enumerate(row) if value is not None}
    if valid_columns:
        table_width = max(valid_columns) + 1
        for row in occupancy:
            if len(row) < table_width:
                row.extend([None] * (table_width - len(row)))
            elif len(row) > table_width:
                del row[table_width:]
    else:
        return None

    table_height = len(occupancy)

    up_rel = defaultdict(set)
    down_rel = defaultdict(set)
    left_rel = defaultdict(set)
    right_rel = defaultdict(set)

    for cell_id, meta in cell_meta.items():
        row_start = meta["row"]
        col_start = meta["col"]
        rowspan = meta["rowspan"]
        colspan = meta["colspan"]
        row_end = row_start + rowspan - 1
        col_end = col_start + colspan - 1

        # Right relations
        for row in range(row_start, row_end + 1):
            for col in range(col_end + 1, table_width):
                neighbor = occupancy[row][col]
                if neighbor is None or neighbor == cell_id:
                    continue
                right_rel[cell_id].add(neighbor)
                break

        # Left relations
        for row in range(row_start, row_end + 1):
            for col in range(col_start - 1, -1, -1):
                neighbor = occupancy[row][col]
                if neighbor is None or neighbor == cell_id:
                    continue
                left_rel[cell_id].add(neighbor)
                break

        # Down relations
        for col in range(col_start, col_end + 1):
            for row in range(row_end + 1, table_height):
                if col >= len(occupancy[row]):
                    continue
                neighbor = occupancy[row][col]
                if neighbor is None or neighbor == cell_id:
                    continue
                down_rel[cell_id].add(neighbor)
                break

        # Up relations
        for col in range(col_start, col_end + 1):
            for row in range(row_start - 1, -1, -1):
                neighbor = occupancy[row][col]
                if neighbor is None or neighbor == cell_id:
                    continue
                up_rel[cell_id].add(neighbor)
                break

    # Ensure every cell has an entry in relations dictionaries
    up_relations = {cell_id: set(up_rel[cell_id]) for cell_id in cell_text}
    down_relations = {cell_id: set(down_rel[cell_id]) for cell_id in cell_text}
    left_relations = {cell_id: set(left_rel[cell_id]) for cell_id in cell_text}
    right_relations = {cell_id: set(right_rel[cell_id]) for cell_id in cell_text}

    return TableData(
        cell_text=cell_text,
        heading_cells=heading_cells,
        up_relations=up_relations,
        down_relations=down_relations,
        left_relations=left_relations,
        right_relations=right_relations,
        is_rectangular=not any(any(x is None for x in row) for row in occupancy),
    )


def parse_markdown_tables(md_content: str) -> list[TableData]:
    """
    Extract and parse all markdown tables from the provided content.
    Uses a direct approach to find and parse tables, which is more robust for tables
    at the end of files or with irregular formatting.

    Args:
        md_content: The markdown content containing tables

    Returns:
        A list of TableData objects, each containing the table data and header information
    """
    # Split the content into lines and process line by line
    lines = md_content.strip().split("\n")

    parsed_tables = []
    current_table_lines = []
    in_table = False

    # Identify potential tables by looking for lines with pipe characters
    for line in lines:
        # Check if this line has pipe characters (a table row indicator)
        if "|" in line:
            # If we weren't in a table before, start a new one
            if not in_table:
                in_table = True
                current_table_lines = [line]
            else:
                # Continue adding to the current table
                current_table_lines.append(line)
        else:
            # No pipes in this line, so if we were in a table, we've reached its end
            if in_table:
                # Process the completed table if it has at least 2 rows
                if len(current_table_lines) >= 2:
                    table_data = _process_table_lines(current_table_lines)
                    if table_data and len(table_data) > 0:
                        row_specs: list[list[dict[str, str | int | bool]]] = []
                        for row_idx, row in enumerate(table_data):
                            row_specs.append(
                                [
                                    {
                                        "text": cell,
                                        "rowspan": 1,
                                        "colspan": 1,
                                        "is_heading": row_idx == 0 or col_idx == 0,
                                    }
                                    for col_idx, cell in enumerate(row)
                                ]
                            )

                        table = _build_table_data_from_specs(row_specs)
                        if table:
                            parsed_tables.append(table)
                in_table = False

    # Process the last table if we're still tracking one at the end of the file
    if in_table and len(current_table_lines) >= 2:
        table_data = _process_table_lines(current_table_lines)
        if table_data and len(table_data) > 0:
            row_specs = []
            for row_idx, row in enumerate(table_data):
                row_specs.append(
                    [
                        {
                            "text": cell,
                            "rowspan": 1,
                            "colspan": 1,
                            "is_heading": row_idx == 0 or col_idx == 0,
                        }
                        for col_idx, cell in enumerate(row)
                    ]
                )

            table = _build_table_data_from_specs(row_specs)
            if table:
                parsed_tables.append(table)

    return parsed_tables


def _process_table_lines(table_lines: list[str]) -> list[list[str]]:
    """
    Process a list of lines that potentially form a markdown table.

    Args:
        table_lines: List of strings, each representing a line in a potential markdown table

    Returns:
        A list of rows, each a list of cell values
    """
    table_data = []
    separator_row_index = None

    # First, identify the separator row (the row with dashes)
    for i, line in enumerate(table_lines):
        # Check if this looks like a separator row (contains mostly dashes)
        content_without_pipes = line.replace("|", "").strip()
        if content_without_pipes and all(c in "- :" for c in content_without_pipes):
            separator_row_index = i
            break

    # Process each line, filtering out the separator row
    for i, line in enumerate(table_lines):
        # Skip the separator row
        if i == separator_row_index:
            continue

        # Skip lines that are entirely formatting
        if line.strip() and all(c in "- :|" for c in line):
            continue

        # Process the cells in this row
        cells = [cell.strip() for cell in line.split("|")]

        # Remove empty cells at the beginning and end (caused by leading/trailing pipes)
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]

        if cells:  # Only add non-empty rows
            table_data.append(cells)

    return table_data


def parse_html_tables(html_content: str) -> list[TableData]:
    """
    Extract and parse all HTML tables from the provided content.
    Identifies header rows and columns, and maps them properly handling rowspan/colspan.

    Args:
        html_content: The HTML content containing tables

    Returns:
        A list of TableData objects, each containing the table data and header information
    """
    soup = BeautifulSoup(html_content, "html.parser")
    tables = soup.find_all("table")

    parsed_tables = []

    for table in tables:
        rows = table.find_all("tr")
        if not rows:
            continue

        row_specs: list[list[dict[str, str | int | bool]]] = []
        total_rows = len(rows)

        for row_idx, row in enumerate(rows):
            cells = row.find_all(["th", "td"], recursive=False)
            heading_context = row.find_parent("thead") is not None

            row_spec: list[dict[str, str | int | bool]] = []
            for cell in cells:
                for br in cell.find_all("br"):
                    br.replace_with("\n")

                text = cell.get_text(separator="").strip()
                raw_rowspan = cell.get("rowspan")
                raw_colspan = cell.get("colspan")

                rowspan = _safe_span_int(raw_rowspan, 1)
                colspan = _safe_span_int(raw_colspan, 1)

                # HTML specifies rowspan=0 to extend to the end of the table section
                if isinstance(raw_rowspan, str) and raw_rowspan.strip() == "0":
                    rowspan = max(1, total_rows - row_idx)

                is_heading = cell.name == "th" or heading_context

                row_spec.append({"text": text, "rowspan": rowspan, "colspan": colspan, "is_heading": is_heading})

            row_specs.append(row_spec)

        table_data = _build_table_data_from_specs(row_specs)
        if table_data:
            parsed_tables.append(table_data)

    return parsed_tables
