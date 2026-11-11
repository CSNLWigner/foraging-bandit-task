"""
Export of the source data of the figures (the 'Source Data' file of the paper).

The notebooks call export_source_data() after plotting a panel. Each panel is saved as CSV table(s) in
source_data/csv/<panel>/, and all panels are collected into source_data/Source Data.xlsx, with one sheet per
figure panel (e.g. 'Figure 2a'), following the Nature Communications example: the panel name and a short description
on top, then the data table(s), each with a title and a header row.
"""
import json
import os
import re

import pandas as pd

from popy.config import PROJECT_PATH_LOCAL

SOURCE_DATA_PATH = os.path.join(PROJECT_PATH_LOCAL, 'source_data')
EXCEL_NAME = 'Source Data.xlsx'


def _panel_key(panel):
    """sort key for panel names like 'Figure 2e-h' or 'Supplementary Figure 3' (main figures first)"""
    m = re.match(r'(Supplementary )?Figure (\d+)(.*)', panel)
    return (m.group(1) is not None, int(m.group(2)), m.group(3)) if m else (True, 999, panel)


def export_source_data(panel, data, description, build_excel=True):
    """
    Save the data plotted in a figure panel.

    panel: name of the panel, used as sheet name, e.g. 'Figure 2a' or 'Supplementary Figure 3' (max. 31 characters)
    data: a DataFrame, or a dict {table title: DataFrame} if the panel has several tables (e.g. one per monkey)
    description: short description of what is plotted (variables, units, error bars, n)
    """
    assert re.fullmatch(r'(Supplementary )?Figure \d+[a-z]?(-[a-z])?', panel), \
        f'panel name should be like "Figure 2a", "Figure 2e-h" or "Supplementary Figure 3b": {panel}'
    tables = data if isinstance(data, dict) else {'': data}
    folder = os.path.join(SOURCE_DATA_PATH, 'csv', panel.replace(' ', '_'))
    os.makedirs(folder, exist_ok=True)
    for f in os.listdir(folder):  # replace the previous export of this panel
        os.remove(os.path.join(folder, f))
    titles = []
    for i, (title, df) in enumerate(tables.items()):
        df = pd.DataFrame(df)
        df.to_csv(os.path.join(folder, f'table_{i + 1}.csv'), index=False)
        titles.append(title)
    with open(os.path.join(folder, 'info.json'), 'w') as f:
        json.dump({'panel': panel, 'description': description, 'tables': titles}, f, indent=1)
    print(f'{panel}: source data saved ({", ".join(f"{len(pd.DataFrame(df))} rows" for df in tables.values())})')
    if build_excel:
        build_source_data_file()


def _read_table(path):
    """read a saved table as text, and convert to numbers only the columns where this loses nothing
    (keeps e.g. session IDs with leading zeros like '010720', and labels like 'None')"""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in df.columns:
        values = df[col]
        num = pd.to_numeric(values.replace('', None), errors='coerce')
        filled = values != ''
        if num[filled].notna().all() and not values[filled].str.match(r'^-?0\d').any():
            df[col] = num
        else:
            df[col] = values.replace('', None)
    return df


def build_source_data_file():
    """Collect all exported panels into source_data/Source Data.xlsx (one sheet per panel)."""
    from openpyxl import Workbook
    from openpyxl.styles import Font

    csv_root = os.path.join(SOURCE_DATA_PATH, 'csv')
    infos = []
    for name in os.listdir(csv_root) if os.path.isdir(csv_root) else []:
        p = os.path.join(csv_root, name, 'info.json')
        if os.path.exists(p):
            infos.append((json.load(open(p)), os.path.join(csv_root, name)))
    infos.sort(key=lambda x: _panel_key(x[0]['panel']))

    wb = Workbook()
    wb.remove(wb.active)
    bold = Font(bold=True)
    for info, folder in infos:
        ws = wb.create_sheet(info['panel'])
        ws.cell(row=1, column=1, value=info['panel']).font = bold
        ws.cell(row=2, column=1, value=info['description'])
        row = 4
        for i, title in enumerate(info['tables']):
            df = _read_table(os.path.join(folder, f'table_{i + 1}.csv'))
            if title:
                ws.cell(row=row, column=1, value=title).font = bold
                row += 1
            for j, col in enumerate(df.columns):
                ws.cell(row=row, column=j + 1, value=str(col)).font = bold
            for values in df.itertuples(index=False):
                row += 1
                for j, v in enumerate(values):
                    ws.cell(row=row, column=j + 1, value=None if pd.isna(v) else (v.item() if hasattr(v, 'item') else v))
            row += 2
    os.makedirs(SOURCE_DATA_PATH, exist_ok=True)
    wb.save(os.path.join(SOURCE_DATA_PATH, EXCEL_NAME))
