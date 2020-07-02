import pandas as pd


def convert_to_number(s):
    hex_code = int.from_bytes(s.encode(), 'little')
    hex_code_str = str(hex_code)
    hex_code_list = map(int, hex_code_str)
    hex_code_sum = sum(hex_code_list)
    return int(hex_code_sum)


input_file_path = '/Users/navomsaxena/Downloads/ICE_budget_sheet_loading_/' \
            '7000009_Vifor_Diamond_WO_Internal_Budget_V.12_14Dec2018 WBS3.0.xlsm'
xl = pd.read_excel(input_file_path, sheet_name=None)

sheets = xl.keys()
file_id = convert_to_number(input_file_path)
file_name = input_file_path.split("/")[-1]
project_code = file_name.split('_')[0]

df = pd.DataFrame(sheets, columns=['sheets'])
df['FileID'] = pd.Series(file_id, index=df.index)
df['SheetID'] = df.apply(lambda row: convert_to_number(str(row.sheets)), axis=1)
df['FileName'] = pd.Series(file_name, index=df.index)
df['ProjectCode'] = pd.Series(project_code, index=df.index)
df['TemplateVersion'] = pd.Series('None', index=df.index)

cols = df.columns.tolist()
cols = cols[1:] + cols[0:1]
df = df[cols]
