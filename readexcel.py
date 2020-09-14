import xlrd

excel_path = "result.xls"

# 打开文件，获取excel文件的workbook（工作簿）对象
excel = xlrd.open_workbook(excel_path, encoding_override="utf-8")

# 获取sheet对象
all_sheet = excel.sheets()

# 循环遍历每个sheet对象
for sheet in all_sheet:
    # print("该Excel共有{0}个sheet,当前sheet名称为{1},该sheet共有{2}行,{3}列"
    #       .format(len(all_sheet), sheet.name, sheet.nrows, sheet.ncols))
    for x in range (sheet.nrows):
        for y in range(sheet.ncols):
            sheet_cell = sheet.cell(x, y)
            sheet_cell_value = sheet_cell.value  # 返回单元格的值
            print(sheet_cell_value)
    print("=w=")
