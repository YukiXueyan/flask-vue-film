from flask import Flask, request,make_response,render_template, redirect, url_for
from werkzeug.utils import secure_filename # 使用这个是为了确保filename是安全的
from os import path
import os
from random import *
import xlrd
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'video')

@app.route("/",methods=['GET','POST'])
def upload():
    if request.method=='POST':
        f = request.files["file"]
        base_path = path.abspath(path.dirname(__file__))
        # file_name = secure_filename(f.filename)
        file_name = f.filename
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], file_name))
        return redirect(url_for('upload'))
    return render_template('hello.html')

@app.route("/hello", methods=['GET', 'POST'])
def Hello():

    message = "hi"
    num = randint(1, 100)
    return render_template("RandomNum.html", temp=num)

@app.route("/readexcel", methods=['GET', 'POST'])
def readexcel():
    excel_path = "result.xls"

    # 打开文件，获取excel文件的workbook（工作簿）对象
    excel = xlrd.open_workbook(excel_path, encoding_override="utf-8")

    # 获取sheet对象
    all_sheet = excel.sheets()

    # 循环遍历每个sheet对象
    for sheet in all_sheet:
        # print("该Excel共有{0}个sheet,当前sheet名称为{1},该sheet共有{2}行,{3}列"
        #       .format(len(all_sheet), sheet.name, sheet.nrows, sheet.ncols))
        for x in range(sheet.nrows):
            for y in range(sheet.ncols):
                sheet_cell = sheet.cell(x, y)
                sheet_cell_value = sheet_cell.value  # 返回单元格的值
                print(sheet_cell_value)
                data = sheet_cell_value

    sheet_cell = sheet.cell(0, 0)
    sheet_cell_value = sheet_cell.value  # 返回单元格的值
    data = sheet_cell_value
    return render_template("hello.html", temp=data)

if __name__ == '__main__':
    app.run(debug=True)  # 普通启动