from flask import Flask, request,make_response,render_template, redirect, url_for
from werkzeug.utils import secure_filename # 使用这个是为了确保filename是安全的
from os import path
import os
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'video')

@app.route("/",methods=['GET','POST'])
def upload():
    if request.method=='POST':
        f = request.files["file"]
        base_path = path.abspath(path.dirname(__file__))
        file_name = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], file_name))
        return redirect(url_for('upload'))
    return render_template('test.html')

if __name__ == '__main__':
    app.run(debug=True)  # 普通启动