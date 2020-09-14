from flask import Flask, render_template
from random import *
app = Flask(__name__)

# message = "hi"
@app.route("/hello", methods=['GET', 'POST'])
def Hello():

    message = "hi"
    num = randint(1, 100)
    print(num)
    return render_template("hello.html", temp=num)


if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=8987, debug=True)
    app.run(debug=True)

