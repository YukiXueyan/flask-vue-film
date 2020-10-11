from flask import Flask, request,make_response,render_template, redirect, url_for
from werkzeug.utils import secure_filename # 使用这个是为了确保filename是安全的
from os import path
import os
from random import *
import xlrd
import pymysql
import http.client
import json
import os,sys
import ssl
import urllib
import urllib.parse
import xlwt
import time
import cv2
import xlrd
from sklearn import preprocessing, model_selection
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
from PIL import Image
from PIL import ImageEnhance
import PIL.Image as video_img
import pandas as pd
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'video')
'''
实现文件上传
'''

@app.route("/",methods=['GET','POST'])
def upload():
    if request.method=='POST':
        f = request.files["file"]
        base_path = path.abspath(path.dirname(__file__))
        file_name = f.filename
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], file_name))
        return redirect(url_for('upload'))
    return render_template('index.html')


'''
测试用函数
'''
@app.route("/hello", methods=['GET', 'POST'])
def hello():

    message = "hi"
    num = randint(1, 100)
    return render_template("RandomNum.html", temp=num)
'''
处理上传的视频
'''
@app.route("/getfilm",methods=['GET','POST'])
def getfilm():
    # ------------------------------------视频分帧----------------------------------------#

    # 视频源文件路径
    videos_src_path = 'Video'
    # 视频分帧图片父级保存路径
    videos_save_path = 'picture'
    # 获取视频源文件路径下的所有视频文件
    videos = os.listdir(videos_src_path)
    # 对各视频进行排序
    # videos.sort(key=lambda x: int(x[5:-4]))

    i = 1

    for each_video in videos:
        # 生成单个视频分帧图片保存路径
        if not os.path.exists(videos_save_path + '/' + str(i)):
            os.mkdir(videos_save_path + '/' + str(i))
        # 视频帧全路径
        each_video_save_full_path = os.path.join(videos_save_path, str(i)) + '/'
        # 视频截取部分帧全路径
        each_video_savePart_full_path = os.path.join(videos_save_path, str(i) + 'Part') + '/'
        # 视频全路径
        each_video_full_path = os.path.join(videos_src_path, each_video)
        # 创建一个视频读写类 读入视频文件
        cap = cv2.VideoCapture(each_video_full_path)
        # 初始化第一个帧号
        frame_count = 1
        success = True

        #
        # # 读取视频的fps(帧率),  size(分辨率)
        # fps = cap.get(cv2.CAP_PROP_FPS)
        # size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # print("fps: {}\n size: {}".format(fps, size))
        #
        # # 读取视频时长（帧总数）
        # total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # print("[INFO] {} total frames in video".format(total))
        #
        # # 设定从视频的第几帧开始读取
        # # From :  https://blog.csdn.net/luqinwei/article/details/87973472
        # frameToStart = 2000
        # cap.set(cv2.CAP_PROP_POS_FRAMES, frameToStart);
        #

        while (success):
            success, frame = cap.read()
            # 存储为图像
            if success == True:
                cv2.imwrite(each_video_save_full_path + "frame%d.jpg" % frame_count,
                            frame)
                # 对图片进行区域裁剪的代码块
                IMG = each_video_save_full_path + "frame%d.jpg" % frame_count  # 图片地址
                # 打印图片地址
                print(IMG)
                im = video_img.open(IMG)  # 用PIL打开一个图片
                box = (0, 0, 318, 360)  # box代表需要剪切图片的位置格式为:xmin ymin xmax ymax
                ng = im.crop(box)  # 对im进行裁剪 保存为ng(这里im保持不变)
                # ng = ng.rotate(20)  # ng为裁剪出来的图片，进行向左旋转20度 向右为负数
                # 创建用于存放裁剪后的图片的文件夹
                if not os.path.exists(each_video_savePart_full_path):
                    os.mkdir(each_video_savePart_full_path)
                # 打印裁剪后的图片地址
                partPath = each_video_savePart_full_path + "frame%d.jpg" % frame_count
                print(partPath)
                # 存储裁剪后的图片
                ng.save(each_video_savePart_full_path + "frame%d.jpg" % frame_count)
            # 裁剪并保存结束

            # 设置分帧间隔(捕获本视频的下一帧)
            frame_count = frame_count + 500
        # 处理下一个视频
        i = i + 1

        cap.release()

    # -------------------------------------调用API && 照片加工--------------------------------#
    ssl._create_default_https_context = ssl._create_unverified_context
    # urllib打开http链接会验证SSL证书，全局取消证书验证防止异常
    subscription_key = '144ce86219b740938b003a1f3d36a26a'  # Face API的key
    uri_base = 'https://aimovie.cognitiveservices.azure.cn/'  # Face API的end point
    # path = '/Users/mac/Documents/filmRating/newPicture' #新电影的处理后图片
    path = 'picture'

    def useApi(img):
        headers = {
            'Content-Type': 'application/octet-stream',
            'Ocp-Apim-Subscription-Key': subscription_key,
        }

        params = urllib.parse.urlencode({
            'returnFaceId': 'true',
            'returnFaceLandmarks': 'false',
            'returnFaceAttributes': 'age,gender,smile,emotion'
        })
        try:
            conn = http.client.HTTPSConnection('api.cognitive.azure.cn')
            conn.request("POST", "/face/v1.0/detect?%s" % params, img, headers)
            response = conn.getresponse()
            data = response.read()
            parsed = json.loads(data)  # 将字符串转化为字典
            # print("Response:")
            # print(json.dumps(parsed, sort_keys=True, indent=2))
            conn.close()

        except Exception as e:
            print("[Errno {0}] {1}".format(e.errno, e.strerror))
        return parsed

    workbook = xlwt.Workbook(encoding='utf-8')

    def writeExcel(img, worksheet, row, file_name, path1, path2):
        parsedAll = []
        parse = useApi(img)
        if len(parse) == 0:
            print("未识别到人脸，正在进行第一次加工")
            img = changePhoto(path2, path1, 1.5, 2.0, file_name, 1, )  # 未识别到人脸，第一次加工
            parse = useApi(img)
            if len(parse) == 0:
                print("未识别到人脸，正在进行第二次加工")
                img = changePhoto(path2, path1, 0.9, 2.0, file_name, 2)  # 仍未识别到人脸，第二次加工
                parse = useApi(img)
                if len(parse) == 0:
                    print('无法识别到人脸')

        parsedAll.append(parse)
        if len(parse) != 0:
            for list_item in parsedAll:
                if type(list_item) == list:  # 正确的输出结果应被转化为list类型
                    l1 = list_item[0]  # list_item里只有一个元素l1，l1是一个字典
                    filename, extension = os.path.splitext(file_name)
                    worksheet.write(row, 0, filename)  # 写入照片的文件名

                    times = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                    worksheet.write(row, 1, times)  # 写入时间戳

                    emotion = []
                    for k in l1.keys():  # l1的key值分别为faceAttributes，faceId，faceRectangle
                        if k == 'faceAttributes':
                            l2 = l1[k]  # faceAttributes的value是一个字典,赋值给l2
                            worksheet.write(row, 2, l2['gender'])  # 性别写进第3列
                            worksheet.write(row, 3, l2['age'])  # age写进第4列

                            l3 = l2['emotion']  # emotion的value是一个字典，赋值给l3
                            for emotion_k in l3.keys():
                                emotion.append(l3[emotion_k])  # 把所有情绪识别分数存入emotion数组

                            max_emotion = 0
                            for i in range(len(emotion)):
                                worksheet.write(row, 4 + i, str(emotion[i]))
                                if emotion[i] > max_emotion:
                                    max_emotion = emotion[i]  # 获取得分最高的情绪

                            for i in range(len(emotion)):
                                if max_emotion == emotion[i]:
                                    worksheet.write(row, 12, str(i))  # 记录得分最高的情绪编号
                                    # 0-anger,1-contempt,2-disgust,3-fear,4-happiness,5-neutral,6-sadness,7-surprise

                        elif k == 'faceId':
                            worksheet.write(row, 13, l1['faceId'])  # faceId写进第14列
                        else:
                            pass

                    print('图片:' + str(file_name) + '已处理完毕')
                else:
                    pass
                    row += 1

        return row, worksheet

    def changePhoto(imgPath, folderPath, bri, sharp, file_name, i):
        img = Image.open(imgPath)
        im_2 = ImageEnhance.Brightness(img).enhance(bri)
        im_3 = ImageEnhance.Sharpness(im_2).enhance(sharp)
        print("第" + str(i) + "次加工完毕！")
        if not os.path.exists(imgPath):
            os.mkdir(imgPath)
        im_3.save(os.path.join(folderPath, file_name))
        print("修改后的图片保存成功！")
        im = open(os.path.expanduser(path2), 'rb')
        return im

    for root, dirs, files in os.walk(path, topdown=False):
        # 创建生成器，查找目录及子目录下所有文件,root-文件夹路径，dirs-文件夹名字，files-文件名
        for folder in dirs:
            error_num = 0
            error_list = []
            row = 0
            file_num = 0
            print('现在开始处理文件夹：' + folder)
            worksheet = workbook.add_sheet(folder)

            title = ['PhotoID', 'Time', 'gender', 'age', 'anger', 'contempt', 'disgust',
                     'fear', 'happiness', 'neutral', 'sadness', 'surprise', 'emotion', 'faceID']  # 设置表头

            for col in range(len(title)):
                worksheet.write(0, col, title[col])
            for root2, dirs2, files2 in os.walk(path + '/' + folder):
                for file_name in files2:
                    if file_name != '.DS_Store':
                        try:
                            path1 = path + '/' + folder
                            path2 = path + '/' + folder + '/' + file_name
                            print('现在处理' + folder + '中的图片:' + str(file_name))
                            img = open(os.path.expanduser(path2), 'rb')  # 打开本地图片
                            row, worksheet = writeExcel(img, worksheet, row, file_name, path1, path2)
                            file_num += 1
                        except Exception as e:
                            print(e)
            print('文件夹：' + folder + '已处理完毕')
            print(error_num, error_list)  # 异常个数、异常内容

    workbook.save('Face.xls')  # 没有打分的新电影的情绪分
    print('处理完毕')

    def excel_to_matrix(path):
        rows = 0
        cols = 0
        for table in xlrd.open_workbook(path).sheets():
            rows += table.nrows - 1
            cols = table.ncols
        datamatrix = np.zeros((rows, cols))

        for table in xlrd.open_workbook(path).sheets():
            # # table = xlrd.open_workbook(path).sheets()[0]  # 获取第一个sheet表
            row = table.nrows - 1  # 行数
            # print(row)
            col = table.ncols  # 列数
            # datamatrix = np.zeros((row, col))  # 生成一个nrows行ncols列，且元素均为0的初始矩阵
            for y in range(row):
                for x in range(col):
                    # 把list转换为矩阵进行矩阵操作
                    datamatrix[y, x] = table.row(y + 1)[x].value  # 按列把数据存进矩阵中
            # 数据归一化
            # min_max_scaler = preprocessing.MinMaxScaler()
            # datamatrix = min_max_scaler.fit_transform(datamatrix)
            return datamatrix

    # ------------------------------将emotion数据转换为一行九列格式------------------------------#
    workBook = xlwt.Workbook(encoding='utf-8')
    count = 0
    col = 0
    data_path = 'Face.xls'

    document = xlrd.open_workbook(data_path)
    allSheetNames = document.sheet_names()
    print(allSheetNames)

    def writeTitle(sheet):

        title = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9']
        for col in range(len(title)):
            sheet.write(0, col, title[col])

    def writeEmotion(sheet, data):
        global count
        row = 1
        col = 0
        for i in data:
            count += 1
            if (col < 9):
                sheet.write(row, col, i)
                col += 1
            else:
                row += 1
                col = 0
                sheet.write(row, col, i)
                col += 1

    for i in range(len(allSheetNames)):
        workSheet = workBook.add_sheet(allSheetNames[i])
        writeTitle(workSheet)
        content = document.sheet_by_index(i)
        print(content.name, content.nrows, content.ncols)

        data = []
        for a in range(content.nrows):
            cells = content.row_values(a)
            data.append(cells[12])  # 第12列是该图片的情绪类别编号

        del data[0]
        print(data)
        writeEmotion(workSheet, data)

    workBook.save('newEmotion.xls')
    print('excel格式转换完毕')

    # ------------------------------------训练模型---------------------------------------#
    datafile = "trainEmotion.xls"  # 五部电影的训练数据，包含情绪分+志愿者打分
    matrix = excel_to_matrix(datafile)
    x = matrix[:, 0:8]
    y = matrix[:, 9]

    # 分割训练数据和测试数据
    # 随机采样25%作为测试 75%作为训练
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.25, random_state=33)

    # 训练数据和测试数据进行标准化处理
    ss_x = StandardScaler()
    x_train = ss_x.fit_transform(x_train)
    x_test = ss_x.transform(x_test)

    ss_y = StandardScaler()
    y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
    y_test = ss_y.transform(y_test.reshape(-1, 1))

    print(ss_y.mean_)
    print(ss_y.scale_)
    # 支持向量机模型进行学习和预测
    # 线性核函数配置支持向量机
    linear_svr = SVR(kernel="linear")
    # 训练
    linear_svr.fit(x_train, np.ravel(y_train))
    # 预测 保存预测结果
    linear_svr_y_predict = linear_svr.predict(x_test)

    # 多项式核函数配置支持向量机
    poly_svr = SVR(kernel="poly")
    # 训练
    poly_svr.fit(x_train, np.ravel(y_train))
    # 预测 保存预测结果
    poly_svr_y_predict = poly_svr.predict(x_test)

    y_test = ss_y.inverse_transform(y_test)
    poly_svr_y_predict = ss_y.inverse_transform(poly_svr_y_predict)
    print(y_test)
    print(poly_svr_y_predict)

    print("对多项式核函数的均方误差为:", mean_squared_error(y_test, poly_svr_y_predict))
    print("对多项式核函数的平均绝对误差为:", mean_absolute_error(y_test, poly_svr_y_predict))

    joblib.dump(poly_svr, "model.m")
    print("训练完成\n")

    # --------------------------使用模型（输入新电影的情绪分）----------------------------#
    def excel_to_matrix(path):
        poly_svr = joblib.load("model.m")
        rows = 0
        cols = 0
        for table in xlrd.open_workbook(path).sheets():
            rows += table.nrows - 1
            cols = table.ncols
        datamatrix = np.zeros((rows, cols))

        for table in xlrd.open_workbook(path).sheets():
            names = xlrd.open_workbook(path).sheet_names()
            counter = 0
            # # table = xlrd.open_workbook(path).sheets()[0]  # 获取第一个sheet表
            row = table.nrows - 1  # 行数
            # print(row)
            col = table.ncols  # 列数
            # datamatrix = np.zeros((row, col))  # 生成一个nrows行ncols列，且元素均为0的初始矩阵
            for y in range(row):
                for x in range(col):
                    # 把list转换为矩阵进行矩阵操作
                    datamatrix[y, x] = table.row(y + 1)[x].value  # 按列把数据存进矩阵中
            # 数据归一化
            ss_x = StandardScaler()
            x = ss_x.fit_transform(datamatrix[:, 0:8])
            # 预测 保存预测结果
            poly_svr_y_predict = poly_svr.predict(x)
            poly_svr_y_predict = poly_svr_y_predict * 1.33333333 + 8.0  # 反归一化
            sum = 0
            for u in range(row):
                sum += poly_svr_y_predict[u]
            wb = xlwt.Workbook(encoding='utf-8')
            ws = wb.add_sheet("Result")
            ws.write(counter, 0, names[counter])
            ws.write(counter, 1, sum / row)
        workbook.save("Result.xls")
        print("处理完毕")
        return datamatrix, cols

    datafile = "newEmotion.xls"  # 新电影情绪分，没有观众打分
    [matrix, col] = excel_to_matrix(datafile)

    # 将excel数据写入数据库-----------------------------------------------------------------------------------------------------


    excelFile = r'result.xls'
    df = pd.DataFrame(pd.read_excel(excelFile))
    from sqlalchemy import create_engine
    import pymysql

    engine = create_engine('mysql+pymysql://root:123456@localhost:3306/filmback', encoding='utf8')
    df.to_sql('filmback', con=engine, if_exists='replace', index=False)

    # 将sql展示到前端-----------------------------------------------------------------------------------------------------
    conn = pymysql.connect(
        host='127.0.0.1',
        user='root',
        password='123456',
        db='filmback',
        charset='utf8'
    )
    cur = conn.cursor()

    # get annual sales rank
    sql = "select * from filmback"
    cur.execute(sql)
    content = cur.fetchall()

    # 获取表头
    sql = "SHOW FIELDS FROM filmback"
    cur.execute(sql)
    labels = cur.fetchall()
    labels = [l[0] for l in labels]
    return render_template('readExcel.html', labels=labels, content=content)


#获取数据库中的数据，在html中用表格显示
@app.route('/film')
def film():
    # 将excel数据写入数据库-----------------------------------------------------------------------------------------------------
    excelFile = r'result.xls'
    df = pd.DataFrame(pd.read_excel(excelFile))
    from sqlalchemy import create_engine
    import pymysql

    engine = create_engine('mysql+pymysql://root:123456@localhost:3306/filmback', encoding='utf8')
    df.to_sql('filmback', con=engine, if_exists='replace', index=False)

    # 将sql展示到前端-----------------------------------------------------------------------------------------------------
    conn = pymysql.connect(
        host='127.0.0.1',
        user='root',#数据库名称
        password='123456',#数据库密码
        db='filmback',#表名称
        charset='utf8'
    )
    cur = conn.cursor()

    # get annual sales rank
    sql = "select * from filmback"
    cur.execute(sql)
    content = cur.fetchall()

    # 获取表头
    sql = "SHOW FIELDS FROM filmback"
    cur.execute(sql)
    labels = cur.fetchall()
    labels = [l[0] for l in labels]
    return render_template('readExcel.html', labels=labels, content=content)


if __name__ == '__main__':
    app.run(debug=True)  # 普通启动
