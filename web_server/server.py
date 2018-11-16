#!/usr/bin/python3
# -*- coding: utf-8 -*-

import socket

# 资源根目录
root = './example'

# 获取响应报文
def getResponse(reqDict):
    method = reqDict['method']
    res = ''

    if method == 'GET':
        res = get(reqDict)
    elif method == 'POST':
        print ('Getpost')
        res = post(reqDict)
    else:
        pass

    return res

# GET
def get(reqDict):
    headers = 'HTTP/1.1 200 OK\r\n'
    path = '/index.html' if reqDict['sourcePath'] == '/' else reqDict['sourcePath']
    content = ''

    content = content.encode('utf-8')

    # 获取资源内容
    try:
        f = open(root + path, 'rb')
        while True:
            chunk = f.read(1024)
            if not chunk:
                f.close()
                break;
            content += chunk
    except:
        pass


    sourceType = path.split('/')[-1].split('.')[-1]
    htmlContentType = "text/html; charset=utf-8"
    cssContentType = "text/css"
    jsContentType = "application/javascript"
    contentType = ''

    if sourceType == 'html':
        contentType = htmlContentType
    elif sourceType == 'css':
        contentType = cssContentType
    elif sourceType == 'js':
        contentType = jsContentType
    else:
        contentType = '*'

    content = content.decode('utf-8')

    res = headers + 'Content-Type: ' + contentType + '\r\n' + 'Content-Length: ' + str(len(content)) + '\r\n\r\n' + content

    return res

def post(reqDict):
    x = web.input(myfile={})
    filedir = './static' # change this to the directory you want to store the file in.
    if 'myfile' in x: # to check if the file-object is created
        filepath=x.myfile.filename.replace('\\','/') # replaces the windows-style slashes with linux ones.
        filename=filepath.split('/')[-1] # splits the and chooses the last part (the filename with extension)
        fout = open(filedir +'/'+ filename,'wb') # creates the file where the uploaded file should be stored
        fout.write(x.myfile.file.read()) # writes the uploaded file to the newly created file.
        fout.close() # closes the file, upload complete.

        infile = filedir +'/'+filename
        outfile = infile + ".thumbnail"
        im = Image.open(filedir +'/'+filename)
        im.thumbnail((128, 128))
        im.save(outfile, im.format)

    return render.upload(outfile)

# 解析请求报文
def parseReq(reqList):
    # 保存解析结果
    parseRet = {}

    method = reqList[0].split(' ')[0]
    sourcePath = reqList[0].split(' ')[1]

    parseRet['method'] = method
    parseRet['sourcePath'] = sourcePath

    i = len(reqList) - 1

    while i:
        if len(reqList[i].split(':')) == 1:
            i = i - 1
            continue

        idx = reqList[i].find(':')
        key, value = reqList[i][0:idx], reqList[i][idx+1:]
        parseRet[key] = value.strip()
        i = i - 1

    return parseRet


# 开启服务器
def serve():
    sk = socket.socket(
        socket.AF_INET, 
        socket.SOCK_STREAM
    )

    host = '127.0.0.1'
    port = 8888

    sk.bind((host, port))
    sk.listen(5)

    while True:
        try:
            clientSk, addr = sk.accept()
            print("address is: %s" % str(addr))

            requestList = clientSk.recv(1024).decode().split("\r\n")
            print('*****requestList************')
            print (requestList)
            
            # 解析HTTP请求报文
            ret = parseReq(requestList)
            print('=============retList==================')
            print (ret)

            # 获取响应报文
            response = getResponse(ret)
            print('@@@@@@@@@@@@@@@@@response@@@@@@@@@@@@@')
            print (response)
            
            # 返回HTTP响应报文
            clientSk.sendall(response.encode('utf-8'))
            clientSk.close()

        except Exception as err:
            print(err)
            clientSk.close()


if __name__ == '__main__':
    serve()