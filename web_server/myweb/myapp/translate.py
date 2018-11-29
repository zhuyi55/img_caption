import urllib.request
import urllib.parse
import json
import time

def translate(content):
    url = 'http://fanyi.youdao.com/translate?smartresult=dict&smartresult=rule'

    '''
    head = {}
    head['user-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36'
    '''

    data={}
    data['i'] = content
    data['from'] = 'AUTO'
    data['to']= 'AUTO'
    data['smartresult']= 'dict'
    data['client']= 'fanyideskweb'
    data['salt']= '1522949952242'
    data['sign']= '0fb5d57201838f0cdf2ecb429030dcbf'
    data['doctype']= 'json'
    data['version']= '2.1'
    data['keyfrom']= 'fanyi.web'
    data['action']= 'FY_BY_CLICKBUTTION'
    data['typoResult']= 'false'
    data = urllib.parse.urlencode(data).encode('utf-8')

    req = urllib.request.Request(url,data)
    req.add_header('user-Agent','Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36')

    response = urllib.request.urlopen(req)
    html = response.read().decode('utf-8')

    target = json.loads(html)
    target = target['translateResult'][0][0]['tgt']
    
    return target
    ##print("翻译结果： %s" % target)
    ##time.sleep(5)
