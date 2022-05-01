# ================================================================
#
#   Editor      : Pycharm
#   File name   : pachong
#   Author      : LLLLLuuusa(HuangDaxu)
#   Created date: 2021-11-4 17:02
#   Email       : 1095663821@qq.com
#   QQ          : 1095663821
#   Description : 爬虫
#
#     (/≧▽≦)/ long mine the sun shine!!!
# ================================================================

import requests
headers={
    'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Mobile Safari/537.36'
}
url='https://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&is=0%2C0&fp=detail&logid=8188458688939100239&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=0&lpn=0&st=-1&word=%E5%A4%A7%E6%A3%9A%E8%A5%BF%E7%93%9C&z=0&ic=&hd=&latest=&copyright=&s=undefined&se=&tab=0&width=&height=&face=undefined&istype=2&qc=&nc=&fr=&simics=&srctype=&bdtype=0&rpstart=0&rpnum=0&cs=1903062189%2C2674459872&catename=&nojc=undefined&album_id=&album_tab=&cardserver=&tabname=&pn=510&rn=60&gsm=21f&1634135303710='
response=requests.get(url,headers=headers)
p_url=response.json()['data']
for i in range(29):
    i=p_url[i]['thumbURL']
    print(i)
    res=requests.get(url=i,headers=headers)
    with open('美图\\'+i[29:36]+'.jpeg',mode='wb')as f:
        f.write(res.content)