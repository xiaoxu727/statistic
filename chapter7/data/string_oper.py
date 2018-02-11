# 字符串操作
import numpy as np
import re
from utils.logger.logger import logger
from pandas import Series
logger.info('字符串对象方法')
val = 'a,b, bank re'
s = [x.strip() for x in val.split(',')]
print(s)
print(type(s))
t = set()
print(type(t))
print("::".join(s))
print( 'a' in val)
# index 找不到会返回异常，而不是-1
try:
    print(val.index('c'))
except Exception:
    print('error')
print(val.find('c'))
print(val.count(','))
print(val.replace(',',';'))
print(val.rjust(10))
print(val.ljust(10))

logger.info('正则表达式')
regex = re.compile(',|\s+')
print(re.split(',|\s+', val))
print(regex.split(val))
print(regex.findall(val))
# search 返回第一个
# match 从头进行匹配
# findall 返回所有匹配项

text='发件人: pd<pd@champion-credit.com> 收件人' \
     ':李甜甜<litiantian@champion-credit.com>' \
     '抄送:tianyilin<tianyilin@champion-credit.com>,   xujunjie<xujunjie@champion-credit.com>,   xuxingyuan<xuxingyuan@champion-credit.com>' \
     '时间:2018年2月9日 (周五) 15:30 大小:70 KB'
regex_mail = re.compile(r'([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})', flags=re.IGNORECASE)
print(regex_mail.findall(text))
print(regex_mail.search(text))
print(regex_mail.match(text))
print(regex_mail.sub('EMAIL', text))
print(regex_mail.match('pd@champion-credit.com').groups())
print(regex_mail.sub(r'username:\1, Doamin:\2, suffix:\3', text))

logger.info('label')
regex =re.compile(r"""(?P<username>[A-z0-9._%+-]+)@(?P<domain>[A-Z0-9.-]+)\.(?P<suffix>[A-Z]{2,4})""",flags=re.IGNORECASE|re.VERBOSE)
print(regex.match("pd@champion-credit.com").groupdict())
print(regex.search(text).groupdict())

logger.info('pandas 中矢量化的字符串函数')

data = {'Dave':'pd@champion-credit.com',
        'tianyilin':'tianyilin@champion-credit.com',
        'xujunjie':'xujunjie@champion-credit.com',
        'mayu': np.nan}

data = Series(data)
print(data.isnull())
print(data.str.contains('com'))
logger.info('matches')
pattern = r'([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})';
findall_data = data.str.findall(pattern,flags = re.IGNORECASE);
print(data.str.findall(pattern,flags = re.IGNORECASE))

print(data)
matches = data.str.match(pattern, flags=re.IGNORECASE)
print(matches)
logger.info('get')
print(matches.str.get(0))
print(findall_data.str.get(0))