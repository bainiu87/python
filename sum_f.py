#!/usr/bin/env python
# -*- coding: utf-8 -*-
import string
b=[]
with open('school.txt','r',encoding='utf-8') as l:
    for i in l.readlines()[1:]:
        a=i.split(',')
        su=int(a[1])+int(a[2])+int(a[3].strip())
        a.append(str(su))
        a[3]=a[3].strip()
        b.append(a)

c=sorted(b,key=lambda k:int(k[4]))

with open('sum.txt','w',encoding='utf-8') as l:
    l.write("姓名，语文，数学，英语，总分\n")
    for i in c:
        d=','.join(i)
        l.write(d+'\n')

