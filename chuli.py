#!/usr/bin/env python
# -*- coding: utf-8 -*-
def gogo(arr,target,start=0,over=0):
    sum_r=0
    sum_f=0
    if  over == 100:
        for i in arr:
            if int(start) <= int(i[int(target)]) <= int(over):
                sum_r += 1
                sum_f += int(i[int(target)])
        ave = round(sum_f / sum_r,2)
        str = "[%s,%s]分数段的总人数是%s,平均分是%s" % (start, over, sum_r, ave)
    else:
        for i in arr:
            if int(start) <= int(i[int(target)]) < int(over):
                sum_r+=1
                sum_f+=int(i[int(target)])
        ave=round(sum_f/sum_r,2)
        str="[%s,%s)分数段的总人数是%s,平均分是%s"%(start,over,sum_r,ave)
    return str

al=[]
with open('school.txt','r',encoding='utf-8') as l:
    info=l.readlines()[1:]
    for i in info:
        i=i.strip()
        i_list=i.split(',')
        al.append(i_list)
print('语文;'+gogo(al,1,0,60)+";"+gogo(al,1,60,75)+";"+gogo(al,1,75,80)+";"+gogo(al,1,80,100))
print('数学;'+gogo(al,2,0,60)+";"+gogo(al,2,60,75)+";"+gogo(al,2,75,80)+";"+gogo(al,2,80,100))
print('英语;'+gogo(al,3,0,60)+";"+gogo(al,3,60,75)+";"+gogo(al,3,75,80)+";"+gogo(al,3,80,100))




