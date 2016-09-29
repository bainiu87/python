#!/usr/bin/python
# -*- coding: utf-8 -*-
from operator import itemgetter,attrgetter
item=[{'name':'sf','age':10},{'name':'tf','age':63},{'name':'sk','age':12},{'name':'qqp','age':54},{'name':'sb','age':34},{'name':'line','age':43}]
item_1=[['100','24','23'],['23','100','22']]
item_2=['1','100','2']
d=sorted(item_1,key=itemgetter(0))
b=sorted(item,key=lambda abc:abc['age'])
c=sorted(item,key=itemgetter('name'))
a=sorted(item_2)
if __name__ =='__main__':
    print(a)