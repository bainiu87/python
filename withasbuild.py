#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
with open('school.txt','w',encoding='utf-8') as L:
    head='{name},{yu},{sx},{yin}\n'.format(name='名字',yu='语文',sx='数学',yin='英语')
    L.write(head)
    for i in range(1,101):
        str = 'zxcvbnmasdfghjklqwertyuiop'
        name = ''.join(random.sample(str, 4))
        yu = random.randint(20, 100)
        sx = random.randint(20, 100)
        yi = random.randint(20, 100)
        info='{name},{yu},{sx},{yin}\n'.format(name=name,yu=yu,sx=sx,yin=yi)
        L.write(info)



