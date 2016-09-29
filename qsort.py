#!/usr/bin/python
# -*- coding: utf-8 -*-
def partition(item,left,right):
    if (left==right-1):
        return left
    pos=left
    key=item[left]
    pl=left
    pr=right
    while pl<pr:
        while pl<right and item[pl]<=key:
            pl += 1
        while pr>=left and item[pr]>key:
            pr -= 1
        if pl<pr:
            a=item[pl]
            item[pl]=item[pr]
            item[pr]=a
    b=item[pos]
    item[pos]=item[pr]
    item[pr]=b
    return pr
def qsort(item,left,right):
    if left>=right:
        return
    post=partition(item,left,right)
    qsort(item,left,post)
    qsort(item,post+1,right)
    return item

if __name__ == '__main__':
    item=[3,5,2,6,8,1]
    print(qsort(item,0,5))



