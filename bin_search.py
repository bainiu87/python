#!/usr/bin/python
# -*- coding: utf-8 -*-
import math
def bin_search(item,target):
    left=0
    right=len(item)
    while left<right:
        mid=math.floor((left+right)/2)
        if target<item[mid]:
            right=mid
        elif target>item[mid]:
            left=mid+1
        elif target == item[mid]:
            return mid
    return -1

if __name__ == '__main__':
    item=[1,2,3,4,5,6,7,8,9]
    print(bin_search(item,8))

