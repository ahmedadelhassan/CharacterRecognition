#! /usr/bin/env python

__author__ = 'thovo'

def long_substr(data):
    substr = ''
    if len(data) > 1 and len(data[0]) > 0:
        for i in range(len(data[0])):
            for j in range(len(data[0])-i+1):
                if j > len(substr) and all(data[0][i:i+j] in x for x in data):
                    substr = data[0][i:i+j]
    return substr

def is_substr(find, data):
    if len(data) < 1 and len(find) < 1:
        return False
    for i in range(len(data)):
        if find not in data[i]:
            return False
    return True


def find_subsequence():
    file = open("freeman_code.txt", "r")
    data = []
    for line in file:
        number, freeman_code = line.split(",")
        if str(number) == "8":
            data.append(freeman_code)
    file.close()

    print long_substr(data)



find_subsequence()