import os
import igl
import numpy as np

def delLastLine(path):
    with open(path, "rb+") as f:
        lines = f.readlines()  # 读取所有行
        last_line = lines[-1]  # 取最后一行
        for i in range(len(last_line) + 2):  ##愚蠢办法，但是有效
            f.seek(-1, os.SEEK_END)
            f.truncate()
    return

def write_point(name,verts):
    igl.write_obj(name,verts, np.array([[1, 2, 3]]))
    delLastLine(name)