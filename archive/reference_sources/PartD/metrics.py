#!/usr/bin/env python3
"""
Simple metrics script that reads /tmp/controller_log.csv produced by Controller.py.
"""
import csv
import math
import sys

def read_log(path):
    rows = []
    with open(path, 'r') as f:
        r = csv.reader(f)
        header = next(r)
        for row in r:
            rows.append(row)
    return header, rows

if __name__ == '__main__':
    path = '/tmp/controller_log.csv'
    if len(sys.argv) > 1:
        path = sys.argv[1]
    header, rows = read_log(path)
    if not rows:
        print('No data in', path)
        sys.exit(1)
    sum_pos2 = 0.0
    sum_yaw2 = 0.0
    n = 0
    for row in rows:
        x_hat = float(row[1])
        y_hat = float(row[2])
        th_hat = float(row[3])
        x_ref = float(row[4])
        y_ref = float(row[5])
        th_ref = float(row[6])
        dx = x_ref - x_hat
        dy = y_ref - y_hat
        dth = math.atan2(math.sin(th_ref - th_hat), math.cos(th_ref - th_hat))
        sum_pos2 += dx * dx + dy * dy
        sum_yaw2 += dth * dth
        n += 1
    rmse_pos = math.sqrt(sum_pos2 / n)
    rmse_yaw = math.sqrt(sum_yaw2 / n)
    print(f"Samples: {n}")
    print(f"Position RMSE: {rmse_pos:.4f} m")
    print(f"Heading RMSE: {rmse_yaw:.4f} rad")
