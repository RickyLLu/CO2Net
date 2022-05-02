import os
f = open('./final_task_filter.txt', 'r')
g = open('./tl_task.txt', 'w')
for line in f.readlines():
    real, mask, tran = line.strip().split()
    real = real.replace('\\','/').replace('JPEGImages', 'real_videos')
    mask = mask.replace('\\','/').replace('final_mask', 'foreground_mask')
    tran = tran.replace('\\','/').replace('trans_results', 'synthetic_composite_videos')
    g.write(real +' ' + mask + ' ' + tran + '\n')


