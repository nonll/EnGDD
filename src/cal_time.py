def cal_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h>0:
        print(f'The program is running time: {h:.0f}h{m:.0f}m{s:.0f}s')
    elif m>0:
        print(f'The program is running time:{m:.0f}m{s:.0f}s')
    else:
        print(f'The program is running time: {s:.4f}s')
