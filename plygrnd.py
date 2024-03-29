# Option A
import re

if __name__ == '__main__' and False:

    s = r'''
0 150 50 800

0 800 150 50

0 800 50 150

0 150 800 50

0 50 800 150

900 0 0 100

900 100 0 0

0 50 150 800

900 0 100 0
'''

    s = s.replace('\n\n', '\n')
    s = s.split('\n')
    s = [st.replace(' ', ' & ') + '\\\\' for st in s if st != '']
    for st in s:
        print(st)

# Option A
if __name__ == '__main__' and False:

    s = r'''
50_950_0

0_950_50

50_950_0

0_950_50
'''

    s = s.replace('\n\n', '\n')
    s = s.split('\n')
    s = [st.replace('_', ' & ') + '\\\\' for st in s if st != '']
    for st in s:
        print(st)

# Option B
if __name__ == '__main__' and False:
    r50 = r'''
[1]900_0_100_0	0.1667
[1]900_0_0_100	0.1667
[1]900_100_0_0	0.1667
[2]0_800_150_50	0.0833
[4]0_50_150_800	0.0833
[2]0_800_50_150	0.0833
[3]0_50_800_150	0.0833
[3]0_150_800_50	0.0833
[4]0_150_50_800	0.0833
'''

    r100 = r'''
[1]800_0_200_0	0.1667
[1]800_0_0_200	0.1667
[1]800_200_0_0	0.1667
[2]0_600_300_100	0.0833
[4]0_100_300_600	0.0833
[2]0_600_100_300	0.0833
[3]0_100_600_300	0.0833
[3]0_300_600_100	0.0833
[4]0_300_100_600	0.0833
'''

    r3p50 = r'''
[3]0_500_500	0.0833
[2]0_500_500	0.0833
[3]500_0_500	0.0833
[1]500_0_500	0.0833
[2]500_500_0	0.0833
[1]500_500_0	0.0833
[1]450_550_0	0.0833
[1]450_0_550	0.0833
[2]550_450_0	0.0833
[2]0_450_550	0.0833
[3]550_0_450	0.0833
[3]0_550_450	0.0833
'''

    r3p100 = r'''
    [1]500_500_0	0.0833
    [1]500_0_500	0.0833
    [2]500_500_0	0.0833
    [2]0_500_500	0.0833
    [3]500_0_500	0.0833
    [3]0_500_500	0.0833
    [3]600_0_400	0.0833
    [3]0_600_400	0.0833
    [2]600_400_0	0.0833
    [2]0_400_600	0.0833
    [1]400_600_0	0.0833
    [1]400_0_600	0.0833
    '''

    s = [rt for rt in r3p100.split('\n') if rt != '']
    lines = list()
    for sline in s:
        pattern_3 = r'\[([0-9]+)\]([0-9]+)_([0-9]+)_([0-9]+)[^0-9]+([0-9+]\.[0-9]+)'
        pattern_4 = r'\[([0-9]+)\]([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)[^0-9]+([0-9+]\.[0-9]+)'
        tokens = re.search(pattern_3, sline).groups()

        leader = int(tokens[0])
        p1 = int(tokens[1])
        p2 = int(tokens[2])
        p3 = int(tokens[3])
        # p4 = int(tokens[4])
        freq = float(tokens[-1])
        # l = f'{p1} & {p2} & {p3} & {p4} & {freq:>.3f}\\\\'
        # lines += [[l, leader, p1, p2, p3, p4, freq]]
        l = f'{p1} & {p2} & {p3} & {freq:>.3f}\\\\'
        lines += [[l, leader, p1, p2, p3, freq]]

    lines.sort(key=lambda lt: -lt[-1])
    for lt in lines:
        print(lt[0])

if __name__ == '__main__':
    r = '''
[1]459_541_0	0.0	0.1667
[1]490_510_0	0.0	0.1667
[2]0_459_541	0.0	0.1667
[2]0_490_510	0.0	0.1667
[3]510_0_490	0.0	0.1667
[3]541_0_459	0.0	0.1667
[1]459_0_541	0.1667	0.0
[1]490_0_510	0.1667	0.0
[2]510_490_0	0.1667	0.0
[2]541_459_0	0.1667	0.0
[3]0_510_490	0.1667	0.0
[3]0_541_459	0.1667	0.0
'''

    s = [rt for rt in r.split('\n') if rt != '']
    lines = list()
    for sline in s:
        pattern_3 = r'\[([0-9]+)\]([0-9]+)_([0-9]+)_([0-9]+)[^0-9]+([0-9+]\.[0-9]+)[^0-9]+([0-9+]\.[0-9]+)'
        # pattern_4 = r'\[([0-9]+)\]([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)[^0-9]+([0-9+]\.[0-9]+)[^0-9]+([0-9+]\.[0-9]+)'
        tokens = re.search(pattern_3, sline).groups()

        leader = int(tokens[0])
        p1 = int(tokens[1])
        p2 = int(tokens[2])
        p3 = int(tokens[3])
        # p4 = int(tokens[4])
        freq1 = float(tokens[-1])
        freq2 = float(tokens[-2])
        freq1_str = f'{freq1:>.3f}'
        if freq1 == 0:
            freq1_str = f'0'
        freq2_str = f'{freq2:>.3f}'
        if freq2 == 0:
            freq2_str = f'0'

        # l = f'{p1} & {p2} & {p3} & {p4} & {freq:>.3f}\\\\'
        # lines += [[l, leader, p1, p2, p3, p4, freq]]
        l = f'{p1} & {p2} & {p3} & {freq1_str} & {freq2_str}\\\\'
        lines += [[l, leader, p1, p2, p3, freq1, freq2]]

    for lt in lines:
        print(lt[0])
