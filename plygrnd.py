# Option A
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
if __name__ == '__main__':

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