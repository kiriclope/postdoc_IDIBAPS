import global_vars as gv

def func():
    if(gv.global_var == 'init'):
        print('init')
    else:
        print('main')
