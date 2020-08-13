import global_vars as gv

import mod 
from mod import func 

print(1)
print(gv.global_var) 

gv.global_var = 'main'
print(2)
print(gv.global_var)

print(3)
func() 

gv.global_var = 'init'
print(4)
func() 
