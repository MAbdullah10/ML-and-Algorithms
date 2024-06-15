'''x = 3
print(type(x))
print(x+1)
print(x/3)
print(x**3)

y =4
x*=y
print("updated value of x is:",x)

y = "ABC"
print("type of y is:",type(y))

t = True
f = False
print(t and f)
print(t or f)
print(t!= f)
print(t and t)

a= 'hello'
b= 'world'
print(len(a))

hw = a + ' ' + b
print(hw)

c = 5
si = a + c
print(si)

a= 'hello'
b= 'world'
hwd = '%s %s %d' % (a, b, 12)
print(hwd)
print('%s %s %d' % (a, b, 12))

print(a.capitalize())
print(a.rjust(25))
print(b.replace ('l','(ell)'))

'''
'''
xs = [1,2,3]
print(xs,xs[2],xs[0])
print(xs[-1])
print(xs[-2])
xs[2] = 'foo'
xs.append('bar')
print(xs)
x = xs.pop()
print(x,xs)
xs.pop()
print(xs)
xs.append('a')
xs.append('2')
del xs[1:3]
print(xs)
l =list(range(100,70,-7))
print(l)


l1 = list(range(3,31,3))
print(l1)


v = 20
c =59
l2 = list(range(v,c,10))

b = list(range (10))
print(b)
print(b[:4])
print(b[7:])
print(b[-1:])
print(b[-2:-1])

evens = list(range (100))
for even in evens :
      if even %2 == 0:
           print(even)

a = 2+3*5
a= (2+3)*6
a = 4587238948923*347843723
a = 2  +     2
b = (5-1)*((7+1)/(3-1))
print(a)
print(b)

'''# Text
x = "NANCY"
print(x)

# combine text and number
s = "My lucky number is %d, what is yours?" % 7
print(s)

# second way
y = "My lucky number is " + str(7) + ", what is yours?"  # Corrected the concatenation
print(y)  # Removed the "ERROR?" at the end
