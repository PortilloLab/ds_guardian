Python 3.10.12 (main, Feb  4 2025, 14:57:36) [GCC 11.4.0] on linux
Type "help", "copyright", "credits" or "license()" for more information.
>>> def evaluar_nota(nota):
...     if nota >= 7:
...         return "Aprobado"
...     else return ("Reprobado")
...     
SyntaxError: expected ':'
>>> 
>>> def evaluar_nota (nota):
...     if nota >= 7:
...         return ("Aprobado")
...     else return ("Desaprobado")
...     
SyntaxError: expected ':'
>>> def evaluar_nota (nota):
...     if nota>=7:
...         return "aprobado"
...     else return desaprobado
...     
SyntaxError: expected ':'
>>> SyntaxError: expected ':'
SyntaxError: invalid syntax
>>> def evaluar_nota (nota):
...     if nota >= 7:
...         return "Aprobado"
...     else
...     
SyntaxError: incomplete input
>>> SyntaxError: incomplete input
SyntaxError: incomplete input
>>> def evaluar_nota (nota):
...     if nota >= 7:
...         return "Aprobado"
...     else:
...         return "Desaprobado"
...     print (evaluar_nota(8)
