import pandas as pd
data = {
	'Nombre': ['Ana', 'Luis', 'Carlos'],
	'Edad': [34, 28, 45],
	'Salario': [55000, 45000, 700000],
}
empleados = pd.DataFrame(data)
print("DataFrame de empleados:\n", empleados)
empleados_bien_pagados = empleados[empleados['Salario'] > 50000]
print("\nEmpleados con salario mayor a 50000:\n", empleados_bien_pagados)