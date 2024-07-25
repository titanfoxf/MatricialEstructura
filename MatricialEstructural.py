# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 02:06:47 2024

@author: titan
"""

import matplotlib.pyplot as plt
import numpy as np

class Vector:
    def __init__(self, x, y, z=None):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        if self.z is not None:
            return f"Vector({self.x}, {self.y}, {self.z})"
        else:
            return f"Vector({self.x}, {self.y})"

class Nodo:
    def __init__(self, nombre, ubicacion, GDL_x=True, GDL_y=True, GDL_giro=True, carga=None, desplazamiento=None):
        self.nombre = nombre
        self.ubicacion = ubicacion
        self.GDL_x = GDL_x
        self.GDL_y = GDL_y
        self.GDL_giro = GDL_giro
        self.GDL_nombres = {"x": None, "y": None, "giro": None}
        self.carga = carga if carga else Vector(0.0, 0.0, 0.0)
        self.desplazamiento = desplazamiento if desplazamiento else Vector(0.0, 0.0, 0.0)

    def asignar_nombres_gdl(self, nombre_x, nombre_y, nombre_giro):
        self.GDL_nombres["x"] = nombre_x
        self.GDL_nombres["y"] = nombre_y
        self.GDL_nombres["giro"] = nombre_giro

    def modificar_carga(self, x, y, momento):
        self.carga = Vector(x, y, momento)

    def modificar_desplazamiento(self, dx, dy, dtheta):
        if self.GDL_x:
            self.desplazamiento.x = dx
        if self.GDL_y:
            self.desplazamiento.y = dy
        if self.GDL_giro:
            self.desplazamiento.z = dtheta

    def graficar_nombre(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        
        ax.plot(self.ubicacion.x, self.ubicacion.y, 'ko')  # Punto del nodo
        ax.text(self.ubicacion.x, self.ubicacion.y, f' {self.nombre}', verticalalignment='bottom')

        ax.set_aspect('equal')
        if ax is None:
            plt.show()

    def graficar_grados_de_libertad(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        
        if self.GDL_x:
            ax.arrow(self.ubicacion.x, self.ubicacion.y, 1, 0, head_width=0.2, head_length=0.2, fc='r', ec='r')
            if self.GDL_nombres["x"] is not None:
                ax.text(self.ubicacion.x + 0.5, self.ubicacion.y, f'{self.GDL_nombres["x"]}', color='r')

        if self.GDL_y:
            ax.arrow(self.ubicacion.x, self.ubicacion.y, 0, 1, head_width=0.2, head_length=0.2, fc='g', ec='g')
            if self.GDL_nombres["y"] is not None:
                ax.text(self.ubicacion.x, self.ubicacion.y + 0.5, f'{self.GDL_nombres["y"]}', color='g')

        if self.GDL_giro:
            theta = np.linspace(0, 2 * np.pi, 100)
            r = 0.5
            x = r * np.cos(theta) + self.ubicacion.x
            y = r * np.sin(theta) + self.ubicacion.y
            ax.plot(x, y, 'b')
            if self.GDL_nombres["giro"] is not None:
                ax.text(self.ubicacion.x + 0.5, self.ubicacion.y + 0.5, f'{self.GDL_nombres["giro"]}', color='b')

        ax.set_aspect('equal')
        if ax is None:
            plt.show()

    def graficar_cargas(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        
        if self.carga.x != 0.0:
            ax.arrow(self.ubicacion.x, self.ubicacion.y, self.carga.x, 0, head_width=0.2, head_length=0.2, fc='m', ec='m')
            ax.text(self.ubicacion.x + self.carga.x / 2, self.ubicacion.y, f'{self.carga.x:.1f} N', color='m')

        if self.carga.y != 0.0:
            ax.arrow(self.ubicacion.x, self.ubicacion.y, 0, self.carga.y, head_width=0.2, head_length=0.2, fc='m', ec='m')
            ax.text(self.ubicacion.x, self.ubicacion.y + self.carga.y / 2, f'{self.carga.y:.1f} N', color='m')

        if self.carga.z != 0.0:
            theta = np.linspace(0, 2 * np.pi, 100)
            r = 0.5 * np.sign(self.carga.z)
            x = r * np.cos(theta) + self.ubicacion.x
            y = r * np.sin(theta) + self.ubicacion.y
            ax.plot(x, y, 'm')
            ax.text(self.ubicacion.x + 0.5, self.ubicacion.y + 0.5 * np.sign(self.carga.z), f'{self.carga.z:.1f} kNm', color='m')

        ax.set_aspect('equal')
        if ax is None:
            plt.show()

    def graficar_desplazamientos(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        
        if self.desplazamiento.x != 0.0:
            ax.arrow(self.ubicacion.x, self.ubicacion.y, self.desplazamiento.x, 0, head_width=0.2, head_length=0.2, fc='b', ec='b')
            ax.text(self.ubicacion.x + self.desplazamiento.x / 2, self.ubicacion.y, f'{self.desplazamiento.x:.1f} mm', color='b')

        if self.desplazamiento.y != 0.0:
            ax.arrow(self.ubicacion.x, self.ubicacion.y, 0, self.desplazamiento.y, head_width=0.2, head_length=0.2, fc='b', ec='b')
            ax.text(self.ubicacion.x, self.ubicacion.y + self.desplazamiento.y / 2, f'{self.desplazamiento.y:.1f} mm', color='b')

        if self.desplazamiento.z != 0.0:
            theta = np.linspace(0, 2 * np.pi, 100)
            r = 0.5 * np.sign(self.desplazamiento.z)
            x = r * np.cos(theta) + self.ubicacion.x
            y = r * np.sin(theta) + self.ubicacion.y
            ax.plot(x, y, 'b')
            ax.text(self.ubicacion.x + 0.5, self.ubicacion.y + 0.5 * np.sign(self.desplazamiento.z), f'{self.desplazamiento.z:.1f} rad', color='b')

        ax.set_aspect('equal')
        if ax is None:
            plt.show()

class Barra:
    def __init__(self, nombre, nodo_inicio, nodo_fin, EA, EI):
        self.nombre = nombre
        self.nodo_inicio = nodo_inicio
        self.nodo_fin = nodo_fin
        self.EA = EA
        self.EI = EI
        self.L = self.calcular_longitud()
        self.angulo = self.calcular_angulo()

    def calcular_longitud(self):
        dx = self.nodo_fin.ubicacion.x - self.nodo_inicio.ubicacion.x
        dy = self.nodo_fin.ubicacion.y - self.nodo_inicio.ubicacion.y
        return np.sqrt(dx**2 + dy**2)

    def calcular_angulo(self):
        dx = self.nodo_fin.ubicacion.x - self.nodo_inicio.ubicacion.x
        dy = self.nodo_fin.ubicacion.y - self.nodo_inicio.ubicacion.y
        return np.arctan2(dy, dx)

    def matriz_de_rigidez_local(self):
        L = self.L
        EA = self.EA
        EI = self.EI

        k = np.array([
            [ EA/L,       0,         0, -EA/L,       0,         0],
            [      0,  12*EI/L**3,  6*EI/L**2,      0, -12*EI/L**3,  6*EI/L**2],
            [      0,   6*EI/L**2,  4*EI/L,         0,  -6*EI/L**2,  2*EI/L],
            [-EA/L,       0,         0,  EA/L,       0,         0],
            [      0, -12*EI/L**3, -6*EI/L**2,      0,  12*EI/L**3, -6*EI/L**2],
            [      0,   6*EI/L**2,  2*EI/L,         0,  -6*EI/L**2,  4*EI/L]
        ])
        
        return k

    def matriz_de_transformacion(self):
        c = np.cos(self.angulo)
        s = np.sin(self.angulo)
        T = np.array([
            [ c, s, 0,  0, 0, 0],
            [-s, c, 0,  0, 0, 0],
            [ 0, 0, 1,  0, 0, 0],
            [ 0, 0, 0,  c, s, 0],
            [ 0, 0, 0, -s, c, 0],
            [ 0, 0, 0,  0, 0, 1]
        ])
        return T

    def matriz_de_rigidez_global(self):
        k_local = self.matriz_de_rigidez_local()
        T = self.matriz_de_transformacion()
        k_global = T.T @ k_local @ T
        return np.round(k_global, 2)

    def nombres_gdl(self):
        return [
            self.nodo_inicio.GDL_nombres["x"], self.nodo_inicio.GDL_nombres["y"], self.nodo_inicio.GDL_nombres["giro"],
            self.nodo_fin.GDL_nombres["x"], self.nodo_fin.GDL_nombres["y"], self.nodo_fin.GDL_nombres["giro"]
        ]

    def graficar(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        
        # Graficar los nodos
        self.nodo_inicio.graficar_nombre(ax)
        self.nodo_fin.graficar_nombre(ax)
        
        # Graficar la barra
        x_values = [self.nodo_inicio.ubicacion.x, self.nodo_fin.ubicacion.x]
        y_values = [self.nodo_inicio.ubicacion.y, self.nodo_fin.ubicacion.y]
        ax.plot(x_values, y_values, 'k-')

        # Colocar el nombre de la barra alineado a la misma
        mid_x = (self.nodo_inicio.ubicacion.x + self.nodo_fin.ubicacion.x) / 2
        mid_y = (self.nodo_inicio.ubicacion.y + self.nodo_fin.ubicacion.y) / 2
        ax.text(mid_x, mid_y, f' {self.nombre}', verticalalignment='bottom', horizontalalignment='right')

        if ax is None:
            plt.show()

class Estructura:
    def __init__(self):
        self.nodos = []
        self.barras = []
        self.matriz_cargas = None
        self.matriz_desplazamientos = None

    def agregar_nodo(self, nombre, ubicacion, GDL_x=True, GDL_y=True, GDL_giro=True, carga=None):
        nodo = Nodo(nombre, ubicacion, GDL_x, GDL_y, GDL_giro, carga)
        self.nodos.append(nodo)

    def eliminar_nodo(self, nombre):
        self.nodos = [nodo for nodo in self.nodos if nodo.nombre != nombre]

    def redefinir_nodo(self, nombre, nueva_ubicacion=None, GDL_x=None, GDL_y=None, GDL_giro=None, carga=None):
        for nodo in self.nodos:
            if nodo.nombre == nombre:
                if nueva_ubicacion is not None:
                    nodo.ubicacion = nueva_ubicacion
                if GDL_x is not None:
                    nodo.GDL_x = GDL_x
                if GDL_y is not None:
                    nodo.GDL_y = GDL_y
                if GDL_giro is not None:
                    nodo.GDL_giro = GDL_giro
                if carga is not None:
                    nodo.carga = carga

    def agregar_barra(self, nombre, nodo_inicio, nodo_fin, EA, EI):
        nodo_inicio_obj = next(nodo for nodo in self.nodos if nodo.nombre == nodo_inicio)
        nodo_fin_obj = next(nodo for nodo in self.nodos if nodo.nombre == nodo_fin)
        barra = Barra(nombre, nodo_inicio_obj, nodo_fin_obj, EA, EI)
        self.barras.append(barra)

    def eliminar_barra(self, nombre):
        self.barras = [barra for barra in self.barras if barra.nombre != nombre]

    def redefinir_barra(self, nombre, nodo_inicio=None, nodo_fin=None, EA=None, EI=None):
        for barra in self.barras:
            if barra.nombre == nombre:
                if nodo_inicio is not None:
                    barra.nodo_inicio = next(nodo for nodo in self.nodos if nodo.nombre == nodo_inicio)
                if nodo_fin is not None:
                    barra.nodo_fin = next(nodo for nodo in self.nodos if nodo.nombre == nodo_fin)
                if EA is not None:
                    barra.EA = EA
                if EI is not None:
                    barra.EI = EI
                barra.L = barra.calcular_longitud()
                barra.angulo = barra.calcular_angulo()

    def numerar_gdl(self):
        contador = 1
        for nodo in self.nodos:
            nodo.asignar_nombres_gdl(f'GDL{contador}', f'GDL{contador+1}', f'GDL{contador+2}')
            contador += 3

    def crear_matrices_cargas_desplazamientos(self):
        num_gdl = len(self.nodos) * 3
        self.matriz_cargas = np.empty((num_gdl, 1), dtype=object)
        self.matriz_desplazamientos = np.empty((num_gdl, 1), dtype=object)
        
        contador = 0
        for nodo in self.nodos:
            if nodo.GDL_x:
                self.matriz_cargas[contador, 0] = nodo.carga.x
                self.matriz_desplazamientos[contador, 0] = 'DD'
            else:
                self.matriz_cargas[contador, 0] = 'CC'
                self.matriz_desplazamientos[contador, 0] = 0
            contador += 1
            
            if nodo.GDL_y:
                self.matriz_cargas[contador, 0] = nodo.carga.y
                self.matriz_desplazamientos[contador, 0] = 'DD'
            else:
                self.matriz_cargas[contador, 0] = 'CC'
                self.matriz_desplazamientos[contador, 0] = 0
            contador += 1
            
            if nodo.GDL_giro:
                self.matriz_cargas[contador, 0] = nodo.carga.z
                self.matriz_desplazamientos[contador, 0] = 'DD'
            else:
                self.matriz_cargas[contador, 0] = 'CC'
                self.matriz_desplazamientos[contador, 0] = 0
            contador += 1

    def graficar(self, mostrar_nombres=True, mostrar_grados_de_libertad=True, mostrar_cargas=True, mostrar_desplazamientos=True):
        fig, ax = plt.subplots()
        for barra in self.barras:
            barra.graficar(ax)
        for nodo in self.nodos:
            if mostrar_nombres:
                nodo.graficar_nombre(ax)
            if mostrar_grados_de_libertad:
                nodo.graficar_grados_de_libertad(ax)
            if mostrar_cargas:
                nodo.graficar_cargas(ax)
            if mostrar_desplazamientos:
                nodo.graficar_desplazamientos(ax)
        plt.show()

    def mostrar_matrices_de_rigidez_global(self):
        np.set_printoptions(precision=2, suppress=True, formatter={'float': '{: 0.2e}'.format})
        for barra in self.barras:
            k_global = barra.matriz_de_rigidez_global()
            nombres_gdl = barra.nombres_gdl()
            print(f"Matriz de rigidez global de {barra.nombre}:")
            print(" " * 10 + " ".join(f"{nombre:>10}" for nombre in nombres_gdl))
            for nombre, fila in zip(nombres_gdl, k_global):
                print(f"{nombre:>10} " + " ".join(f"{valor:>10.2e}" for valor in fila))
            print("\n")

    def matriz_rigidez_absoluta(self):
        self.numerar_gdl()  # Asegurarse de que los GDL están numerados
        num_gdl = len(self.nodos) * 3
        matriz_absoluta = np.zeros((num_gdl, num_gdl))

        gdl_indices = {}
        contador = 0
        for nodo in self.nodos:
            for tipo in ["x", "y", "giro"]:
                gdl_indices[nodo.GDL_nombres[tipo]] = contador
                contador += 1

        for barra in self.barras:
            k_global = barra.matriz_de_rigidez_global()
            nombres_gdl = barra.nombres_gdl()
            indices = [gdl_indices[nombre] for nombre in nombres_gdl]

            for i, idx_i in enumerate(indices):
                for j, idx_j in enumerate(indices):
                    matriz_absoluta[idx_i, idx_j] += k_global[i, j]

        print("Matriz de rigidez absoluta de la estructura:")
        nombres_gdl_total = [nodo.GDL_nombres[gdl] for nodo in self.nodos for gdl in ["x", "y", "giro"]]
        print(" " * 10 + " ".join(f"{nombre:>10}" for nombre in nombres_gdl_total))
        for nombre, fila in zip(nombres_gdl_total, matriz_absoluta):
            print(f"{nombre:>10} " + " ".join(f"{valor:>10.2e}" for valor in fila))
        print("\n")
        
        return matriz_absoluta
        
    def matriz_rigidez_reducida(self):
        if self.matriz_desplazamientos is None:
            self.crear_matrices_cargas_desplazamientos()
        
        # Encontrar los índices de los grados de libertad que tienen 'DD' en la matriz de desplazamientos
        indices_dd = [i for i, val in enumerate(self.matriz_desplazamientos) if val == 'DD']
        
        # Crear la matriz de rigidez reducida
        matriz_reducida = self.matriz_rigidez_absoluta()[np.ix_(indices_dd, indices_dd)]
        
        print("Matriz de rigidez reducida:")
        print(matriz_reducida)
        return matriz_reducida
    
    
    # def resolver_desplazamientos(self):
    #     if self.matriz_desplazamientos is None:
    #         self.crear_matrices_cargas_desplazamientos()
        
    #     # Crear vector reducido de cargas
    #     vector_reducido_cargas = []
    #     for carga in self.matriz_cargas:
    #         if carga[0] != 'CC':
    #             vector_reducido_cargas.append(carga[0])
        
    #     vector_reducido_cargas = np.array(vector_reducido_cargas, dtype=float).reshape(-1, 1)
        
    #     # Crear matriz de rigidez reducida
    #     matriz_rigidez_reducida = self.matriz_rigidez_reducida()
        
    #     # Calcular los desplazamientos reducidos
    #     desplazamientos_reducidos = np.linalg.inv(matriz_rigidez_reducida) @ vector_reducido_cargas
        
    #     print("Desplazamientos reducidos:")
    #     print(desplazamientos_reducidos)
        
    #     return desplazamientos_reducidos

    def resolver_desplazamientos(self):
        if self.matriz_desplazamientos is None:
            self.crear_matrices_cargas_desplazamientos()
        
        # Crear vector reducido de cargas
        vector_reducido_cargas = []
        indices_dd = []
        for i, carga in enumerate(self.matriz_cargas):
            if carga[0] != 'CC':
                vector_reducido_cargas.append(carga[0])
                indices_dd.append(i)
        
        vector_reducido_cargas = np.array(vector_reducido_cargas, dtype=float).reshape(-1, 1)
        
        # Crear matriz de rigidez reducida
        matriz_rigidez_reducida = self.matriz_rigidez_reducida()
        
        # Calcular los desplazamientos reducidos
        desplazamientos_reducidos = np.linalg.inv(matriz_rigidez_reducida) @ vector_reducido_cargas
        
        print("Desplazamientos reducidos:")
        print(desplazamientos_reducidos)
        
        # Agregar desplazamientos reducidos a la matriz columna de desplazamientos global
        contador_reducido = 0
        for i in indices_dd:
            self.matriz_desplazamientos[i, 0] = desplazamientos_reducidos[contador_reducido, 0]
            contador_reducido += 1
        
        print("\nMatriz de Desplazamientos Global:")
        print(self.matriz_desplazamientos)
        
        return self.matriz_desplazamientos

    # def calcular_cargas_globales(self):
    #     if self.matriz_desplazamientos is None:
    #         raise ValueError("La matriz de desplazamientos no está definida. Primero debe resolver los desplazamientos.")
        
    #     matriz_rigidez_absoluta = self.matriz_rigidez_absoluta()
    #     matriz_cargas_globales = matriz_rigidez_absoluta @ self.matriz_desplazamientos
        
    #     print("Matriz de Cargas Globales:")
    #     print(matriz_cargas_globales)
        
    #     return matriz_cargas_globales
    
    def calcular_cargas_globales(self):
        if self.matriz_desplazamientos is None:
            raise ValueError("La matriz de desplazamientos no está definida. Primero debe resolver los desplazamientos.")
        
        matriz_rigidez_absoluta = self.matriz_rigidez_absoluta()
        matriz_cargas_globales = matriz_rigidez_absoluta @ self.matriz_desplazamientos
        # matriz_cargas_globales = np.round(matriz_cargas_globales, 2)
        
        print("Matriz de Cargas Globales:")
        print(matriz_cargas_globales)
        
        # Actualizar las cargas de los nodos usando la función modificar_carga
        contador = 0
        for nodo in self.nodos:
            carga_x = matriz_cargas_globales[contador, 0] if nodo.GDL_x else 0
            carga_y = matriz_cargas_globales[contador + 1, 0] if nodo.GDL_y else 0
            carga_giro = matriz_cargas_globales[contador + 2, 0] if nodo.GDL_giro else 0
            nodo.modificar_carga(carga_x, carga_y, carga_giro)
            contador += 3
        
        return matriz_cargas_globales


# Ejemplo de uso
estructura = Estructura()
estructura.agregar_nodo("1", Vector(0, 4), GDL_x=True, GDL_y=True, GDL_giro=True)
estructura.agregar_nodo("2", Vector(5, 4), GDL_x=False, GDL_y=True, GDL_giro=True)
estructura.agregar_nodo("3", Vector(0, 0), GDL_x=False, GDL_y=False, GDL_giro=False)
estructura.agregar_nodo("4", Vector(5, 0), GDL_x=False, GDL_y=False, GDL_giro=False)

estructura.agregar_barra("Barra1", "1", "3", EA=5e9, EI=1e8)
estructura.agregar_barra("Barra2", "1", "2", EA=5e9, EI=1e8)
estructura.agregar_barra("Barra3", "2", "4", EA=5e9, EI=1e8)

# Modificar las cargas en los nodos 1 y 2
estructura.nodos[0].modificar_carga(0.0, 0.0, 10e3)  # Nodo 1: 10 kNm
estructura.nodos[1].modificar_carga(0.0, 0.0, 20e3)  # Nodo 2: 20 kNm

# Modificar los desplazamientos en los nodos
estructura.nodos[0].modificar_desplazamiento(0, 0, 0.0)  # Nodo 1: Desplazamientos
estructura.nodos[1].modificar_desplazamiento(0, 0, 0.0)  # Nodo 2: Desplazamientos

# Numerar y graficar grados de libertad, cargas y desplazamientos
estructura.numerar_gdl()
# estructura.graficar(mostrar_nombres=True, mostrar_grados_de_libertad=False, mostrar_cargas=True, mostrar_desplazamientos=False)

# Crear y mostrar matrices de cargas y desplazamientos
estructura.crear_matrices_cargas_desplazamientos()
print("Matriz de Cargas:")
print(estructura.matriz_cargas)
print("\nMatriz de Desplazamientos:")
print(estructura.matriz_desplazamientos)

# Mostrar las matrices de rigidez globales de cada barra
estructura.mostrar_matrices_de_rigidez_global()

# Crear y mostrar la matriz de rigidez absoluta
estructura.matriz_rigidez_absoluta()

# Crear y mostrar la matriz de rigidez reducida
estructura.matriz_rigidez_reducida()

# Resolver y mostrar los desplazamientos desconocidos
estructura.resolver_desplazamientos()

# Calcular y mostrar las cargas globales
estructura.calcular_cargas_globales()

estructura.graficar(mostrar_nombres=True, mostrar_grados_de_libertad=True, mostrar_cargas=False, mostrar_desplazamientos=False)