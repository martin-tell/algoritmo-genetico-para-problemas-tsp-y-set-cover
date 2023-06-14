from numpy import concatenate, array, argsort, empty, argmin, uint8, uint16, arange, where, unique, Infinity, array_equiv, zeros, ones
from numpy.random import randint, random, shuffle
from random import sample
import matplotlib.pyplot as plt

tamano_poblacion=1000
generaciones=100 
universo = arange(1,508) #números consecutivos del 1 al 507
nsubconjuntos = 63009

def obtener_conjuntos():
    conjuntos = []
    with open("rail507.txt") as archivo:
        conjuntos = [array(linea.strip().split(), dtype=uint16) for linea in archivo]
    return array(conjuntos, dtype=object)

def generar_poblacion_binaria():
    poblacion = empty((tamano_poblacion, 63009), dtype=uint8)
    for i in range(tamano_poblacion):
        n = randint(10000, 30000)
        m = 63009 - n
        ceros = zeros(n, dtype=uint8)
        unos = ones(m, dtype=uint8)
        p = array(concatenate((ceros, unos)))
        shuffle(p)
        poblacion[i] = p
    return poblacion

def set_cover(subconjuntos, cromosoma):
    indices = where(cromosoma)[0]
    union = concatenate(subconjuntos[indices])
    universo_candidato = unique(union)
    if array_equiv(universo_candidato, universo):
        return len(indices)
    else:
        return Infinity

def fitness(subconjuntos, poblacion):
    costos = empty((tamano_poblacion), "float")
    for i, cromosoma in enumerate(poblacion):
        costos[i] = set_cover(subconjuntos, cromosoma)
    return costos

def seleccion(poblacion, total_subconjuntos, tamano_torneo):
    participantes = randint(tamano_poblacion, size=tamano_torneo)
    return poblacion[participantes[argmin(total_subconjuntos[participantes])]]

def crossover(cromosoma1, cromosoma2):
    c = randint(1, 63007)
    cruce1 = concatenate((cromosoma1[:c], cromosoma2[c:]))
    cruce2 = concatenate((cromosoma2[:c], cromosoma1[c:]))
    return cruce1, cruce2

def mutar(cromosoma, tasa_mutacion):
    if random() < tasa_mutacion:
      i = sample(range(1, nsubconjuntos), 2)
      i1 = min(i)
      i2 = max(i)
      cromosoma[i1:i2] = cromosoma[i2-1:i1-1:-1]
      i = sample(range(0, nsubconjuntos), 5000)
      cromosoma[i] = 0
    return cromosoma

def ga_set_cover(proceso=True):
    iteraciones = []
    solucion = None
    subconjuntos = obtener_conjuntos()
    p = generar_poblacion_binaria()
    f = fitness(subconjuntos, p)
    for i in range(generaciones):
        padres = []
        for k in range(0, tamano_poblacion, 2):
            s1 = seleccion(p, f, 2)
            s2 = seleccion(p, f, 2)
            padres.append(s1)
            padres.append(s2)
        cruzes = []
        for k in range(0, tamano_poblacion, 2):
            cr1, cr2 = crossover(array(padres[k]), array(padres[k+1]))
            cruzes.append(cr1)
            cruzes.append(cr2)
        mutaciones = [mutar(array(cruzes[l]), 0.3) for l in range(tamano_poblacion)]
        nf = fitness(subconjuntos, mutaciones)
        f = concatenate((f, nf))
        p = concatenate((p, mutaciones))
        indices = argsort(f)
        f = f[indices][:-tamano_poblacion]
        p = p[indices][:-tamano_poblacion]
        indice = argmin(f)
        if proceso:
            print(f"{i}. variables: {p[indice]} evaluacion: {f[indice]}")
        iteraciones.append([i, f[indice]])
        solucion = p[indice]
    return solucion, iteraciones

if __name__ == "__main__":
  resultados = []
  ejecuciones = []
  for i in range(5):
    x, y = ga_set_cover(proceso=True)
    y = list(zip(*y))
    resultados.append(y[1])
    fig, ax = plt.subplots()
    plt.title(f'Algoritmo Genético para Set Cover prueba {i+1}')
    plt.xlabel('generaciones')
    plt.ylabel('cantidad de subconjuntos necesarios')
    ax.text(0.5, -0.4, f"Cantidad mínima de subconjuntos a usar: {y[1][len(y[1])-1]}", transform=ax.transAxes, ha='center')
    ax.plot(y[0], y[1])
    plt.savefig(f'grafica{i+1}.jpg', bbox_inches='tight', dpi=300)
    plt.close()