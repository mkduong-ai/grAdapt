# https://en.wikipedia.org/wiki/Test_functions_for_optimization

import numpy as np


def ackley(x):
    x = np.array(x)
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))) - np.exp(
        0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))) + np.e + 20


def beale(x):
    x = np.array(x)
    return (1.5 - x[0] + x[0] * x[1]) ** 2 + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2 + (
            2.625 - x[0] + x[0] * x[1] ** 3) ** 2


def booth(x):
    y = x[1]
    x = x[0]
    return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2


def buking(x):
    y = x[1]
    x = x[0]
    return 100 * np.sqrt(np.abs(y - 0.01 * x ** 2)) + 0.01 * np.abs(x + 10)


def matyas(x):
    y = x[1]
    x = x[0]
    return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y


def levi(x):
    y = x[1]
    x = x[0]
    return np.sin(3 * np.pi * x) ** 2 + (x - 1) ** 2 * (1 + np.sin(3 * np.pi * y) ** 2) + (y - 1) ** 2 * (
            1 + np.sin(2 * np.pi * y) ** 2)


def himmelblau(x):
    y = x[1]
    x = x[0]
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


def three_hump_camel(x):
    y = x[1]
    x = x[0]
    return 2 * x ** 2 - 1.05 * x ** 4 + (x ** 6 / 6) + x * y + y ** 2


def easom(x):
    y = x[1]
    x = x[0]
    return -np.cos(x) * np.cos(y) * np.exp(-((x - np.pi) ** 2 + (y - np.pi) ** 2))


def cross_in_tray(x):
    y = x[1]
    x = x[0]
    return -0.0001 * (np.abs(np.sin(x) * np.sin(y) * np.exp(np.abs(100 - (np.sqrt(x ** 2 + y ** 2) / np.pi)))) + 1) ** (
        0.1)


def eggholder(x):
    y = x[1]
    x = x[0]
    return -(y + 47) * np.sin(np.sqrt(np.abs(x / 2 + (y + 47)))) - x * np.sin(np.sqrt(np.abs(x - (y + 47))))


def hoelder_table(x):
    y = x[1]
    x = x[0]
    return -np.abs(np.sin(x) * np.cos(y) * np.exp(np.abs(1 - (np.sqrt(x ** 2 + y ** 2) / np.pi))))


def mccormick(x):
    y = x[1]
    x = x[0]
    return np.sin(x + y) + (x - y) ** 2 - 1.5 * x + 2.5 * y + 1


def schaffern2(x):
    y = x[1]
    x = x[0]
    return 0.5 + (np.sin(x ** 2 - y ** 2) ** 2 - 0.5) / ((1 + 0.001 * (x ** 2 + y ** 2)) ** 2)


def schaffern4(x):
    y = x[1]
    x = x[0]
    return 0.5 + (np.cos(np.sin(np.abs(x ** 2 - y ** 2))) ** 2 - 0.5) / (1 + 0.001 * (x ** 2 + y ** 2)) ** 2


def goldstein_price(x):
    y = x[1]
    x = x[0]
    firstTerm = (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2))
    secondTerm = (30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2))
    return firstTerm * secondTerm


ackley_bounds = [(-5., 5.), (-5., 5.)]
ackley_ySol = 0.
ackley_xSol = np.array([[0., 0.]])

beale_bounds = [(-4.5, 4.5), (-4.5, 4.5)]
beale_ySol = 0.0
beale_xSol = np.array([[3., 0.5]])

booth_bounds = [(-10., 10.), (-10., 10.)]
booth_ySol = 0.
booth_xSol = np.array([[1., 3.]])

buking_bounds = [(-15., 5.), (-3., 3.)]
buking_ySol = 0.
buking_xSol = np.array([[-10., 1.]])

matyas_bounds = [(-10., 10.), (-10., 10.)]
matyas_ySol = 0.
matyas_xSol = np.array([[0., 0.]])

levi_bounds = [(-10., 10.), (-10., 10.)]
levi_ySol = 0.
levi_xSol = np.array([[1.0, 1.0]])

himmelblau_bounds = [(-5., 5.), (-5., 5.)]
himmelblau_ySol = 0.
himmelblau_xSol = np.array([[3., 2.], [-2.805118, 3.131312],
                            [-3.779310, -3.283186], [3.584428, -1.848126]])

three_hump_camel_bounds = [(-5., 5.), (-5., 5.)]
three_hump_camel_ySol = 0.
three_hump_camel_xSol = np.array([[0., 0.]])

easom_bounds = [(-100., 100.), (-100., 100.)]
easom_ySol = -1.
easom_xSol = np.array([[np.pi, np.pi]])

cross_in_tray_bounds = [(-10., 10.), (-10., 10.)]
cross_in_tray_ySol = -2.06261
cross_in_tray_xSol = np.array([[1.34941, -1.34941], [1.34941, 1.34941],
                               [-1.34941, 1.34941], [-1.34941, -1.34941]])

eggholder_bounds = [(-512., 512.), (-512., 512.)]
eggholder_ySol = -959.6407
eggholder_xSol = np.array([[512., 404.2319]])

hoelder_table_bounds = [(-10., 10.), (-10., 10.)]
hoelder_table_ySol = -19.2085
hoelder_table_xSol = np.array([[8.05502, 9.66459], [-8.05502, 9.66459],
                               [8.05502, -9.66459], [-8.05502, -9.66459]])

mccormick_bounds = [(-1.5, 4.), (-3., 4.)]
mccormick_ySol = -1.9133
mccormick_xSol = np.array([[-0.54719, -1.54719]])

schaffern2_bounds = [(-100., 100.), (-100., 100.)]
schaffern2_ySol = 0.
schaffern2_xSol = np.array([[0., 0.]])

schaffern4_bounds = [(-100., 100.), (-100., 100.)]
schaffern4_ySol = 0.292579
schaffern4_xSol = np.array([[0, 1.25313], [0, -1.25313]])

goldstein_price_bounds = [(-2., 2.), (-2., 2.)]
goldstein_price_ySol = 3.
goldstein_price_xSol = np.array([[0., -1.]])

all_functions_string = ['ackley', 'beale', 'booth', 'buking', 'matyas', 'levi', 'himmelblau', 'three_hump_camel',
                        'easom', 'cross_in_tray', 'eggholder', 'hoelder_table', 'mccormick', 'schaffern2', 'schaffern4',
                        'goldstein_price']

all_functions = [eval(function_string) for function_string in all_functions_string]

all_bounds = [eval(function_string + '_bounds') for function_string in all_functions_string]

all_xSol = [eval(function_string + '_xSol') for function_string in all_functions_string]

all_ySol = [eval(function_string + '_ySol') for function_string in all_functions_string]
