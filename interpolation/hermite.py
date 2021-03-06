import numpy as np
from sympy import *
from sympy.parsing.sympy_parser import parse_expr
import math

def print_2d_array(tab):
    for i in range(0, len(tab[0])):
        print(tab[0][i], end="\t")
    print('')
    for r in tab[1:]:
        for c in r:
            print(c, end="\t")
        print()


def divided_differences(nodes, fun):
    tab = [[]]
    n = len(nodes)

    j = 0
    for node in nodes:
        tab.append([])
        tab[j].append(node)
        tab[j].append(fun[j])
        j += 1

    for j in range(1, n):
        for k in range(n, j, -1):
            result = (tab[k - 1][j] - tab[k - 2][j]) / (tab[k - 1][0] - tab[k - j - 1][0])
            if j == n-1:
                return result
            else:
                tab[k - 1].append(result)

    return 0


def hermite_interpolationg(nodes, fun, x, print_tab=false):
    tab = [[]]
    tab.insert(0,[])
    tab[0].append('x_i\t')
    tab[0].append('f(x_i)\t')
    i = 1
    for item in nodes:
        tab.insert(i, [])
        tab[i].append(item)
        tab[i].append(fun.subs(x, item))
        i += 1

    n = len(nodes)

    i = 1
    tmp_fun = fun
    polynomial = str(tab[1][1])
    polynomial_tmp = ''
    for _ in range(0, n-1):
        tab[0].append('R_'+str(i+1)+'(x_i)')

        fun_prime = tmp_fun.diff(x)
        tmp_fun = fun_prime

        for j in range(1, n+1):

            if j+1 <= i+1:
                tab[j].append('-')
            else:
                if tab[j][0] == tab[j - i][0]:
                    tab[j].append(fun_prime.subs(x, tab[j][0]) / math.factorial(i))
                else:
                    tab[j].append(divided_differences([tab[j][0], tab[j - i][0]], [tab[j][i], tab[j - 1][i]]))

                if j == i+1:
                    polynomial_tmp += '*(x-(' + str(tab[j-1][0]) + '))'
                    polynomial += '+' + str(tab[j][i+1]) + polynomial_tmp

        i += 1
    if print_tab:
        print_2d_array(tab)
    return simplify(polynomial)


def hermite_interpolationg_return_array(nodes, fun, x, print_tab=false):
    tab = [[]]
    tab.insert(0,[])
    tab[0].append('x_i\t')
    tab[0].append('f(x_i)\t')
    i = 1
    for item in nodes:
        tab.insert(i, [])
        tab[i].append(item)
        tab[i].append(fun.subs(x, item))
        i += 1

    n = len(nodes)

    i = 1
    tmp_fun = fun
    for _ in range(0, n-1):
        tab[0].append('R_'+str(i+1)+'(x_i)')

        fun_prime = tmp_fun.diff(x)
        tmp_fun = fun_prime

        for j in range(1, n+1):

            if j+1 <= i+1:
                tab[j].append('-')
            else:
                if tab[j][0] == tab[j - i][0]:
                    tab[j].append(fun_prime.subs(x, tab[j][0]) / math.factorial(i))
                else:
                    tab[j].append(divided_differences([tab[j][0], tab[j - i][0]], [tab[j][i], tab[j - 1][i]]))
        i += 1
    if print_tab:
        print_2d_array(tab)
    return tab[len(tab[0])-1]


def calculate_value_for_polynomial_in_array(n, array, nodes):
    if len(array) == 0 or len(nodes) == 0:
        return 0
    result = array[1]
    i = 0
    tmp = 1
    for coefficient in array[2:]:
        tmp *= (n - nodes[i])
        result += coefficient*tmp
        i += 1

    return result


if __name__ == '__main__':

    x = Symbol('x')
    fun = parse_expr('x**8+1')

    nodes = np.array([-1,-1,-1,0,0,0, 1,1,1])
    array = hermite_interpolationg_return_array(nodes, fun, x, True)
