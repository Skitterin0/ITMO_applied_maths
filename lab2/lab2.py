from math import exp, sin, log, copysign


def func(x):
    return exp(sin(x) * log(x))


#  все методы возващают пару (x,y), соответствующие минимальному значению функции
def dichotomy_method(f: callable, a: float, b: float, epsilon=1e-6) -> (float, float):
    n = 0
    tol = epsilon/2
    intervals = []
    while abs(b - a) > epsilon:
        n += 1
        intervals.append((a, b))
        c = (a + b)/2
        if f(c - tol) < f(c + tol):
            b = c
        else:
            a = c
    # print(f"intervals on every iteration: {intervals}")
    print(f"required number of iterations for epsilon = {epsilon}: {n}")
    return (a + b)/2, f((a + b)/2)


def golden_section_method(f: callable, a: float, b: float, epsilon=1e-6) -> (float, float):
    phi = (3 - 5**0.5)/2
    x1 = a + (b - a) * phi
    x2 = b - (b - a) * phi
    n = 0
    intervals = []
    f1, f2 = f(x1), f(x2)
    while abs(b - a) > epsilon:
        n += 1
        intervals.append((a, b))
        if f1 < f2:
            b = x2
            x2 = x1
            x1 = a + (b - a) * phi
            f1, f2 = f(x1), f1
        else:
            a = x1
            x1 = x2
            x2 = b - (b - a) * phi
            f1, f2 = f2, f(x2)
    # print(f"intervals on every iteration: {intervals}")
    print(f"required number of iterations for epsilon = {epsilon}: {n}")
    return (a + b)/2, f((a + b)/2)


def fibonacci_method(f: callable, a: float, b: float, n: int) -> (float, float):
    fib = [1, 1]
    for i in range(2, n + 1):
        fib.append(fib[i - 1] + fib[i - 2])

    intervals = []

    k = 1
    x1 = a + (b - a) * fib[n - k - 1] / fib[n - k + 1]
    x2 = a + (b - a) * fib[n - k] / fib[n - k + 1]
    f1, f2 = f(x1), f(x2)

    while k < n - 1:
        intervals.append((a, b))
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            k += 1
            x1 = a + (b - a) * fib[n - k - 1] / fib[n - k + 1]
            f1 = f(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            k += 1
            x2 = a + (b - a) * fib[n - k] / fib[n - k + 1]
            f2 = f(x2)

    # print(f"intervals on every iteration: {intervals}")
    if f1 < f2:
        return x1, f(x1)
    else:
        return x2, f(x2)


def parabolic_interpolation(f: callable, x1: float, x3: float, epsilon=1e-6) -> (float, float):
    x2 = (x1 + x3)/2
    f1, f2, f3 = f(x1), f(x2), f(x3)
    n = 0
    intervals = []
    while abs(x3 - x1) > epsilon:
        n += 1
        intervals.append((x1, x3))
        u = x2 - ((x2 - x1) ** 2 * (f2 - f3) - (x2 - x3) ** 2 * (f2 - f1)) \
            / (2 * ((x2 - x1) * (f2 - f3) - (x2 - x3) * (f2 - f1)))
        fu = f(u)
        if x2 <= u:
            if f2 <= fu:
                x1, x2, x3 = x1, x2, u
                f1, f2, f3 = f1, f2, fu
            else:
                x1, x2, x3 = x2, u, x3
                f1, f2, f3 = f2, fu, f3
        else:
            if fu <= f2:
                x1, x2, x3 = x1, u, x2
                f1, f2, f3 = f1, fu, f2
            else:
                x1, x2, x3 = u, x2, x3
                f1, f2, f3 = fu, f2, f3

    # print(f"intervals on every iteration: {intervals}")
    print(f"required number of iterations for epsilon = {epsilon}: {n}")
    return (x1 + x3)/2, f((x1 + x3)/2)


def brent_method(f: callable, a: float, c: float, epsilon=1e-6) -> (float, float):
    phi = (3 - 5 ** 0.5)/2
    x = w = v = a + (c - a) * phi
    fx = fw = fv = f(x)
    d = e = c - a
    n = 0
    intervals = []
    while abs(d) > epsilon:
        n += 1
        intervals.append((a, c))
        g, e = e, d
        if x != w and x != v and w != v and fx != fw and fx != fv and fv != fw:
            # здесь параболическая аппроксимация
            if v < w:
                x1, x3 = v, w
                f1, f3 = fv, fw
            else:
                x1, x3 = w, v
                f1, f3 = fw, fv
            u = x - ((x - x1) ** 2 * (fx - f3) - (x - x3) ** 2 * (fx - f1)) \
                / (2 * ((x - x1) * (fx - f3) - (x - x3) * (fx - f1)))
            if a + epsilon < u < c - epsilon and abs(u - x) < g/2:
                true_u = u
                d = abs(true_u - x)
        else:
            if x < (c - a)/2:
                true_u = x + (c - x) * phi
                d = c - x
            else:
                true_u = x - (x - a) * phi
                d = x - a
        if abs(true_u - x) < epsilon:
            true_u = x + copysign(epsilon, true_u - x)
        fu = f(true_u)
        if fu <= fx:
            if true_u >= x:
                a = x
            else:
                c = x
            v, w, x = w, x, true_u
            fv, fw, fx = fw, fx, fu
        else:
            if true_u >= x:
                c = true_u
            else:
                a = true_u
            if fu <= fw or w == x:
                v, w = w, true_u
                fv, fw = fw, fu
            elif fu <= fv or v == x or v == w:
                v, fv = true_u, fu

    # print(f"intervals on every iteration: {intervals}")
    print(f"required number of iterations for epsilon = {epsilon}: {n}")
    return x, f(x)


x1, x2 = 2.2, 7.8
accuracies = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
# accuracies = [1e-6]
for accuracy in accuracies:
    print(f"minimal value found by dichotomy_method: {dichotomy_method(func, x1, x2, accuracy)}\n")
    print(f"minimal value found by golden_section_method: {golden_section_method(func, x1, x2, accuracy)}\n")
    n = 0
    fib1 = 1
    fib2 = 1
    while True:
        n += 1
        a = (x2-x1)/accuracy
        if a < fib2:
            break
        tmp = fib1
        fib1 = fib2
        fib2 += tmp
    print(f"required number of iterations for epsilon {accuracy}: {n}")
    print(f"minimal value found by fibonacci_method: {fibonacci_method(func, x1, x2, n)}\n")
    print(f"minimal value found by parabolic_interpolation: {parabolic_interpolation(func, x1, x2, accuracy)}\n")
    print(f"minimal value found by brent_method: {brent_method(func, x1, x2, accuracy)}\n\n")
