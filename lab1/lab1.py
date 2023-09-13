import numpy as np
import matplotlib.pyplot as plt


# функции, используемые в лабе, их производные и интегралы
def f(x0):
    return np.sin(x0)


def g(x0):
    return 1 / x0


def f_prime(x0):
    return np.cos(x0)


def g_prime(x0):
    return -1 / x0 ** 2


def f_integral(x0):
    return -np.cos(x0)


def g_integral(x0):
    return np.log(abs(x0))


# task 1
def forward_difference(func: callable, x0, step: float):
    return (func(x0 + step) - func(x0)) / step


def central_difference(func: callable, x0, step: float):
    return (func(x0 + step) - func(x0 - step)) / (2 * step)


def backward_difference(func: callable, x0, step: float):
    return (func(x0) - func(x0 - step)) / step


x1 = 1
h = 0.1

# 0.50
print(forward_difference(f, x1, h))

# 0.54
print(central_difference(f, x1, h))

# 0.58
print(backward_difference(f, x1, h))


# task 2
# Определяем точки сетки
x2 = np.linspace(np.pi / 2, 4.5 * np.pi, 100)

# Вычисляем численные производные с шагом h = 0.1
h = 0.1
df_num = forward_difference(f, x2, h)
dg_num = forward_difference(g, x2, h)

# Строим графики
plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
plt.plot(x2, f_prime(x2), "b-", label="f'(x)")
plt.plot(x2, df_num, "r-", label="f'(x) calc")
plt.title("y = cos(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(x2, g_prime(x2), "b-", label="g'(x)")
plt.plot(x2, dg_num, "r-", label="g'(x) calc")
plt.title("y = -1/x^2")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.tight_layout(pad=1.7)
plt.show()

# сразу видно дальше тип не вывез стал юзать гптшку
# task 3
x3 = np.linspace(0, np.pi, 100)

true_derivatives = f_prime(x3)

dx = x3[1] - x3[0]
numerical_derivatives = forward_difference(f, x3, dx)

std_dev = np.std(true_derivatives - numerical_derivatives)

print("Standard deviation: ", std_dev)

# task 4
x4 = np.linspace(0, np.pi, 100)

true_derivatives = f_prime(x4)

steps = [np.pi / 2 ** i for i in range(5)]

std_devs = []
for step in steps:
    numerical_derivatives = forward_difference(f, x4, step)
    std_dev = np.std(true_derivatives - numerical_derivatives)
    std_devs.append(std_dev)

plt.plot(steps, std_devs, 'o-')
plt.xscale('log')
plt.xlabel('Step size')
plt.ylabel('Standard deviation')
plt.show()


# task 5
def left_rectangles(func: callable, left: float, right: float, step: float) -> float:
    result = 0
    i = 1
    while left + step * i <= right:
        result += step * func(left + step * (i - 1))
        i += 1

    return result


def right_rectangles(func: callable, left: float, right: float, step: float) -> float:
    result = 0
    i = 1
    while left + step * i < right:
        result += step * func(left + step * i)
        i += 1

    return result


def central_rectangles(func: callable, left: float, right: float, step: float) -> float:
    result = 0
    i = 1
    while left + step * i <= right:
        result += step * func(left + step * (i - 0.5))
        i += 1

    return result


def trapezoid(func: callable, left: float, right: float, step: float) -> float:
    result = 0
    i = 1
    while left + step * i <= right:
        result += 0.5 * step * (func(left + step * (i - 1)) + func(left + step * i))
        i += 1

    return result


def simpson(func: callable, left: float, right: float, step: float) -> float:
    result = 0
    i = 1
    while left + step * i <= right:
        result += 1/6 * step * (func(left + step * (i - 1)) + 4 * func(left + step * (i - 0.5)) + func(left + step * i))
        i += 1

    return result


# task 6
def calc_integral(integrated_func: callable, left: float, right: float):
    return integrated_func(right) - integrated_func(left)


xL = np.pi
xR = 5 * np.pi
dx = 4 * np.pi / 100


def print_results(integrated_func: callable, og: callable):
    print("\n")
    print(f"Theoretical value: {calc_integral(integrated_func, xL, xR)}")
    print(f"Value calculated with left rectangles method: {left_rectangles(og, xL, xR, dx)}")
    print(f"Value calculated with right rectangles method: {right_rectangles(og, xL, xR, dx)}")
    print(f"Value calculated with central rectangles method: {central_rectangles(og, xL, xR, dx)}")
    print(f"Value calculated with trapezoids method: {trapezoid(og, xL, xR, dx)}")
    print(f"Value calculated with Simpson's method: {simpson(og, xL, xR, dx)}")


print_results(f_integral, f)
print_results(g_integral, g)


# task 7
steps = [dx / 2 ** i for i in range(5)]

std_devs = [[0] * len(steps) for i in range(5)]
G = calc_integral(f_integral, xL, xR)

for i, step in enumerate(steps):
    std_devs[0][i] = abs(left_rectangles(f, xL, xR, step) - G)
    std_devs[1][i] = abs(right_rectangles(f, xL, xR, step) - G)
    std_devs[2][i] = abs(central_rectangles(f, xL, xR, step) - G)
    std_devs[3][i] = abs(trapezoid(f, xL, xR, step) - G)
    std_devs[4][i] = abs(simpson(f, xL, xR, step) - G)

funcs = ["left_rectangles", "right_rectangles", "central_rectangles", "trapezoid", "simpson"]
for i in range(5):
    plt.plot(steps, std_devs[i], 'o-', label=funcs[i])

plt.xlabel('Step size')
plt.xscale('log')
plt.ylabel('Standard deviation')
plt.legend()
plt.show()
