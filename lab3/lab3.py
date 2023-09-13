import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


# Можешь заменить на лямбду если не похуй
class f1:
    @staticmethod
    def f(point):
        return point[0] ** 2 + (point[1] - 2) ** 2 + 2

    @staticmethod
    def grad(point):
        return 2 * point[0], 2 * point[1] - 4


class f2:
    @staticmethod
    def f(point):
        return 2 * point[0] ** 2 + point[0] * point[1] + point[1] ** 2

    @staticmethod
    def grad(point):
        return 4 * point[0] + point[1], point[0] + 2 * point[1]


class RandFunc:
    def __init__(self, f, grad):
        self.f = f
        self.grad = grad

    def f(self, point):
        return self.f(point)

    def grad(self, point):
        return self.grad(point)


def const_grad_desc(point, func, learning_rate=0.05, tolerance=0.06):
    iteration = 1
    # Заводим список для того чтобы в последствии построить графики
    point = np.array(point)
    points = [point]
    while True:
        # Находим градиент функции и смоотрим, достаточно ли его норма мала
        # Чтобы можно было выйти из цикла
        gradient = np.array(func.grad(point))
        if np.linalg.norm(gradient) < tolerance:
            break

        # Релаксируем точку
        new_point = point - learning_rate * gradient
        points.append(new_point)
        point = new_point

        iteration += 1

    print(f"required number of iterations: {iteration}")
    return points, iteration


# По сути почти ничем не отличается от предыдущего шага
def armijo_grad_desc(point, func, learning_rate=0.5, c=0.5, tolerance=0.06):
    iteration = 1
    point = np.array(point)
    points = [point]

    while True:
        gradient = np.array(func.grad(point))
        if np.linalg.norm(gradient) < tolerance:
            break

        # Armijo condition
        while True:
            # Тут смотрим какие значения мы получим с текущим learning_rate
            new_point = point - learning_rate * gradient
            # Эта поебота нужна для того чтобы сравнить новое значение функции с текущим
            # И потом смотреть, принимать ли текущий learning_rate
            decrease = c * learning_rate * np.linalg.norm(gradient) ** 2

            # Соответственно здесь мы этим и занимаемся
            # В двух словах здесь если новое значение слишком слабо отличается то мы уменьшаем шаг
            if func.f(point) <= func.f(new_point) + decrease:
                learning_rate *= 0.5
            else:
                break

        # Та же релаксация
        new_point = point - learning_rate * gradient
        points.append(new_point)
        point = new_point

        iteration += 1

    print(f"required number of iterations: {iteration}")
    return points, iteration


# Спиздил из второй лабы
def golden_section_method(f, a: float, b: float, epsilon=1e-6) -> float:
    phi = (3 - 5 ** 0.5)/2
    x1 = a + (b - a) * phi
    x2 = b - (b - a) * phi
    f1, f2 = f(x1), f(x2)

    while abs(b - a) > epsilon:
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

    return (a + b)/2


# Так же сам алгоритм ничем не отличается, просто выбираем шаг по-друому
def fastest_grad_desc(point, func, tolerance=0.06):
    iteration = 1
    point = np.array(point)
    points = [point]

    # Я сосал меня ебали я еле вдуплил и то не вдуплил походу
    # Не слушай меня сверху я того самого того этого ебал реп
    f_lr = lambda lr: func.f(point - np.dot(lr, gradient))

    while True:
        gradient = np.array(func.grad(point))
        if np.linalg.norm(gradient) < tolerance:
            break


        # Ищем learning_rate с помощью золотого сечения. Как нахуй? А вот так
        # Закидываем лямбда функцию в золотое сечение, с её помощью находим такой learing_rate,
        #  Чтобы для текущего градиента мы получили наименьшее значение нашей функции
        learning_rate = golden_section_method(f_lr, 0, 1, 0.05)
        new_point = point - learning_rate * gradient
        points.append(new_point)
        point = new_point

        iteration += 1

    print(f"required number of iterations: {iteration}")
    return points, iteration


# Можете почитать про этот метод если не похуй как я понял есть несколько реализаций
# Одна хуета полная которую я не понял от слова совсем,
# Две другие основываются на методах Флетчера-Ривса и Полака-Рибьера
# которые друг от друга почти ничемм не отличаются. У нас Флетчер
def conjugate_grad_desc(point, func, tolerance=0.06):
    iteration = 1
    point = np.array(point)
    points = [point]

    f_lr = lambda lr: func.f(point - np.dot(lr, gradient))
    gradient = np.array(func.grad(point))

    while True:
        if np.linalg.norm(gradient) < tolerance:
            break

        learning_rate = golden_section_method(f_lr, 0, 1, 0.05)

        new_point = point - learning_rate * gradient
        new_gradient = np.array(func.grad(new_point))

        gamma = np.dot(new_gradient, new_gradient) / np.dot(gradient, gradient)
        gradient = new_gradient

        point = new_point - np.dot(gamma, gradient)
        points.append(point)
        iteration += 1

    print(f"required number of iterations: {iteration}")
    return points, iteration


def generate_quadratic_func(n, k):
    A = np.random.rand(n, n)
    A = np.dot(A, A.transpose())

    b = np.random.rand(n)

    condition_number = np.linalg.cond(A)

    A *= np.sqrt(k / condition_number)

    # Тут тоже можешь заменить на лямбду если не похуй
    def quadratic_function(x):
        return 0.5 * np.dot(x, np.dot(A, x)) - np.dot(b, x)

    def gradient(x):
        return np.dot(A, x) - b

    return RandFunc(quadratic_function, gradient)


init_point = (-10, 8)

points_const, iteration_const = const_grad_desc(init_point, f2)
print(points_const[-1])

points_armijo, iteration_armijo = armijo_grad_desc(init_point, f2)
print(points_armijo[-1])

points_fastest, iteration_fastest = fastest_grad_desc(init_point, f2)
print(points_fastest[-1])

points_conjugate, iteration_conjugate = conjugate_grad_desc(init_point, f2)
print(points_conjugate[-1])

# plt.ion()
# fig, ax = plt.subplots()
# X, Y = np.meshgrid(np.linspace(-10, 10, 10), np.linspace(-8, 12, 10))
# Z = f1.f(X, Y)
#
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot_wireframe(X, Y, Z, color ='royalblue')
#
#
# for i in range(len(points[0])):
#     x = points[0][i]
#     y = points[1][i]
#     ax.scatter(x, y, f1.f(x, y), color='red')
#
# # ax.scatter(X, Y, Z, color='red')
#
# plt.show()


def draw_level(points, func, x_min=-10, y_min=-10, x_max=10, y_max=10):
    x = [row[0] for row in points]
    y = [row[1] for row in points]

    X, Y = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = func.f((X, Y))
    plt.contour(X, Y, Z,  levels=30)
    plt.plot(x, y, marker="o")
    plt.show()


draw_level(points_armijo, f1, x_min=-15, x_max=5, y_max=15)

# for f1
# required number of iterations: 58
# [-0.02465035  2.01479021]
# required number of iterations: 10
# [-0.01953125  2.01171875]
# required number of iterations: 3
# [0.00453104 1.99728138]
# required number of iterations: 3
# [-0.00433454  2.00260072]

# for f2
# required number of iterations: 70
# [-0.01436334  0.03467562]
# required number of iterations: 13
# [-0.01003087  0.02421665]
# required number of iterations: 9
# [-0.01178931  0.00848169]
# required number of iterations: 9
# [-0.0076211   0.01553793]
