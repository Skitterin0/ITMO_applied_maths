import numpy as np
import matplotlib.pyplot as plt


class f1:
    @staticmethod
    def f(point):
        return np.array([point[0] ** 2 + (point[1] - 2) ** 2 + 2])

    @staticmethod
    def grad(point):
        return np.array([2 * point[0], 2 * point[1] - 4])


class f2:
    @staticmethod
    def f(point):
        return np.array([2 * point[0] ** 2 + point[0] * point[1] + point[1] ** 2])

    @staticmethod
    def grad(point):
        return np.array([4 * point[0] + point[1], point[0] + 2 * point[1]])


class RandFunc:
    def __init__(self, f, grad):
        self.f = f
        self.grad = grad

    def f(self, point):
        return self.f(point)

    def grad(self, point):
        return self.grad(point)


def const_grad_desc(point, func, learning_rate=0.005, tolerance=0.06):
    count_it, count_f, count_g = 0, 0, 0
    points = [point]
    while True:
        # Находим градиент функции и смотрим, достаточно ли его норма мала
        # Чтобы можно было выйти из цикла
        gradient = func.grad(point)
        count_g += 1
        if np.linalg.norm(gradient) < tolerance:
            break

        # Релаксируем точку
        new_point = point - learning_rate * gradient
        points.append(new_point)
        point = new_point

        count_it += 1

    return points, count_it, count_f, count_g


def armijo_grad_desc(point, func, learning_rate=1.0, c=0.5, tolerance=0.06):
    count_it, count_f, count_g = 0, 0, 0
    points = [point]

    while True:
        gradient = func.grad(point)
        count_g += 1

        if np.linalg.norm(gradient) < tolerance:
            break

        f_value = func.f(point)
        count_f += 1

        # Armijo condition
        while True:
            new_point = point - learning_rate * gradient
            new_f_value = func.f(new_point)
            count_f += 1
            decrease = c * learning_rate * gradient @ gradient

            if f_value - decrease <= new_f_value:
                learning_rate *= 0.5
            else:
                break

        point = point - learning_rate * gradient
        points.append(point)

        count_it += 1

    return points, count_it, count_f, count_g


def golden_section_method(f, a: float, b: float, epsilon=1e-4):
    phi = (3 - 5 ** 0.5)/2
    x1 = a + (b - a) * phi
    x2 = b - (b - a) * phi
    f1, f2 = f(x1), f(x2)
    count_f = 2
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
        count_f += 1

    return (a + b)/2, count_f


def fastest_grad_desc(point, func, tolerance=0.06):
    count_it, count_f, count_g = 0, 0, 0
    points = [point]

    f_lr = lambda lr: func.f(point - lr * gradient)

    while True:
        gradient = func.grad(point)
        count_g += 1
        if np.linalg.norm(gradient) < tolerance:
            break

        learning_rate, count = golden_section_method(f_lr, 0, 100)
        count_f += count
        new_point = point - learning_rate * gradient
        points.append(new_point)
        point = new_point

        count_it += 1

    return points, count_it, count_f, count_g


def conjugate_grad_desc(point, func, tolerance=0.0001):
    count_it, count_f, count_g = 0, 0, 0
    points = [point]
    N = len(point)
    k = 0

    f_lr = lambda lr: func.f(point - lr * gradient)
    gradient = func.grad(point)
    count_g += 1

    while np.linalg.norm(gradient) > tolerance:
        learning_rate, count = golden_section_method(f_lr, 0, 100)
        count_f += count

        new_point = point - learning_rate * gradient
        k += 1

        if k == N:
            k = 0
            gradient = func.grad(new_point)
            count_g += 1
        else:
            tmp_grad = func.grad(new_point)
            prev_grad = func.grad(point)
            lr_2 = (tmp_grad @ tmp_grad) / (prev_grad @ prev_grad)
            gradient = lr_2 * gradient + tmp_grad
            count_g += 2

        points.append(new_point)
        point = new_point
        count_it += 1

    return points, count_it, count_f, count_g


init_point = np.array([-10, 8])
tolerance = 0.001
lr = 0.05

func = RandFunc(f1.f, f1.grad)

print("point, iters, f_count, g_count")
points_const, iteration_const, f_const, grad_const = const_grad_desc(init_point, func, learning_rate=lr, tolerance=tolerance)
print(points_const[-1], iteration_const, f_const, grad_const)

points_armijo, iteration_armijo, f_armijo, grad_armijo = armijo_grad_desc(init_point, func, tolerance=tolerance)
print(points_armijo[-1], iteration_armijo, f_armijo, grad_armijo)

points_fastest, iteration_fastest, f_fastest, grad_fastest = fastest_grad_desc(init_point, func, tolerance=tolerance)
print(points_fastest[-1], iteration_fastest, f_fastest, grad_fastest)

points_conjugate, iteration_conjugate, f_conjugate, grad_conjugate = conjugate_grad_desc(init_point, func, tolerance=tolerance)
print(points_conjugate[-1], iteration_conjugate, f_conjugate, grad_conjugate)


def draw_level(points, func, x_min=-10, y_min=-10, x_max=10, y_max=10):
    x = [row[0] for row in points]
    y = [row[1] for row in points]

    X, Y = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = func.f((X, Y))
    plt.contour(X, Y, Z[0],  levels=30)
    plt.plot(x, y, marker="o")
    plt.show()


draw_level(points_const, func, x_min=-15, x_max=5, y_max=15)
draw_level(points_armijo, func, x_min=-15, x_max=5, y_max=15)
draw_level(points_fastest, func, x_min=-15, x_max=5, y_max=15)
draw_level(points_conjugate, func, x_min=-15, x_max=5, y_max=15)


def generate_quadratic_function(n, k):
    while True:
        B = np.random.normal(size=(n, n))
        A = B.T @ B

        cond = np.linalg.cond(A)

        if cond <= k:
            break

        s = np.sqrt(cond / k)
        A = (1 / s) * A + (s - 1) * np.eye(n)

    b = np.random.normal(size=n)
    c = np.random.normal()

    def quadratic_function(x):
        return x.T @ A @ x + b.T @ x + c

    def gradient(x):
        return 2 * A @ x + b

    x0 = np.random.normal(size=n)

    return quadratic_function, gradient, x0


def graphic(X, Y, Z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(X, Y)
    Z = np.array(Z).reshape(X.shape)

    ax.plot_surface(X, Y, Z)

    ax.set_xlabel('n')
    ax.set_ylabel('k')
    ax.set_zlabel('Number of iterations')

    plt.show()


def gen_test_const():
    Z_1 = []
    Z_2 = []
    Z_3 = []
    Z_4 = []
    X = range(2, 7)
    Y = range(3, 15)
    res_1 = 0
    res_2 = 0
    res_3 = 0
    res_4 = 0
    for x in X:
        for y in Y:
            for i in range(10):
                if (x - y >= 2):
                    break
                f, df, p_0 = generate_quadratic_function(x, y)
                func = RandFunc(f, df)
                history, it_1 = const_grad_desc(p_0, func, tolerance=tolerance)
                history, it_2 = armijo_grad_desc(p_0, func, tolerance=tolerance)
                history, it_3 = fastest_grad_desc(p_0, func, tolerance=tolerance)
                history, it_4 = conjugate_grad_desc(p_0, func, tolerance=tolerance)
                res_1 += it_1
                res_2 += it_2
                res_3 += it_3
                res_4 += it_4
            Z_1.append(res_1 // 10)
            Z_2.append(res_2 // 10)
            Z_3.append(res_3 // 10)
            Z_4.append(res_4 // 10)

    print("Метод градиентного спуска с постоянным шагом")
    graphic(X, Y, Z_1)
    print("Метод градиентного спуска с дроблением шага")
    graphic(X, Y, Z_2)
    print("Наискорейший спуск")
    graphic(X, Y, Z_3)
    print("Метод сопряженных градиентов")
    graphic(X, Y, Z_4)


# gen_test_const()