import numpy
import matplotlib.pyplot as plt

from matplotlib import cm


def phi(x, y, mu, eta, S, sigma):
    s =  numpy.less(x/mu, y/eta)*(x/mu) + numpy.greater_equal(x/mu, y/eta)*(y/eta)
    print(s)
    phi = (S/sigma)*(1-numpy.exp(-sigma*s))
    return phi



a = 12
Nxy = 100

eta = 0.9
mu = numpy.sqrt(1-eta**2)

S = 1
sigma = 0.2


x_points = numpy.linspace(0, a, Nxy)
y_points = x_points

x_char = x_points
y_char = x_points*eta/mu

delete_index = numpy.logical_or(numpy.greater(x_char, a) , numpy.greater(y_char, a))
x_char = numpy.delete(x_char, delete_index)
y_char = numpy.delete(y_char, delete_index)

X, Y = numpy.meshgrid(x_points, y_points)
Z = phi(X, Y, mu, eta, S, sigma)

z_char = phi(x_char, y_char, mu, eta, S, sigma)


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
s1 = ax.plot_surface(X, Y, Z, vmin=0 , vmax=numpy.max(Z)*1.2, cmap=cm.cool, alpha=0.8, label="\phi(x, y)")
s2 = ax.plot(x_char, y_char, z_char, color='black', linewidth=2, label="Characteristic curve")
plt.title("$\psi(x, y)$;   $\eta > \mu > 0$")

fig, ax = plt.subplots()
ax.pcolormesh(X, Y, Z, cmap=cm.cool, label="\phi(x, y)")
ax.plot(x_char, y_char, color='black', linestyle='--', label="Characteristic curve")
plt.legend()
plt.title("$\psi(x, y)$;   $\eta > \mu > 0$")


plt.show()
