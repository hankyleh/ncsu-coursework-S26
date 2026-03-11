import numpy
import matplotlib.pyplot as plt
from matplotlib import cm

def phi(x, y, mu, eta, S, sigma):
    s =  numpy.less(x/mu, y/eta)*(x/mu) + numpy.greater_equal(x/mu, y/eta)*(y/eta)
    print(s)
    phi = (S/sigma)*(1-numpy.exp(-sigma*s))
    return phi, s

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
Z, _ = phi(X, Y, mu, eta, S, sigma)

z_char, s_char = phi(x_char, y_char, mu, eta, S, sigma)

fig = plt.figure()
ax = plt.axes(projection = '3d')
s1 = ax.plot_surface(X, Y, Z, vmin=0 , vmax=numpy.max(Z)*1.2, cmap=cm.cool, alpha=1.0, label="\phi(x, y)")
s2 = ax.plot(x_char, y_char, z_char, color='black', linewidth=2, label="Characteristic curve")
ax.contour(X, Y, Z, colors='black', linewidths=0.25)
plt.title("$Angular flux sketch for  $\eta > \mu > 0$")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$\psi(x, y)$") 
ax.view_init(20, -120)
plt.savefig("soln.png")

fig = plt.figure()
ax = plt.axes()
s1 = ax.pcolormesh(X, Y, Z, cmap=cm.cool, label="\phi(x, y)")
ax.plot(x_char, y_char, color='black', linestyle='--', label="Characteristic curve")
ax.contour(X, Y, Z, colors='black', linewidths=0.25)
plt.title("$Angular flux sketch for  $\eta > \mu > 0$")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.colorbar(mappable=s1, ax=ax)
plt.legend()
plt.savefig("proj.png")

fig = plt.figure()
plt.plot(s_char, z_char, color='black', linestyle='--', label="Characteristic curve" )
plt.xlabel("$s$")
plt.ylabel("$\psi(s)$")
plt.title("Solution along characteristic")
plt.savefig("characteristic.png")

# plt.show()