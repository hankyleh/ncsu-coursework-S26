import numpy as np
import matplotlib.pyplot as plt

a = 1
S = 1


def lam(n):
    global a
    return (2*n + 1)*np.pi/(2*a)

def alpha(m):
    global S
    result = ((-1)**m)*((S/(np.pi*lam(m)**2))*(4/(2*m + 1)))
    return result

def f(x, n):
    global a
    return np.cos((2*n + 1)*(np.pi)*x/(2*a))



x_mesh = np.linspace(-a+0.001, a-0.001, 100)
true = (S/2)*( a**2 - (x_mesh)**2 )

fig, (sol_ax, err_ax) = plt.subplots(1, 2, dpi=300)

fig.set_figheight(3)
fig.set_figwidth(8)

sol_ax.plot(x_mesh, true, "--", label="True solution")


n_list = [1, 3, 5, 7]

for max_n in n_list:

    recon = np.zeros((x_mesh.size))
    for n in range(0, max_n+1):
        recon += (alpha(n))*f(x_mesh, n)
    sol_ax.plot(x_mesh, recon, label=f"n={n}")

    error = (recon/true)-1
    err_ax.plot(x_mesh, error, label=f"n={n}")

sol_ax.legend()
err_ax.legend()

sol_ax.set_xlabel("x")
sol_ax.set_title("Solutions")

err_ax.set_xlabel("x")
err_ax.set_title("Relative error")

fig.savefig('figure.png')
plt.close()