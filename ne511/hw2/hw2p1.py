import numpy
import matplotlib.pyplot as plt

def quadratic(a, b, c):
    r1 = (-b + numpy.sqrt(b**2 - (4*a*c)))/(2*a)
    r2 = (-b - numpy.sqrt(b**2 - (4*a*c)))/(2*a)
    return r1, r2

def analytic(n0, t, lamb, rho, beta, cap_lam):
    w1, w2 = quadratic(cap_lam, (beta - rho + lamb*cap_lam), -lamb)
    result =  (n0/(w1 - w2))*(
        ((rho/cap_lam) - w2)*numpy.exp(w1 * t) + 
        (w1 - (rho/cap_lam))*numpy.exp(w2 * t)
    )
    return result

def approx(n0, t, lamb, rho, beta, cap_lam):
    result = n0 * (
        (beta / (beta - rho))*numpy.exp(t*((lamb*rho)/(beta - rho))) - 
        (rho / (beta - rho))*numpy.exp(t*((-beta + rho)/cap_lam))
    )
    return result


rho_vec = [0.0011, 0.0022, 0.0044, 0.0001]
for case in [0, 1, 2, 3]:
    lam = 0.08
    rho = rho_vec[case]
    beta = 0.0065
    cap_lam = 0.001

    n0 = 1
    t = numpy.linspace(0, 0.1, 40)
    analytic_result = analytic(n0, t, lam, rho, beta, cap_lam)
    approx_result = approx(n0, t, lam, rho, beta, cap_lam)



    plt.figure()
    plt.plot(t, analytic_result, label="Analytic Solution")
    plt.plot(t, approx_result, label="Approximation")
    plt.title(f"Comparison, Case {case + 1}")
    plt.legend()
    plt.ylabel("$n/n_0$")
    plt.xlabel("t [s]")
    plt.ylim([1, 1.75])
    plt.savefig(f"case{case+1}.png")
plt.show()