import numpy
import matplotlib.pyplot as plt


def p0(x):
    return numpy.sqrt(1/2)+(x*0)
def p1(x):
    return numpy.sqrt(3/2)*x
def p2(x):
    return numpy.sqrt(5/2)*0.5*(3*x**2 - 1)
def pta(x):
    return 3*x +7
def ptb(x):
    return numpy.exp(x)
def ptc(x):
    return numpy.exp(5*x)

def p2approx(x, f0, f1, f2):
    return (f0*p0(x))+(f1*p1(x))+(f2*p2(x))



x_plt = numpy.linspace(-1, 1, 100)


# Part a

f0 = 7*numpy.sqrt(2)
f1 = numpy.sqrt(6)
f2 = 0

plt.figure()
plt.plot(x_plt, pta(x_plt), linewidth=4)
plt.plot(x_plt, p2approx(x_plt, f0, f1, f2), "-.", linewidth=3, color="black")
plt.legend(["$f(x)$", "$P_2$ Approx"])
plt.xlabel("x")
plt.ylabel("f(x)")
plt.axvline(0, 0, color="black", linewidth=0.5)
plt.title("Part a")
plt.savefig("parta.png")


# Part b

f0 = numpy.sqrt(1/2)*(numpy.exp(1) - numpy.exp(-1))
f1 = numpy.sqrt(6)*numpy.exp(-1)
f2 = 0.5*numpy.sqrt(5/2)*(2*numpy.exp(1) - 14*numpy.exp(-1))

plt.figure()
plt.plot(x_plt, ptb(x_plt), linewidth=4)
plt.plot(x_plt, p2approx(x_plt, f0, f1, f2), "-.", linewidth=3, color="black")
plt.legend(["$f(x)$", "$P_2$ Approx"])
plt.xlabel("x")
plt.ylabel("f(x)")
plt.axvline(0, 0, color="black", linewidth=0.5)
plt.title("Part b")
plt.savefig("partb.png")

# Part c

f0 = (1/(5*numpy.sqrt(2)))*(numpy.exp(5) - numpy.exp(-5))
f1 = (1/25)*numpy.sqrt(3/2)*(4*numpy.exp(5)-6*numpy.exp(-5))
f2 = (1/250)*numpy.sqrt(5/2)*(26*numpy.exp(5) - 86*numpy.exp(-5))

plt.figure()
plt.plot(x_plt, ptc(x_plt), linewidth=4)
plt.plot(x_plt, p2approx(x_plt, f0, f1, f2), "-.", linewidth=3, color="black")
plt.legend(["$f(x)$", "$P_2$ Approx"])
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Part c")
plt.axvline(0, 0, color="black", linewidth=0.5)
plt.savefig("partc.png")




