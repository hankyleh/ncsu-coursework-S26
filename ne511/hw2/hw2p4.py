import numpy
import matplotlib.pyplot as plt

def sppstep(t, pm, pb, pp1):
    result = pm/((numpy.cosh((0.5*pb)*(t-(16.5/pp1))))**2)
    return result

pm = 388.5
pb = 1.001289
lam = 1/0.62e4

time = numpy.linspace(150, 180, 300)
result = sppstep(time, pm, pb, 0.1)

plt.figure()
plt.plot(time, result)
plt.ylabel("$p/p_0$", fontsize=13)
plt.xlabel("$t/\Lambda$", fontsize=13)
plt.title("Pulse")
plt.savefig("pulse.png")
