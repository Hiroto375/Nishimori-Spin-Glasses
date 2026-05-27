import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.hermite import hermgauss

# Gaussian integral: ∫Du f(u)
x, w = hermgauss(80)
u_nodes = np.sqrt(2) * x
u_weights = w / np.sqrt(np.pi)

def gauss_int(f):
    return np.sum(u_weights * f(u_nodes))

def solve_m0(beta_s, n_iter=10000):
    m0 = 0.9
    for _ in range(n_iter):
        m0_new = np.tanh(beta_s * m0)
        if abs(m0_new - m0) < 1e-12:
            break
        m0 = m0_new
    return m0

def trace_xi(beta_s, m0, func):
    total = 0.0
    for xi in [-1, 1]:
        weight = np.exp(beta_s * m0 * xi)
        total += weight * func(xi)
    return total / (2 * np.cosh(beta_s * m0))

def rhs_m(m, beta, beta_m, beta_s, m0, tau0, tau):
    def func(xi):
        return gauss_int(
            lambda u: np.tanh(beta_m*m + beta*tau0*xi + beta*tau*u)
        )
    return trace_xi(beta_s, m0, func)

def solve_m(beta, beta_m, beta_s, m0, tau0, tau):
    m = 0.0
    for _ in range(10000):
        m_new = rhs_m(m, beta, beta_m, beta_s, m0, tau0, tau)
        m_new = 0.5*m + 0.5*m_new  # damping
        if abs(m_new - m) < 1e-12:
            break
        m = m_new
    return m

def compute_M(m, beta, beta_m, beta_s, m0, tau0, tau):
    def func(xi):
        return gauss_int(
            lambda u: np.sign(beta_m*m + beta*tau0*xi + beta*tau*u)
        )
    return trace_xi(beta_s, m0, func)

# parameters
beta_s = 1 / 0.9   # source inverse temperature
tau0 = 1.0     # signal strength
tau = 1.0      # noise strength

m0 = solve_m0(beta_s)

Tm = np.linspace(0.01, 2.0, 200)
Ts = 0.9
Ms = []
ms = []

for T in Tm:
    beta = Ts / T
    beta_m = 1 / T   # model temperature dependence

    m = solve_m(beta, beta_m, beta_s, m0, tau0, tau)
    M = compute_M(m, beta, beta_m, beta_s, m0, tau0, tau)

    ms.append(m)
    Ms.append(M)

plt.plot(Tm, Ms, label="M")
plt.xlabel("Temperature $T_m$")
plt.ylabel("Overlap M")
plt.title("Temperature dependence of M")
plt.grid()
plt.legend()
plt.savefig("重なりの修復温度依存性/M vs Tm")
plt.show()