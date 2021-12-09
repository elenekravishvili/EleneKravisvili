import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

alpha=600
beta=1.57
def K(T):
    return alpha+beta*T
kappa=300
gamma=0.5
def Cp(T):
    return kappa+gamma*T
delta=500
omega=0.7
def rho(T):
    return delta+omega*T

h=400                #heat transfer coefficient
Tg=100               #temperature of surraundings
epsilon=0.7          #emiisivity
sigma=5.67*10**(-8)

#radiation boundary condition
def q_right(T):
    return -sigma*epsilon*(T**4-Tg**4)

#radiation and Newtonian boundary conditions
def q_left(T):
    return h*(Tg-T)-sigma*epsilon*(T**4-Tg**4)

L=10
T0_init=200
#initial distribution of temperature in the rod
def g(x):
    return  T0_init*np.cos(np.pi*x/L)**2

t_init=0.0
t_final=5000
N_t=301         #number of time intervals
dt=(t_final-t_init)/N_t
t_array=np.linspace(t_init,t_final,N_t)


x_init=0
x_final=L
R=170
N_x = R + 1           #number of coordinate intervals
dx=(x_final-x_init)/N_x
x_array=np.linspace(x_init,x_final,N_x)


#define T in the integer and non-integer coordinates
def T_(T_n_array, x_i):
    if (2*x_i)%2 == 0:
        return T_n_array[x_i]
    else:
        return (T_n_array[int(x_i-1/2)] + T_n_array[int(x_i+1/2)])/2
    
#define T in the non-integer time moment, in the (n+1/2)th moment
def T_half_time_p(n, i, T_sol):
    T_nm1_i = T_(T_sol[n - 1], i)
    T_nm1_im1 = T_(T_sol[n - 1], i - 1)
    T_nm1_ip1 = T_(T_sol[n - 1], i + 1)
    T_nm1_im05 = T_(T_sol[n - 1], i - 1 / 2)
    T_nm1_ip05 = T_(T_sol[n - 1], i + 1 / 2)

    return T_nm1_i + (dt / (2 * rho(T_nm1_i) * Cp(T_nm1_i) * (dx ** 2))) * (
                K(T_nm1_ip05) * (T_nm1_ip1 - T_nm1_i) - K(T_nm1_im05) * (T_nm1_i - T_nm1_im1))
    
#define the coefficients
def A(T_np05_array, i):
    if i==0:
        return 0 # Remove this
    elif i==R:
        return 0
    else:
        T_np05_im05 = T_(T_np05_array, i-1/2)
    return (1/(2*(dx**2)))*K(T_np05_im05)

def B(T_np05_array,i):
    if i==0:
        return 1
    elif i==R:
        return 1
    else:
        T_np05_i    = T_(T_np05_array, i)
        T_np05_ip05 = T_(T_np05_array, i+1/2)
        T_np05_im05 = T_(T_np05_array, i-1/2)
    return -((1/(2*(dx**2)))*(K(T_np05_ip05) + K(T_np05_im05)) + rho(T_np05_i)*Cp(T_np05_i)/dt)

def C(T_np05_array,i):
    if i==0:
        return 0
    elif i==R:
        return 0 # Remove this
    else:
        T_np05_ip05 = T_(T_np05_array, i+1/2)
    return (1/(2*(dx**2)))*K(T_np05_ip05)

def D(T_np05_array, n, i, T_sol, F_root_right, F_root_left):
    if i==0:
        return F_root_left
    elif i==R:
        return F_root_right            #solution by minimization of the function delta T at the R point
    else:
        T_nm1_i    = T_(T_sol[n-1], i)
        T_nm1_im1  = T_(T_sol[n-1], i-1)
        T_nm1_ip1  = T_(T_sol[n-1], i+1)
        T_np05_i    = T_(T_np05_array, i)
        T_np05_ip05 = T_(T_np05_array, i+1/2)
        T_np05_im05 = T_(T_np05_array, i-1/2)
    return -(1/(2*(dx**2)))*(K(T_np05_ip05)*(T_nm1_ip1 - T_nm1_i) - K(T_np05_im05)*(T_nm1_i - T_nm1_im1)) - (rho(T_np05_i)*Cp(T_np05_i)/dt)*T_nm1_i

#initialization of thomas algorithm to solve the themperature in any moment of time
def TDMAsolver(a, b, c, d):
    nf = len(d)  # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d))  # copy arrays
    for it in range(1, nf):
        mc = ac[it - 1] / bc[it - 1]
        bc[it] = bc[it] - mc * cc[it - 1]
        dc[it] = dc[it] - mc * dc[it - 1]

    xc = bc
    xc[-1] = dc[-1] / bc[-1]

    for il in range(nf - 2, -1, -1):
        xc[il] = (dc[il] - cc[il] * xc[il + 1]) / bc[il]

    return xc


#initialization of regula-falsi method to solve temperature at the boundaries
def F_root(F,x0_init,x1_final,x2_guess):
    x0=x0_init
    x1=x1_final
    step=0
    x2=x2_guess
    condition = abs(F(x2)) > 0.0001
    while condition:
        x2 = x0 - (x1 - x0) * F(x0) / (F(x1) - F(x0))

        if F(x0) * F(x2) < 0:
            x1 = x2
        else:
            x0 = x2

        step = step + 1
        condition = abs(F(x2)) > 0.0001

    return x2

#define the functions, after optimization they will give us the solution of temperature at the boundaries
def F_left(T, T_sol, n):
    return K(T_sol[n-1,0])*T-K(T_sol[n-1,1])*T_sol[n-1,1]-dx*q_left(T)
def F_right(T, T_sol, n):
    return K(T_sol[n-1,R-1])*T_sol[n-1,R-1]-K(T_sol[n-1,R])*T-dx*q_right(T)


# Initialize T_sol initial condition
T_sol = np.zeros((N_t, N_x))
T_sol[0, :] = g(x_array)
plt.figure()
plt.plot(x_array, T_sol[0, :])
plt.title("Plot of initial T at n=0, 10, 50, 100, 150, 200, 290 moments")
plt.grid(True)


for n in range(1, N_t):
    T_sol[n, 0] = F_root(lambda T: F_left(T, T_sol, n), 0, 5000, 1)
    T_sol[n,-1] = F_root(lambda T: F_right(T, T_sol, n), 0, 5000, 1)
    T_np05_array = [T_half_time_p(n, i, T_sol) for i in range(R)]
    T_np05_array.append(T_np05_array[-1]+0.01)

    # Compute coefficients
    A_values = [A(T_np05_array, i) for i in range(1,R+1)]
    B_values = [B(T_np05_array, i) for i in range(R+1)]
    C_values = [C(T_np05_array, i) for i in range(R)]
    D_values = [D(T_np05_array, n, i, T_sol, T_sol[n, -1], T_sol[n, 0]) for i in range(R+1)]

    T_sol[n, :] = TDMAsolver(A_values, B_values, C_values, D_values)




plt.plot(x_array[:], T_sol[10, :])
plt.plot(x_array[:], T_sol[200, :])
plt.plot(x_array[:], T_sol[100, :])
plt.plot(x_array[:], T_sol[0, :])
plt.plot(x_array[:], T_sol[50, :])
plt.plot(x_array[:], T_sol[150, :])
plt.plot(x_array[:], T_sol[290, :])


fig = plt.figure()
plts = []
plt.grid(True)
for i in range(0, T_sol.shape[0], 2):
    p, = plt.plot(x_array, T_sol[i, :], 'k')
    plt.title("Plot of Temperature")
    plts.append([p])
ani = animation.ArtistAnimation(fig, plts, interval=1, repeat_delay=1)

#ani.save("anim_random_cos_colder.gif", writer="pillow")
plt.show()