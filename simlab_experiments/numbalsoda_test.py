import numba as nb
from numba import types, carray, experimental
import numpy as np
from numbalsoda import lsoda_sig, solve_ivp
from numbalsoda import address_as_void_pointer

def rhs_numbalsoda(t, clust_conc, dcdt, p):
    system_size = len(p.km)
    
    # Initialize space for dcdt
    dcdt.fill(0)

    kmc = p.km * clust_conc
    kpc = p.kp * clust_conc[0] * clust_conc

    # Treat first and last DEs as special cases
    dcdt[0] = np.sum(kmc[2:]) + 2 * kmc[1] - np.sum(kpc[1:]) - 2 * p.kp[0] * clust_conc[0]**2
    
    for i in range(1, system_size-1):
        dcdt[i] = kpc[i-1] - kpc[i] - kmc[i] + kmc[i+1]

    dcdt[-1] = -kmc[-1] - kpc[-1] + p.kp[-2] * clust_conc[0] * clust_conc[-2]

# Define the args_dtype for the C struct
args_dtype = types.Record.make_c_struct([
    ('km_p', types.uintp),
    ('km_len', types.int64),
    ('kp_p', types.uintp),
    ('kp_len', types.int64),
])

# Define the spec for the UserData class
spec = [
    ('km', types.double[:]),
    ('kp', types.double[:]),
]

@nb.experimental.jitclass(spec)
class UserData():
    
    def __init__(self, km=None, kp=None):    
        self.km = km
        self.kp = kp
        
    def make_args_dtype(self):
        args = np.zeros(1, dtype=args_dtype)
        args[0]['km_p'] = self.km.ctypes.data
        args[0]['km_len'] = self.km.shape[0]
        args[0]['kp_p'] = self.kp.ctypes.data
        args[0]['kp_len'] = self.kp.shape[0]
        return args
    
    def unpack_pointer(self, user_data_p):
        # Takes in pointer, and unpacks it
        user_data = carray(user_data_p, 1)[0]
        self.km = carray(address_as_void_pointer(user_data.km_p), (user_data.km_len,), dtype=np.float64)
        self.kp = carray(address_as_void_pointer(user_data.kp_p), (user_data.kp_len,), dtype=np.float64)

# This function will create the numba function to pass to lsoda.
def create_jit_fcns(rhs, args_dtype):
    jitted_rhs = nb.njit(rhs)
    @nb.cfunc(types.void(types.double,
             types.CPointer(types.double),
             types.CPointer(types.double),
             types.CPointer(args_dtype)))
    def wrapped_rhs(t, u, du, user_data_p):    
        p = UserData()
        p.unpack_pointer(user_data_p)
        jitted_rhs(t, u, du, p) 
    
    return wrapped_rhs

# Create the function to be called
rhs_cfunc = create_jit_fcns(rhs_numbalsoda, args_dtype)

def solve_ODE_system_numbalsoda(c0, t_values, km, kp, eps_value=1e-6):
    p = UserData(km, kp)
    args = p.make_args_dtype()

    funcptr = rhs_cfunc.address
    
    # Solve ODE system
    t_eval = np.array(t_values)
    t_span = np.array([min(t_eval), max(t_eval)])
    sol = solve_ivp(funcptr, t_span, c0, 
                    t_eval=t_values, data=args.ctypes.data, rtol=1e-6)

    return sol

# Example usage
km = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
kp = np.array([0.05, 0.1, 0.15, 0.2, 0.25])
c0 = np.array([5.0, 0.8, 1.0, 0.5, 0.3, 0.2])
t_values = np.linspace(0.0, 50.0, 1000)

sol = solve_ODE_system_numbalsoda(c0, t_values, km, kp)
print(sol.t)
print(sol.y)