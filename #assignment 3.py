# #assignment 3 
# import numpy as np
# import matplotlib.pyplot as plt

# # Load the data
# file_path = r'E:\intro to turbulance\Data and Papers_8649701b-b667-467e-823e-9d819f1b826c\isotropic1024_stack3.npz'
# data = np.load(file_path)
# print(data.files)

# # Extract velocity components
# u = data['u']
# v = data['v']
# w = data['w']

# # Define domain parameters
# Nx = Ny = 1024
# Lx = Ly = 2 * np.pi
# dx = dy = dz = Lx / Nx
# L = 1.364  # Characteristic length scale

# # Slice the velocity components at the middle plane
# u_mid = u[:, :, 1]
# v_mid = v[:, :, 1]
# w_mid = w[:, :, 1]

# # Calculate spatial derivatives for the velocity gradient tensor
# # For x-derivatives with periodic boundaries
# dudx = (np.roll(u_mid, -1, axis=0) - np.roll(u_mid, 1, axis=0)) / (2 * dx)
# dvdx = (np.roll(v_mid, -1, axis=0) - np.roll(v_mid, 1, axis=0)) / (2 * dx)
# dwdx = (np.roll(w_mid, -1, axis=0) - np.roll(w_mid, 1, axis=0)) / (2 * dx)

# # For y-derivatives with periodic boundaries
# dudy = (np.roll(u_mid, -1, axis=1) - np.roll(u_mid, 1, axis=1)) / (2 * dy)
# dvdy = (np.roll(v_mid, -1, axis=1) - np.roll(v_mid, 1, axis=1)) / (2 * dy)
# dwdy = (np.roll(w_mid, -1, axis=1) - np.roll(w_mid, 1, axis=1)) / (2 * dy)

# # For z-derivatives using central differences
# dudz = (u[:, :, 2] - u[:, :, 0]) / (2 * dz)
# dvdz = (v[:, :, 2] - v[:, :, 0]) / (2 * dz)
# dwdz = (w[:, :, 2] - w[:, :, 0]) / (2 * dz)

# # Initialize arrays to store eigenvalues
# eigenvalues = np.zeros((Nx, Ny, 3), dtype=complex)
# #find coffiecient of equation
# def characteristic_equation(A):
#     P = -(A[0,0]+A[1,1]+A[2,2])
#     Q = (A[0,0]*A[1,1]+A[0,0]*A[2,2]+A[1,1]*A[2,2]-
#          A[0,1]*A[1,0]-A[0,2]*A[2,0]-A[1,2]*A[2,1])
#     R = -(A[0,0]*(A[1,1]*A[2,2]-A[1,2]*A[2,1])-
#           A[0,1]*(A[1,0]*A[2,2]-A[1,2]*A[2,0])+
#           A[0,2]*(A[1,0]*A[2,1]-A[1,1]*A[2,0]))
#     coefficient = [1,P,Q,R]
#     eigenvalue = np.roots(coefficient)
#     return eigenvalue,Q,R

# # Calculate eigenvalues at each point
# Q_vals =np.zeros((Nx,Ny))
# R_vals =np.zeros((Nx,Ny))
# for i in range(Nx):
#     for j in range(Ny):
#         A = np.array([
#             [dudx[i, j], dudy[i, j], dudz[i, j]],
#             [dvdx[i, j], dvdy[i, j], dvdz[i, j]],
#             [dwdx[i, j], dwdy[i, j], dwdz[i, j]]
#         ])
#         # Compute eigenvalues
#         evals,Q,R = characteristic_equation(A)
#         Q_vals[i,j] = Q
#         R_vals[i,j] = R
#         # Store eigenvalues
#         eigenvalues[i, j, :] = evals
#         # print(evals)

# # eigenvalue_flat =  (eigenvalues).reshape(-1, 3)
# # # Separate and sort eigenvalues
# # lambda_sorted = np.zeros_like(eigenvalue_flat, dtype=complex)

# # for i in range(eigenvalue_flat.shape[0]):
# #     lambda_sorted[i, :] = np.sort(eigenvalue_flat[i, :])
# #     # print(lambda_sorted[i,:])

# # # Now separate them into three arrays
# # lambda3 = np.abs(lambda_sorted[:, 0]) # Smallest eigenvalue
# # lambda2 = np.abs(lambda_sorted[:, 1]) # Middle eigenvalue
# # lambda1 = np.abs(lambda_sorted[:, 2])  # Largest eigenvalue

# # # Plot
# # fig, ax = plt.subplots(figsize=(10, 6))

# # for lambdas, label, color in zip([lambda1, lambda2, lambda3],[r'$\lambda_1$ (Smallest)', r'$\lambda_2$ (Middle)', r'$\lambda_3$ (Largest)'],['blue', 'green', 'red']):
# #     hist, bin_edges = np.histogram(lambdas, bins=200, density=True)
# #     bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
# #     ax.semilogy(bin_centers, hist, label=label, color=color)

# # ax.set_title("PDF of Sorted Eigenvalues (Semilog Y-axis)")
# # ax.set_xlabel("Eigenvalue")
# # ax.set_ylabel("Probability Density (log scale)")
# # ax.legend()
# # ax.grid(True, which='both', linestyle='--', linewidth=0.5)
# # plt.tight_layout()
# # plt.show()

# # print("|P| is ",np.mean(dudx+dvdy+dwdz))


# # Q_rms = np.sqrt(np.mean(Q_vals**2))
# # R_rms = np.sqrt(np.mean(R_vals**2))

# # Q_norm = Q_vals / Q_rms
# # R_norm = R_vals / R_rms

# # # Plot the first image
# # plt.figure(figsize=(20, 6))
# # plt.subplot(1, 2, 1)
# # plt.imshow(Q_norm, cmap='viridis', aspect='auto', vmin=-1, vmax=1)
# # plt.colorbar(label='Normalized Q')
# # plt.title('Normalized Q')

# # # Plot the second image
# # plt.subplot(1, 2, 2)
# # plt.imshow(R_norm, cmap='viridis', aspect='auto', vmin=-1, vmax=1)
# # plt.colorbar(label='Normalized R')
# # plt.title('Normalized R')
# # plt.tight_layout()
# # plt.show()

# omega = ((dwdy-dvdz)**2)+((dudz-dwdx)**2)+((dvdx-dudy)**2)
# Q_w = np.mean(omega) / 4
# Q_scatter = Q_vals / Q_w
# R_scatter = R_vals / (Q_w**(3/2))

# R_vals1 = np.linspace(np.min(R_scatter), np.max(R_scatter), 1000)
# Q_curve = -((27/4) * R_vals1**2)**(1/3)

# plt.figure(figsize=(10, 6)) 
# plt.scatter(R_scatter, Q_scatter, s=1, color='black', alpha=0.5, label='Data Points')
# plt.plot(R_vals1, Q_curve, color='red', label='Discriminant Curve')
# plt.xlabel('R', fontsize=14)
# plt.ylabel('Q', fontsize=14)
# plt.legend()
# plt.title('Q-R Scatter with Discriminant Curve')
# plt.grid(True, which="both", linestyle="--")

# # Set limits to center (0,0) in the plot
# plt.xlim()
# plt.ylim()

# plt.tight_layout()
# plt.show()
# #part 2

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import matplotlib.cm as cm

# Load the data
file_path = r'E:\intro to turbulance\Data and Papers_8649701b-b667-467e-823e-9d819f1b826c\isotropic1024_stack3.npz'
data = np.load(file_path)
print(data.files)

# Extract velocity components
u = data['u']
v = data['v']
w = data['w']

# Define domain parameters
Nx = Ny = 1024
Lx = Ly = 2 * np.pi
dx = dy = dz = Lx / Nx
L = 1.364  # Characteristic length scale

# Slice the velocity components at the middle plane
u_mid = u[:, :, 1]
v_mid = v[:, :, 1]
w_mid = w[:, :, 1]

# Calculate scales
nu = 0.000185
U_sum = np.sum(u_mid**2 + v_mid**2)
U_mean = U_sum / (Nx * Ny)
U_rms = np.sqrt(U_mean)
Epsilon = U_rms**3 / L
print('Epsilon:', Epsilon)
eta = (nu**3 / Epsilon)**0.25
print('Kolmogorov length scale eta:', eta)
tau_eta = np.sqrt(nu / Epsilon)
dt = tau_eta
T = 10
total_step = int(T / dt)

# Grid and interpolators
x = np.linspace(0, Lx, Nx, endpoint=False)
y = np.linspace(0, Ly, Ny, endpoint=False)
X, Y = np.meshgrid(x, y)
u_interp = RegularGridInterpolator((x, y), u_mid, method='linear', bounds_error=False, fill_value=None)
v_interp = RegularGridInterpolator((x, y), v_mid, method='linear', bounds_error=False, fill_value=None)

# Number of particles to track - change this to any number you want
num_particles = 100

# Initialize random particles
np.random.seed(42)  # For reproducibility
initial_positions = []
for _ in range(num_particles):
    x0 = np.random.uniform(0, Lx)
    y0 = np.random.uniform(0, Ly)
    initial_positions.append((x0, y0))

# Initialize trajectories list
trajectories = [[(x0, y0)] for x0, y0 in initial_positions]

# Particle integration loop for each particle
for i in range(num_particles):
    x0, y0 = initial_positions[i]
    for _ in range(total_step):
        x_map = x0 % Lx
        y_map = y0 % Ly
        pos = np.array([x_map, y_map])
        u_val = u_interp(pos)[0]
        v_val = v_interp(pos)[0]
        x1 = x0 + u_val * dt
        y1 = y0 + v_val * dt
        trajectories[i].append((x1, y1))
        x0, y0 = x1, y1

# Convert to arrays
trajectories = [np.array(traj) for traj in trajectories]
time_array = np.linspace(0, T, total_step)
msd = np.zeros(total_step) 
for i in range(num_particles):
    traj = trajectories[i]
    x0, y0 = traj[0]
    for step in range(total_step):
        x, y = traj[step]
        dx = x - x0
        dy = y - y0
        msd[step] += dx**2 + dy**2
msd /= num_particles

# Plot
plt.figure(figsize=(12, 10))
# Create a colormap for the trajectories
cmap = cm.get_cmap('rainbow', num_particles)
# Plot each trajectory with a unique color
for i, trajectory in enumerate(trajectories):
    x_traj, y_traj = trajectory[:, 0], trajectory[:, 1]
    color = cmap(i / num_particles)
    # Plot the trajectory line
    plt.plot(x_traj, y_traj, '-', color=color, linewidth=1.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Trajectories of {num_particles} Particles in 2D Turbulent Flow')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
plt.loglog(time_array, msd, '-', linewidth=2, markersize=4)
plt.xlabel('Time (t)')
plt.ylabel(r'Mean Square Displacement $\langle\Delta x^2(t)\rangle$')
plt.title('Mean Square Displacement vs Time')
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend()
plt.tight_layout()
plt.show()