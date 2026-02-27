import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pandas as pd
# import matplotlib.animation as animation
plt.close('all')  # Close all open figures at the start

# ---------------------------
# Simulation Parameters
# ---------------------------

g = 9.81
frequency = 250
disk_radius = 0.55 #0.25
cart_speed_y =-1.11# -2 cart now moves in Y direction
angular_speed =94.25#10.47#100 rpm 10.47 #20.94 200 rpm #  31.42 300 rpm #41.89 400 rpm #500rpm 52.36 #62.83 600 rpm   # 700 rpm 73.30 #800 rpm 83.78 #900 rpm 94.25 
n_blades = 4
disk_configs = [(-0.555, 0 ,1,1) , (0.555 ,0 ,1,-1)]#0.27 height

# Time settings
dt = 1/frequency/2
total_time = 5
n_steps = int(total_time / dt)
release_interval = int(1 /frequency/dt)

# Drag and particles
drag_coefficient = 0.07#1.95e-6#0.45  0.07
particle_mass = 0.01
positions = []
particles = []

class Particle:
    def __init__(self, position, velocity):
        self.pos = np.array(position, dtype=float)
        self.vel = np.array(velocity, dtype=float)
        self.alive = True

    def update(self):
        if not self.alive:
            return
        v = self.vel
        drag_acc = -drag_coefficient * v * np.linalg.norm(v)
        acc = np.array([0, 0, -g]) + drag_acc
        self.vel += acc * dt
        self.pos += self.vel * dt
        if self.pos[2] <= 0:
            self.pos[2] = 0
            self.alive = False
            positions.append((self.pos[0], self.pos[1]))

# ---------------------------
# Camera Setup
# ---------------------------
cam_z = 3
fov = 180# 120degrees
frame_width =60##60 # 23 meters across (approximate field of view at Z=3)
frame_height =60#60.0#60#23
frames = []

fig, ax = plt.subplots(figsize=(10.24, 10.24), dpi=50)
sc = ax.scatter([], [], s=10, color='white')
ax.set_facecolor('black')

ax.set_xlim(-frame_width/2, frame_width/2)
ax.set_ylim(-frame_height/2, frame_height/2)
#ax.set_title("Camera View from Above (Fixed on Cart)")
ax.set_xticks([])
ax.set_yticks([])

# REMOVE padding around plot
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

from tqdm import tqdm

# ---------------------------
# Main Simulation Loop
# ---------------------------


for step in tqdm(range(n_steps), desc="Simulating"):
    t = step * dt
    cart_y = cart_speed_y * t  # cart moves in Y direction
    if step % release_interval == 0:
        for disk_x, disk_y, disk_z, direction in disk_configs:
            for blade_index in range(n_blades):
            # Blade angular position
                angle = (angular_speed * t + blade_index * (2 * np.pi / n_blades)) % (2 * np.pi)
                adjusted_angle = direction * angle
                normalized_angle = (adjusted_angle + np.pi) % (2 * np.pi) - np.pi

                # Limit ejection to 180°: only throw forward relative to disc rotation
                if not (-np.pi / 2 <= normalized_angle <= np.pi / 2):
                    continue

                # Direction from disc center to blade tip
                #dir_vector = np.array([np.cos(normalized_angle), np.sin(normalized_angle)])

                # Particle position at blade tip
                #pos_offset = disk_radius * dir_vector
                position = np.array([
                    disk_x,# + pos_offset[0],
                    cart_y,# + disk_y + pos_offset[1],
                    disk_z
                ])

                # Tangential velocity due to rotation
                tangential = angular_speed * disk_radius * np.array([
                    -np.sin(normalized_angle), np.cos(normalized_angle), 0
                ])

                # Total velocity = tangential + cart motion
                velocity = np.array([0, cart_speed_y, 0]) + tangential

                # Add the new particle
                particles.append(Particle(position, velocity))

    for p in particles:
        p.update()

    # Capture frame from camera
    current_positions = np.array([p.pos for p in particles if p.alive])
    if len(current_positions) > 0:
        in_view = current_positions[(np.abs(current_positions[:,0]) <= frame_width/2) & 
                                    (np.abs(current_positions[:,1] - cart_y) <= frame_height/2)]
        
        sc.set_offsets(np.c_[in_view[:,0], in_view[:,1] - cart_y])
        plt.pause(0.001)  # Real-time display during simulation
        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        frames.append(image.copy())
plt.close(fig)


# ---------------------------
# Plot Scatter of Landing Points
# ---------------------------
positions = np.array(positions)
# ---------------------------
# Distribution analysis in the sampling rectangle (fixed strip method)
plt.figure(figsize=(10, 5))
plt.scatter(positions[:, 0], positions[:, 1], alpha=0.7, s=10, color='green')

# Draw horizontal sampling rectangle centered at Y=5, rotated to be perpendicular to Y axis
rect_center_x = 0
rect_center_y=0#-2.5#6
rect_x_width = 15
rect_y_width = 0.5
rect_x_min = rect_center_x - rect_x_width / 2
rect_x_max = rect_center_x + rect_x_width / 2
rect_y_min = rect_center_y - rect_y_width / 2
rect_y_max = rect_center_y + rect_y_width / 2

plt.plot([rect_x_min, rect_x_max, rect_x_max, rect_x_min, rect_x_min],
         [rect_y_min, rect_y_min, rect_y_max, rect_y_max, rect_y_min],
         color='red', linestyle='--', linewidth=2)

plt.title("Landing Points of Particles with Sampling Rectangle")
plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.grid(True)
plt.axis("equal")

plt.savefig("/home/arezou/UBONTO/result.png") 
plt.show()
# ---------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.mixture import GaussianMixture

filtered_x = [x for x, y in positions if rect_x_min <= x <= rect_x_max and rect_y_min <= y <= rect_y_max]
plt.figure(figsize=(8, 4))

# Histogram and capture bin values
counts, bin_edges, _ = plt.hist(
    filtered_x, 
    bins=50, #50, 
    color='blue', 
    edgecolor='black', 
    alpha=0.6, 
    range=(positions[:,0].min(), positions[:,0].max()),
    label='Histogram'
)

# Compute bin centers for polynomial fitting
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
# Line connecting bar tops
plt.plot(bin_centers, counts, marker='o', linestyle='-', color='darkorange', linewidth=2, label='Bar Top Curve')

plt.title("Histogram of X Positions (Particles Landed in Rectangle at Y=5)")
plt.xlabel("X position (m)")
plt.ylabel("Count")
plt.legend()
plt.grid(True)

# Save and show plot
plot_path = "/home/arezou/UBONTO/hist_with_poly_fit.png"
plt.savefig(plot_path)
plt.show()

# Print bin edges and counts
for i in range(len(counts)):
    print(f"Bin {i+1}: Range = ({bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}), Count = {int(counts[i])}")

# Prepare data for CSV
bin_ranges = [f"({bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})" for i in range(len(counts))]
data = {
    "Bin Index": list(range(1, len(counts)+1)),
    "Range (m)": bin_ranges,
    "Particle Count": counts.astype(int)
}

# Create DataFrame and save to CSV
df_bins = pd.DataFrame(data)
csv_path = "/home/arezou/UBONTO/histogram_bins.csv"
df_bins.to_csv(csv_path, index=False)

print(f"✅ Bin data saved to: {csv_path}")
print(f"✅ Plot with polynomial fit saved to: {plot_path}")

# ---------------------------
# Save video as MP4 only (requires imageio-ffmpeg)
# import imageio_ffmpeg
import imageio

output_path_mp4 = '/home/arezou/UBONTO/camera_view.mp4'
writer = imageio.get_writer(output_path_mp4, fps=int(1/(dt*10)), format='ffmpeg')
for frame in frames:
    writer.append_data(frame)
writer.close()
print(f"✅ MP4 saved to: {output_path_mp4}")

# Optionally show the first frame as confirmation
plt.figure(figsize=(6, 6))
plt.imshow(frames[0])
plt.axis('off')
plt.title("First Frame Preview")
plt.show()

# Graceful exit to prevent async task warnings
import time, sys
time.sleep(0.5)
sys.exit(0)




