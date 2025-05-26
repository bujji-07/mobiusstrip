import numpy as np
from scipy.integrate import simps
from scipy.spatial import distance
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class MobiusStrip:
    def __init__(self, R=1.0, w=0.2, n=100):
        self.R = R  # Radius
        self.w = w  # Width of the strip
        self.n = n  # Resolution (number of points along u and v)
        self.u = np.linspace(0, 2 * np.pi, n)
        self.v = np.linspace(-w / 2, w / 2, n)
        self.U, self.V = np.meshgrid(self.u, self.v)
        self.X, self.Y, self.Z = self.compute_coordinates()

    def compute_coordinates(self):
        """Compute the 3D coordinates of the Mobius strip."""
        u, v = self.U, self.V
        x = (self.R + v * np.cos(u / 2)) * np.cos(u)
        y = (self.R + v * np.cos(u / 2)) * np.sin(u)
        z = v * np.sin(u / 2)
        return x, y, z

    def compute_surface_area(self):
        """Approximate the surface area using numerical integration."""
        dx_du = np.gradient(self.X, axis=1)
        dx_dv = np.gradient(self.X, axis=0)
        dy_du = np.gradient(self.Y, axis=1)
        dy_dv = np.gradient(self.Y, axis=0)
        dz_du = np.gradient(self.Z, axis=1)
        dz_dv = np.gradient(self.Z, axis=0)

        cross_x = dy_du * dz_dv - dz_du * dy_dv
        cross_y = dz_du * dx_dv - dx_du * dz_dv
        cross_z = dx_du * dy_dv - dy_du * dx_dv

        dA = np.sqrt(cross_x**2 + cross_y**2 + cross_z**2)
        surface_area = simps(simps(dA, self.v), self.u)
        return surface_area

    def compute_edge_length(self):
        """Approximate the edge length by summing distances along the boundary (v = ±w/2)."""
        edge1 = np.array([
            (self.X[0, i], self.Y[0, i], self.Z[0, i]) for i in range(self.n)
        ])
        edge2 = np.array([
            (self.X[-1, i], self.Y[-1, i], self.Z[-1, i]) for i in range(self.n)
        ])
        length1 = np.sum(np.linalg.norm(np.diff(edge1, axis=0), axis=1))
        length2 = np.sum(np.linalg.norm(np.diff(edge2, axis=0), axis=1))
        return length1 + length2

    def plot(self):
        """Render the Mobius strip in 3D."""
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.X, self.Y, self.Z, cmap='viridis', edgecolor='k', alpha=0.8)
        ax.set_title("Mobius Strip")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    mobius = MobiusStrip(R=1.0, w=0.2, n=200)
    mobius.plot()

    area = mobius.compute_surface_area()
    edge_len = mobius.compute_edge_length()

    print(f"Surface Area ≈ {area:.4f}")
    print(f"Edge Length ≈ {edge_len:.4f}")
