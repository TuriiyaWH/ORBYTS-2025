import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.patches import Circle, Arc

import os
import PIL
from PIL import Image, ImageFilter
from matplotlib import cm
from matplotlib.colors import Normalize

import matplotlib as mpl
from tqdm import tqdm

from sklearn.tree import DecisionTreeClassifier


def get_rgb_color(value, clmap):
    rgb = clmap(value)[:3]  # Get RGB values from the colormap
    rgb = [int(x * 255) for x in rgb]  # Scale RGB values to 0-255 range
    color_code = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"  # Convert RGB values to hexadecimal string
    return color_code


def test_colormap(cmap):

    # Create a figure and an axis for the colorbar
    fig, ax = plt.subplots(figsize=(8, 1))
    fig.subplots_adjust(bottom=0.5)

    # Create a horizontal colorbar
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')

    # Set colorbar ticks and label
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cb.set_label("Wavelength (nm)\n[colour]")
    # cb.set_ticks([0.18,0.4,0.62,0.84])
    # cb.set_ticklabels([400,500,600,700])
    # cb.ax.set_xlim(0.15,0.85)

    # Show the plot
    plt.show()


def default_colourmap(plot=True):
    clmap = plt.get_cmap("nipy_spectral")
    if plot:
        test_colormap(clmap)

    return clmap


def generate_planet(ax, atmosphere_type, clmap=default_colourmap(plot=False)):
    """
    Draws a cartoon planet with multiple rings that appear to go behind and in front of the planet.
    """
    # Set colors based on atmosphere type
    if atmosphere_type == "h2o":
        color = get_rgb_color(np.random.randint(20,38)/100, clmap)
        ring_color = get_rgb_color(0.40, clmap)
    elif atmosphere_type == "co2":
        color = get_rgb_color(np.random.randint(75,85)/100, clmap)
        ring_color = get_rgb_color(0.70, clmap)
    elif atmosphere_type == "ch4":
        color = get_rgb_color(np.random.randint(45,55)/100, clmap)
        ring_color = get_rgb_color(0.40, clmap)

    # Planet radius
    planet_radius = random.uniform(0.22, 0.4)
    atmosphere_depth = random.uniform(0.01, 0.1)

    # Draw multiple rings if applicable
    rings_bool = random.choice([True, False])
    if rings_bool:
        n_rings = random.randint(2, 5)  # Randomly select n rings
        ring_angle = random.uniform(-30, 30)  # Common angle for all rings
        ring_factor = random.uniform(0.1, 0.65)  # Factor to scale the ring size
        ring_top_bottom = random.choice([True, False])

        ring_diameter_x = [0] * n_rings
        ring_width = [0] * n_rings
        ring_thickness = [0] * n_rings

        if ring_top_bottom:
            thet1 = 180
            thet2 = 360
        else:
            thet1 = 0
            thet2 = 180

        for i in range(n_rings):
            # Randomly vary the ring properties
            ring_diameter_x[i] = planet_radius * random.uniform(2.5, 3.5)  # Horizontal diameter
            ring_width[i] = ring_diameter_x[i]*ring_factor  #planet_radius * random.uniform(1.5, 2.0)       # Vertical diameter
            ring_thickness[i] = random.randint(1, 5)  # Thickness of the ring


            # Draw the back half of the ring
            ring_back = Arc(
                (0.5, 0.5),                # Center of the ellipse
                width=ring_diameter_x[i],      # Horizontal diameter
                height=ring_width[i],          # Vertical diameter
                angle=ring_angle,           # Tilt angle
                theta1=thet1, theta2=thet2,     # Back half of the ring (180°-360°)
                color=ring_color,
                lw=ring_thickness[i]
            )
            ax.add_artist(ring_back)

    # Draw the planet itself
    planet_circle = Circle((0.5, 0.5), planet_radius- atmosphere_depth, color='black', ec="none", lw=1)
    ax.add_artist(planet_circle)

    # Draw random spots (like clouds or storms)
    for _ in range(random.randint(1, 3)):
        spot_x = random.uniform(0.5-planet_radius, 0.5+planet_radius)
        spot_y = random.uniform(0.35, 0.65)
        spot_size = random.uniform(0.01, planet_radius*0.2)
        spot = Circle((spot_x, spot_y), spot_size, color="#001010", alpha=1, ec='none')
        ax.add_artist(spot)

    planet_surface = Circle((0.5, 0.5), planet_radius, color=color, ec="none", lw=1, alpha=0.2)
    ax.add_artist(planet_surface)

    # Draw the front half of each ring to complete the look
    if rings_bool:
        if ring_top_bottom:
            thet1 = 0
            thet2 = 180
        else:
            thet1 = 180
            thet2 = 360
        for i in range(n_rings):

            ring_front = Arc(
                (0.5, 0.5),                # Center of the ellipse
                width=ring_diameter_x[i],      # Horizontal diameter
                height=ring_width[i],          # Vertical diameter
                angle=ring_angle,           # Tilt angle
                theta1=thet1, theta2=thet2,       # Front half of the ring (0°-180°)
                color=ring_color,
                lw=ring_thickness[i]
            )
            ax.add_artist(ring_front)

    # Remove axes, set equal aspect, and set black background
    ax.set_facecolor("white")
    ax.set_aspect("equal")
    ax.axis("off")

def generate_planet_grid(show_chemistry=False):
    """
    Generate a 5x5 grid of planets with different atmospheres.
    """
    fig, axs = plt.subplots(5, 5, figsize=(10, 10), facecolor="white")
    atmosphere_types = ['co2', 'h2o', 'ch4']

    for i in range(5):
        for j in range(5):
            atmosphere_type = random.choice(atmosphere_types)
            generate_planet(axs[i, j], atmosphere_type)
            if show_chemistry:
                axs[i, j].set_title(atmosphere_type, fontsize=8, color="black")

    plt.tight_layout()
    plt.show()

class generate_single_planet():
    def __init__(self, atmosphere_type=None, show_chemistry=False, show=True):
        """
        Generate a single planet with the specified atmosphere type.
        """

        if atmosphere_type is None:
            atmosphere_type = random.choice(['co2', 'h2o', 'ch4'])
        fig, ax = plt.subplots(figsize=(5, 5), facecolor="none")
        generate_planet(ax, atmosphere_type)
        if show_chemistry:
            ax.set_title(atmosphere_type, fontsize=8, color="black", backgroundcolor="white")
        self.fig = fig
        self.atmosphere_type = atmosphere_type

        if show:
            plt.show()
        else:
            plt.close()

    def save(self, filename):

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        self.fig.savefig(filename, dpi=80)



def generate_star(ax, clmap=default_colourmap(plot=False), radius_range=(0.2, 0.5)):
    """
    Draws a large star with variable size and sunspots behind the planet
    """
    # Randomize star size and color
    star_radius = random.uniform(*radius_range)
    star_color = get_rgb_color(np.random.randint(70,85)/100, clmap)
    star_center = ((np.random.rand(2)-0.5)*radius_range[1]*0.5)+0.5  # Randomize star position
    star_circle = Circle((star_center[0], star_center[1]), star_radius, color=star_color, ec="none", lw=2)
    ax.add_artist(star_circle)

    # Add sunspots on the star
    for _ in range(random.randint(3, 7)):
        sunspot_x = random.uniform(star_center[0] - star_radius, star_center[0] + star_radius)
        sunspot_y = random.uniform(star_center[1] - star_radius, star_center[1] + star_radius)
        sunspot_size = random.uniform(0.02, star_radius * 0.2)

        # Only add the sunspot if it's within the star's bounds
        if np.sqrt((sunspot_x - star_center[0])**2 + (sunspot_y - star_center[1])**2) < star_radius:
            for i in range(20):
                sunspot = Circle((sunspot_x, sunspot_y), sunspot_size*((20-i)/20), color="black", alpha=0.1, ec='none')
                ax.add_artist(sunspot)

    ax.set_facecolor("black")
    ax.set_aspect("equal")
    ax.axis("off")



def generate_star_grid(radius_range=(0.2, 0.3)):
    """
    Generate a 5x5 grid of planets with different atmospheres.
    """
    fig, axs = plt.subplots(5, 5, figsize=(10, 10), facecolor="black")

    for i in range(5):
        for j in range(5):
            generate_star(axs[i, j], radius_range=radius_range)


    plt.tight_layout()
    plt.show()

class generate_single_star():
    def __init__(self, radius_range=(0.2, 0.3), show=True):
        """
        Generate a single star.
        """
        fig, ax = plt.subplots(figsize=(5, 5), facecolor="black")
        generate_star(ax, radius_range=radius_range)
        self.fig = fig

        if show:
            plt.show()
        else:
            plt.close()

    def save(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.fig.savefig(filename, dpi=80)


class generate_combined_image():
    def __init__(self, planet_file, star_file, show=True):

        # Open the planet and star images
        planet_image = Image.open(planet_file)
        star_image = Image.open(star_file)

        # scale the star image to match the planet image
        star_image = star_image.resize(planet_image.size)

        # Overlay the star image on top of the planet image
        overlay_image = Image.alpha_composite(star_image.convert("RGBA"), planet_image.convert("RGBA"))

        fig, ax = plt.subplots(figsize=(5, 5), facecolor="black")
        self.fig = fig
        ax.imshow(overlay_image)
        ax.set_xlim(50, 350)
        ax.set_ylim(350, 50)
        ax.axis("off")
        if show:
            plt.show()
        else:
            plt.close()

    def save(self, filename):

        # Save the figure in the corresponding folder
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.fig.savefig(filename, dpi=80, bbox_inches='tight', facecolor="black")

def generate_colorbar():
    """
    Creates a color map representing the visible spectrum from red to violet.
    """
    # Use a colormap that approximates the visible light spectrum
    spectrum_colormap = cm.get_cmap(default_colourmap(plot=False), 256)
    # print('spectrum_colormap shape')
    # print(spectrum_colormap(np.linspace(0, 1, 256)).shape)
    return spectrum_colormap

def calculate_normalised_wavelength(rgb, colorbar):
    """
    Maps an RGB value to a position on a spectrum colorbar.
    Returns a 'wavelength' value between 0 and 1.
    """
    # Convert the RGB color to a normalized array [0, 1]
    rgb_normalized = np.array(rgb) / 255.0

    # Find the closest color on the colorbar to the RGB color
    color_diffs = np.linalg.norm(colorbar[:, :4] - rgb_normalized, axis=1)
    # print(f"colour diffs shape: {color_diffs.shape}")
    # print(f"colour diffs max: {color_diffs.max()}")
    min_index = np.argmin(color_diffs)
    # print(f"min index: {min_index}")

    # Normalize the index to get a value between 0 and 1
    normalized_wavelength = min_index / (len(colorbar) - 1)
    return normalized_wavelength

class generate_spectrum():
    def __init__(self, image_path, y_axis_scale='linear', show=True, bins=256):
        """
        Reads in an image file, calculates normalized 'wavelengths' for each pixel, and generates a histogram.
        """
        # Load the image and convert it to an array
        image = Image.open(image_path)
        image_array = np.array(image)

        # print(f'image shape:{image_array.shape}')

        # Flatten the 2D image array into a 1D list
        pixels = image_array.reshape(-1, 4)

        # Generate the colorbar for the visible spectrum
        colorbar = generate_colorbar()(np.linspace(0, 1, 256))

        # Calculate the 'wavelength' for each pixel
        wavelengths = [0,1]
        for pixel in pixels:
            wavelength = calculate_normalised_wavelength(pixel, colorbar)
            wavelengths.append(wavelength)



        # Compute histogram using numpy
        hist, bin_edges = np.histogram(wavelengths, bins=bins, density=True)

        # Prepare the x values for the line plot (midpoints of the bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        self.intensity = hist
        self.wavelength = bin_centers

        if show:
            # Plot the histogram as a line
            fig, axs = plt.subplots(1,2,width_ratios=[4,1] , figsize=(8, 5))
            ax=axs[0]
            ax.plot(bin_centers, hist, "k-", linewidth=2)
            ax.set_title("Spectrum of Image [Colour Distribution]")
            ax.set_xticks([])
            ax.set_ylabel("[number of pixels]\nIntensity")
            ax.set_xlim(0, 1)
            if y_axis_scale == 'log':
                ax.set_yscale('log')
                ax.set_ylim(-0.01, None)
            elif y_axis_scale == 'linear':
                ax.set_ylim(-1, None)
            else:
                raise ValueError("y_axis_scale must be either 'linear' or 'log'")

            # Create the colorbar
            norm = Normalize(vmin=0, vmax=1)
            sm = cm.ScalarMappable(cmap=default_colourmap(plot=False), norm=norm)#nipy_spectral
            sm.set_array([])  # Dummy array for color mapping



            # Add colorbar below the histogram
            cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.02)
            cbar.set_label("Wavelength (nm)\n[colour]")
            cbar.set_ticks([0.18,0.4,0.62,0.84])
            cbar.set_ticklabels([400,500,600,700])

            axs[1].imshow(np.array(image)[15:-15,15:-15])
            axs[1].axis('off')

            plt.tight_layout()
            plt.show()

    def save(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # use numpy to save the data as a csv file
        np.savetxt(filename, np.array([self.wavelength, self.intensity]).T, delimiter=',', header='wavelength,intensity', comments='')

def add_colorbar():
    # Create the colorbar
    norm = Normalize(vmin=0, vmax=1)
    sm = cm.ScalarMappable(cmap=default_colourmap(plot=False), norm=norm)#nipy_spectral
    sm.set_array([])  # Dummy array for color mapping
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    # Add colorbar below the histogram
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.02)
    cbar.set_label("Wavelength (nm)\n[colour]")
    cbar.set_ticks([0.18,0.4,0.62,0.84])
    cbar.set_ticklabels([400,500,600,700])

    plt.tight_layout()