import numpy as np
from compare_params_of_representations import auto_regr_repr
from other_classifiers import load_time_series
import pywt

from manim import *
import numpy as np


# Define the output file path
OUTPUT_PATH = r"D:\1.Thesis\1.Data_Analysis\media"

class SlidingDotsAnimation(Scene):
    def construct(self):
        # Set white background
        self.camera.background_color = WHITE

        # Define axes
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[-1.5, 1.5, 0.5],
            axis_config={"color": BLACK}  # Change axis color to black for visibility
        ).add_coordinates()

        # Define function for time series (sine wave)
        def func(x):
            return np.sin(x) * np.exp(-0.1 * x)

        # Create the curve
        curve = axes.plot(func, x_range=[0, 10], color=BLUE, stroke_width=3)

        # Number of dots
        num_dots = 10  

        # Create dots with very small spacing
        dots = []
        for i in range(num_dots):
            x_start = i * (10 / (num_dots * 5))  # Very small spacing for "worm" effect
            color = GREEN if i == 0 else RED  # First dot is green, others are red
            dot = Dot(color=color, radius=0.07)  
            dot.move_to(axes.c2p(x_start, func(x_start)))  
            dots.append(dot)

        # Add axes, curve, and dots to the scene
        self.play(Create(axes), Create(curve), run_time=2)
        self.add(*dots)

        # Animate dots moving along the curve for 10 seconds
        animations = [
            MoveAlongPath(dot, curve, rate_func=linear).set_run_time(10) for dot in dots
        ]

        # Use a very small `lag_ratio` so they stick close together
        self.play(AnimationGroup(*animations, lag_ratio=0.02), run_time=10)
        self.wait(1)

# Save the animation to the specified path
config.media_dir = OUTPUT_PATH





