from manim import *
from colour import Color


def color_map(val, np_mode=True):
    val = val/3 + 20
    # Above 200 the color starts to look redish, looping back to value 0
    if val >= 200 or val <= 0: print('Warning: value out of bounds for coloring ', val)
    hsl_color =  np.float32([val/255,0.5,0.5])
    rgb_color = np.float32(Color(hsl=hsl_color).rgb) * 255
#    rgb_color = rgb_color.astype(np.uint8)

    if np_mode == False:
        rgb_color = Color(rgb=rgb_color/255)
    return rgb_color

def poly(x,y,z=0):
    return  4*x**2 + y**2 -2*x*y - x + y

f = poly
#f_text = 'f(x,y) = x^2 + y^2'
f_text = 'f(x,y) = x^2 - y^2 -3xy - x'


# Potentially a Scene introducing how to read an elevation map

# the simplest version
class BruteForce(Scene):
    def construct(self, low_q=True):
        self.low_q = low_q
        self.grid_lag = 0.01
        self.x_range = [-6, 6, 2]
        self.y_range = [-6, 6, 2]
        print('top right', f(6,6))
        print('top left', f(-6,6))
        print('bottom right', f(6,-6))
        print('bottom left', f(-6,-6))

        # Full screen color map
        self.draw_axes()
        self.make_elevation_map()
        self.play(FadeIn(self.elev_img))

        # Display grid of grey dots
        self.make_gird()
        self.play(self.fade_in_grid_animation)
        self.wait()
        self.play(FadeOut(self.elev_img))
        # Perform the measurement 1 by 1. The dots pick up the color.
        self.play(self.color_points_by_elev_animation)
        # Hightlight the loest dot in the screen
        self.highlight_best_point()
        self.wait(2)


    def draw_axes(self):
        # TODO: figure out why length and unit_size aren't working. Make axes squared
        self.axes = Axes( x_range = self.x_range,
                          y_range = self.y_range,
                          tips = False,
                          axis_config={'include_numbers': False,
                                        'include_tip': False,
                                        #'unit_size': 0.5},
                                        'length': 1}
                        )
        self.labels = self.axes.get_axis_labels(x_label="x", y_label="y")
#        self.add(self.axes)
        self.play(Create(self.axes))
        self.play(FadeIn(self.labels))

    def make_elevation_map(self):
        # Make the image full screen
        img_width = config.frame_width
        img_height = config.frame_height
        res_downsample = 10 if self.low_q else 1
        img_pixel_width = config.pixel_width//res_downsample
        img_pixel_height = config.pixel_height//res_downsample

        elev_np = np.empty([img_pixel_height,img_pixel_width])
        elev_color_np = np.empty([img_pixel_height,img_pixel_width,3], dtype=np.uint8)
        for i in range(img_pixel_height):
            for j in range(img_pixel_width):
                # Assumes image is centered
                point_in_screen = [-img_width/2 + j*img_width/img_pixel_width,
                                    img_height/2 - i*img_height/img_pixel_height,
                                    0]
                coords = self.axes.point_to_coords(point_in_screen)
                elev_np[i,j] = f(*coords)
                elev_color_np[i,j] = color_map( elev_np[i,j] )

        self.elev_img = ImageMobject(elev_color_np)
        self.elev_img.height = config.frame_height
        self.elev_img.set_z_index(self.axes.z_index -1)
        self.elev_img.add_updater(lambda img: img.move_to(self.axes.get_center()))

    def make_gird(self):
        axis_points = 10
        # TODO: using the same buffer for x and y looks weird since screen isn't a square
        buff = 0.3
        x_positions = np.linspace(-config.frame_width/2 + buff, config.frame_width/2 -buff, axis_points)
        y_positions = np.linspace(-config.frame_height/2 + buff, config.frame_height/2 - buff, axis_points)
        grid_coords = []
        for y_p in y_positions:
            for x_p in x_positions:
                coords = self.axes.point_to_coords([x_p, y_p,0])
                grid_coords.append(coords)
        self.grid_points = VGroup(*[ Dot(self.axes.coords_to_point( *coord ), stroke_width=1., radius=0.1) for coord in grid_coords])
        self.grid_values = [f(*coord) for coord in grid_coords]
        self.grid_colors = [color_map(val, np_mode=False) for val in self.grid_values]

        fadein_points_animation = [FadeIn(point) for point in self.grid_points]
        self.fade_in_grid_animation = AnimationGroup(*reversed(fadein_points_animation), lag_ratio = self.grid_lag)

        color_points_by_elev_animation = [ point.animate.set_fill(color) for point, color in zip(self.grid_points, self.grid_colors)]
        self.color_points_by_elev_animation = AnimationGroup(*reversed(color_points_by_elev_animation), lag_ratio = self.grid_lag)

    def highlight_best_point(self):
        fade_out_points_animation = [point.animate.set_opacity(0.3) for point in self.grid_points]
        best_point = np.argmin(self.grid_values)
        del fade_out_points_animation[best_point]
        self.play(AnimationGroup(*fade_out_points_animation))


class TheCurseOfDimensionality(ThreeDScene):
    def construct(self):
        points_per_par = 8
        assert points_per_par%2 ==0, 'Number of poins must be even'

        # TODO: point counter on top left
        axes = ThreeDAxes(axis_config={'include_numbers': False,
                                        'include_ticks': False,
                                        'include_tip': False,
                                        'length':50.}
        )
        y_axis = axes.get_y_axis()
        z_axis = axes.get_z_axis()
        y_axis.set_opacity(0.)
        z_axis.set_opacity(0.)
        self.play(Create(axes))

        buff = 0.3

        # Show line with points - Counter of evaluations
        step_size = 1
        points_line = []
        for i in range(points_per_par//2):
            right_dot = Dot(axes.coords_to_point(*[step_size/2 + step_size * i, 0, 0]), fill_opacity=0.5)
            points_line.append(right_dot)
            left_dot = Dot(axes.coords_to_point(*[-step_size/2 -step_size * i, 0, 0]), fill_opacity=0.5)
            points_line.append(left_dot)
        points_line = VGroup(*points_line)

        self.play(FadeIn(points_line, fade_in=0.6))
        self.wait()
        self.play(y_axis.animate.set_opacity(1.))

        # Show grid with 100X100 points - Counter of evaluations = 10,000
        points_plane = []
        for i in range(points_per_par//2):
            if i ==0:
                self.play(points_line.animate.shift(UP*step_size/2))
            else:
                up_line = points_line.copy().shift( UP*(step_size*i) )
                points_plane.append(up_line)
            down_line = points_line.copy().shift(UP*(-step_size -step_size*i) )
            points_plane.append(down_line)
        self.play(FadeIn(VGroup(*points_plane), lag_ratio=0.1))
        points_plane.append(points_line)
        points_plane = VGroup(*points_plane)

        self.wait()
        self.play(z_axis.animate.set_opacity(1.))
        self.move_camera(phi=75 * DEGREES, theta=-15 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.1)
        self.wait()

        # Show cube with 100X100X100 points - Counter of evaluations = 1,000,000

        points_cube = []
        for i in range(points_per_par//2):
            if i ==0:
                self.play(points_plane.animate.shift(IN*step_size/2))
            else:
                up_plane = points_plane.copy().shift(IN*( step_size*i) )
                points_cube.append(up_plane)
            down_plane= points_plane.copy().shift(IN*(- step_size -step_size*i) )
            points_cube.append(down_plane)
        points_cube = VGroup(*points_cube)

        self.play(FadeIn(points_cube, lag_ratio=0.3))
        self.wait(3)


class ExponentialGrouth(Scene):
    def construct(self):
        ax = Axes(
            x_range=[0, 10], y_range=[0, 100, 10], axis_config={"include_tip": False}
        )


        # Show the exponential curve of the number of evaluations we need for each dimension. Highlight when we exceed our budget.
        # Modify the resolution, see how the curve changes. Use numbers from script
        # Zoom out to see when we exceed the budget

class LocalSearch(Scene):
    def construct(self):
        # Mostly empty grid but with a few evaluations
        # Hightlight the best current point
        # highlight each neighbor drawing an arow and meause (have the point pick up color)
        # Hightlight the new best point
        # Iterate the procedue. Slow a couple of times.  Then faster until convergence.

        # Change the elevation map to a new equation with local minima. Show how it fails to find the best option.

        # Sow a zig zag path, or a spiral path that ends up going through all the points in the grid.

        # Still a problem.
        still_a_problem = FadeIn('There is still a problem')
        self.play(still_a_problem)
        self.wait(2)
        self.play(FadeOut(still_a_problem))

        # Show number of neighbors in 1D = 2. A point with 2 arrows
        # Show number of neighbors in 2D = 4. A point with 4 arrows
        # Show number of neighbors in 3D = 6. A point with 6 arrows
        # Show general formula

class TheGradient(Scene):
    def construct(self):
        pass
