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
class SearchScene(Scene):
    def setup(self):
        # TODO: automate the low_q from config
        self.low_q = True
        self.grid_lag = 0.01
        self.x_range = [-6, 6, 2]
        self.x_steps = 10
        self.y_range = [-6, 6, 2]
        self.y_steps = 10

        self.make_axes()
        self.make_elevation_map()
        self.make_grid()

    def make_axes(self):
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
        self.create_axes_animation = AnimationGroup(Create(self.axes),  FadeIn(self.labels))

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

    def make_grid(self):
        # TODO: move to setup
        # TODO: using the same buffer for x and y looks weird since screen isn't a square
        buff = 0.3
        x_positions = np.linspace(-config.frame_width/2 + buff, config.frame_width/2 -buff, self.x_steps)
        y_positions = np.linspace(-config.frame_height/2 + buff, config.frame_height/2 - buff, self.y_steps)
        self.grid_coords = []
        for y_p in y_positions:
            for x_p in x_positions:
                coords = self.axes.point_to_coords([x_p, y_p,0])
                self.grid_coords.append(coords)
        self.grid_points = VGroup(*[ Dot(self.axes.coords_to_point( *coord ), stroke_width=1., radius=0.1) for coord in self.grid_coords])
        self.grid_values = [f(*coord) for coord in self.grid_coords]
        self.grid_colors = [color_map(val, np_mode=False) for val in self.grid_values]

    def sample_random_points(self, n):
        return list(np.random.randint(0, len(self.grid_points), [n]))

    def get_neighbors(self, i):
        left = i -1
        right = i +1
        up = i + self.x_steps
        down = i - self.x_steps

        neighbors = []
        if i%self.y_steps != 0: neighbors.append(left)
        if i%self.y_steps!= self.x_steps - 1: neighbors.append(right)
        if up < self.x_steps * self.y_steps: neighbors.append(up)
        if down >= 0: neighbors.append(down)
        return neighbors

    def best_point(self, point_indices=None):
        if  point_indices is None: point_indices = list(range(len(self.grid_points)))
        point_values = [self.grid_values[i] for i in point_indices]
        best_point = np.argmin(point_values)
        return best_point

    def fadeout_points(self, point_indices=None):
        if  point_indices is None: point_indices = list(range(len(self.grid_points)))
        fadeout_points_animation = [FadeOut(self.grid_points[i]) for i in point_indices]
        anim = AnimationGroup(*reversed(fadeout_points_animation), lag_ratio = self.grid_lag)
        self.play(anim)

    def fadein_points(self, point_indices=None, color_by_elevation=False, run_time=1):
        if  point_indices is None: point_indices = list(range(len(self.grid_points)))
        if color_by_elevation:
            fadein_points_animation = [FadeIn(self.grid_points[i].set_fill(self.grid_colors[i])) for i in point_indices]
        else:
            fadein_points_animation = [FadeIn(self.grid_points[i].set_fill(WHITE)) for i in point_indices]
        anim = AnimationGroup(*reversed(fadein_points_animation), run_time=run_time, lag_ratio = self.grid_lag)
        self.play(anim)

    def color_points(self, point_indices=None, color_by_elevation=True, dark=False, run_time=1):
        opacity = 0.3 if dark else 1.
        if  point_indices is None: point_indices = list(range(len(self.grid_points)))
        if color_by_elevation:
            fadein_points_animation = [self.grid_points[i].animate.set_fill(self.grid_colors[i]).set_opacity(opacity) for i in point_indices]
        else:
            fadein_points_animation = [self.grid_points[i].animate.set_fill(WHITE).set_opacity(opacity) for i in point_indices]
        anim = AnimationGroup(*reversed(fadein_points_animation), run_time=run_time, lag_ratio = self.grid_lag)
        self.play(anim)

    def darken_points_except_best(self, point_indices=None):
        if  point_indices is None: point_indices = list(range(len(self.grid_points)))
        darken_points_animation = [self.grid_points[i].animate.set_opacity(0.3) for i in point_indices]
        best_point = self.best_point(point_indices)
        del darken_points_animation[best_point]
        darken_points_animation.append(self.grid_points[point_indices[best_point]].animate.set_opacity(1))
        self.play(AnimationGroup(*darken_points_animation))
        return best_point

    def get_arrows(self, origin, targets):
        arrows = []
        for target in targets:
            o = self.axes.c2p(*self.grid_coords[origin])
            t = self.axes.c2p(*self.grid_coords[target])
            arrow = Arrow(start=o, end=t, buff=0.2, color=GREY)
            arrows.append(arrow)
        return arrows


class BruteForce(SearchScene):
    def construct(self):
        self.play(self.create_axes_animation)
        # Full screen color map
        self.play(FadeIn(self.elev_img))

        # Display grid of grey dots
        self.fadein_points(color_by_elevation=False)
        self.wait()
        self.play(FadeOut(self.elev_img))
        # Perform the measurement 1 by 1. The dots pick up the color.
        self.color_points(color_by_elevation=True)
        # Hightlight the loest dot in the screen
        self.darken_points_except_best()
        self.wait(2)
        self.fadeout_points()

class LocalSearch(SearchScene):
    def construct(self):
        self.play(self.create_axes_animation)

        # Display grid of grey dots
        # Perform the measurement 1 by 1. The dots pick up the color.
        points = self.sample_random_points(3)
        self.fadein_points(points, color_by_elevation=True)
        self.wait(1)

        best_point = None
        while True:
            best = self.darken_points_except_best(points)
            if points[best] == best_point: break
            best_point = points[best]
            neighbors = self.get_neighbors(best_point)
            arrows = self.get_arrows(best_point, neighbors)
            # If a neighbor is already in the dataset ignore it
            neighbors = [n for n in neighbors if n not in points]
            self.play(FadeIn(*arrows, run_time=0.3))
            self.fadein_points(neighbors, run_time=0.5)
            self.play(FadeOut(*arrows, run_time=0.3))
            self.color_points(neighbors, dark=False, run_time=0.5)
            points.extend(neighbors)

        best = self.darken_points_except_best(points)
        best_point = points[best]


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


class ExponentialGrowth(Scene):
    def construct(self):
        axes = self.get_zoom_axes()

    def get_zoom_axes(self):
        axes = Axes(x_range=[0,30, 2],
                    y_range=[0, 1000000, 100000])
        self.add(axes)

        print(list(axes.y_axis.ticks))

        def get_y_labels(axes, y_values):
            labels = VGroup()
            for y in y_values:
                try:
                    tick = axes.y_axis.ticks[y]
                    if y < 1000:
                        label = Tex(f"{y}")
                    elif y < 1000000:
                        label = Tex(f"{y/1000}k")
                    else:
                        label = Tex(f"{y/1000000}m")
                    always(label.next_to, tick)
                    labels.add(label)
                except IndexError:
                    pass
            return labels

        main_labels = get_y_labels(axes, range(3, 30, 5))
        axes.y_labels = main_labels
        axes.small_y_labels = get_y_labels(axes, range(1,6))

        tiny_ticks = VGroup()
        tiny_labels = VGroup()
        for y in range(200, 1000, 200):
            tick = axes.y_axis.ticks[0].copy()
            point = axes.c2p(0, y)
            tick.move_to(point)
            label = Integer(y)
            label.set_height(0.25)
            always(label.next_to, tick, LEFT, SMALL_BUFF)
            tiny_labels.add(label)
            tiny_ticks.add(tick)

        axes.tiny_y_labels = tiny_labels
        axes.tiny_ticks = tiny_ticks
        axes.add(main_labels)
        origin = axes.c2p(0, 0)
        self.add(Dot(axes.c2p(10,10000)))
        axes.y_axis.stretch(25, 1, about_point=origin)
        self.wait()
        self.add(
                    axes.tiny_y_labels, axes.tiny_ticks,
                )
        self.play(
                    axes.y_axis.animate.stretch(0.2, 1, about_point= origin),
                    FadeOut(axes.tiny_y_labels),
                    FadeOut(axes.tiny_ticks),
                    run_time=2,
                )
        return axes



class TheGradient(Scene):
    def construct(self):
        pass
