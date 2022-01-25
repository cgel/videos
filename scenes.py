from manim import *
from colour import Color
from math import sin, cos, sqrt, exp

def normalize_val(val, min, max): return 150*(val-min)/(max - min)

def color_map(val, np_mode=True):
    # Above 200 the color starts to look redish, looping back to value 0
    if val >= 200 or val < 0: print('Warning: value out of bounds for coloring ', val)
    hsl_color =  np.float32([val/255,0.5,0.5])
    rgb_color = np.float32(Color(hsl=hsl_color).rgb) * 255
#    rgb_color = rgb_color.astype(np.uint8)

    if np_mode == False:
        rgb_color = Color(rgb=rgb_color/255)
    return rgb_color
#
# def poly(x,y,z=0):
#     return  4*x**2 + y**2 -2*x*y - x + y
#
# def sym_poly(x,y,z=0):
#     return (x**4)*(y**2) + (x**2)*(y**4) - 80*(x**2)*(y**2) + 10 *x - 6*y
#

def f(x,y,z=0):
    return sin((x-4)/3) + cos( (x+y -4)/3  )**2 + (y)**2/100
def dfdx(x,y):
    return cos( (x-4)/3 )/3 -1/3* sin(2*(x+y-4)/3)
def dfdy(x,y):
    return -1/3*sin( 2*(x+y-4)/3) + y/50
def gradf(x,y, z=0):
    return [dfdx(x,y), dfdy(x,y)]
f_tex = "f(x,y)=sin\left(\\frac{x -4}{3}\\right) + cos\\left( \\frac{x+y-4}{3}\\right)^2 + \\frac{y^2}{100}"

# Potentially a Scene introducing how to read an elevation map

# the simplest version
class SearchScene(Scene):
    def setup(self):
        # TODO: automate the low_q from config
        self.low_q = True
        self.grid_lag = 0.01
        self.x_range = [-6, 6, 2]
        #self.x_range = [-2, 2, 1]
        self.x_steps = 10
        self.y_range = [-6, 6, 2]
        #self.y_range = [-2, 2, 1]
        self.y_steps = 10

        self.make_axes()
        self.make_elevation_map()
        self.make_grid()

    def make_axes(self):
        # TODO: figure out why length and unit_size aren't working. Make axes squared
        self.axes = Axes( x_range = self.x_range,
                          y_range = self.y_range,
                          tips = False,
                          axis_config={'include_numbers': True,
                                        'include_tip': False,
                                        #'unit_size': 0.5},
                                        'length': 1}
                        )
        self.labels = self.axes.get_axis_labels(x_label="x", y_label="y")
        self.create_axes_animation = AnimationGroup(Create(self.axes),  FadeIn(self.labels))

    def make_elevation_map(self):
        # TODO: fix the black gap on the right of the screen
        # Make the image full screen
        img_width = config.frame_width
        img_height = config.frame_height
        res_downsample = 5 if self.low_q else 1
        img_pixel_width = config.pixel_width//res_downsample
        img_pixel_height = config.pixel_height//res_downsample

        elev_np = np.empty([img_pixel_height,img_pixel_width])
        for i in range(img_pixel_height):
            for j in range(img_pixel_width):
                # Assumes image is centered
                point_in_screen = [-img_width/2 + j*img_width/img_pixel_width,
                                    img_height/2 - i*img_height/img_pixel_height,
                                    0]
                coords = self.axes.point_to_coords(point_in_screen)
                elev_np[i,j] = f(*coords)
        self.min_v = np.min(elev_np)
        self.max_v = np.max(elev_np)
        elev_color_np = np.empty([img_pixel_height,img_pixel_width,3], dtype=np.uint8)
        for i in range(img_pixel_height):
            for j in range(img_pixel_width):
                elev_color_np[i,j] = color_map( normalize_val(elev_np[i,j], self.min_v, self.max_v) )

        self.elev_img = ImageMobject(elev_color_np)
        self.elev_img.move_to(self.axes.get_center())
        self.elev_img.height = config.frame_height
        self.elev_img.width = config.frame_width
        self.elev_img.set_z_index(self.axes.z_index -1)

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
        self.grid_colors = [color_map( normalize_val(val, self.min_v, self.max_v), np_mode=False) for val in self.grid_values]

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

    def display_equation(self):
        self.tex_mobject = MathTex(f_tex)
        self.tex_mobject.scale(0.5)
        self.tex_mobject.move_to(UP*2 + LEFT*4)
        self.tex_rect = BackgroundRectangle(self.tex_mobject, fill_opacity=0.5, buff=MED_SMALL_BUFF)
        self.play(FadeIn(self.tex_rect), Write(self.tex_mobject))

    def remove_equation(self):
        self.play(FadeOut(self.tex_rect, self.tex_mobject))



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
        y_zooms = [ [0, 100, 10],
                    [0, 10000, 1000],
                    [0, 1000000, 100000]
                   ]
        axes = self.get_zoom_axes(y_zooms)
        self.add(axes)
        self.wait()

        for i in range(0, 30):
            dot = Dot(axes.c2p(i, exp(i)), color=BLUE)
            dot.x = i
            dot.y = exp(i)
            dot.add_updater(lambda ob: ob.move_to(axes.c2p(ob.x, ob.y)))
            self.play(FadeIn(dot), run_time = 0.1)
            self.wait(0.3)

            if i==4: self.play(axes.y_axis.zoom_out_anim(), run_time=2)
            if i==10: self.play(axes.y_axis.zoom_out_anim(), run_time=2)


    def get_zoom_axes(self, y_zooms):
        """
        y_zooms: a list of y_ranges containing min, max and steps in increasing order
        """
        x_range=[0,30, 2]
        y_range=y_zooms[-1]
        axes = Axes( x_range = x_range,
                      y_range = y_range,
                      tips = False,
                      axis_config={'include_tip': False},
                      y_axis_config={'include_ticks': False},
                      x_axis_config={'include_ticks': True,
                                    'include_numbers': True}
                    )
        axes_labels = axes.get_axis_labels(x_label="\# Dims", y_label="Evaluations")
        axes.add(axes_labels)
        origin = axes.c2p(0,0)

        axes.y_axis.tick_list = []
        def add_y_ticks(y_values):
            ticks = VGroup()
            for y in y_values[1:]:
                ticks.add(axes.y_axis.get_tick(y))
            axes.y_axis.add(ticks)
            axes.y_axis.tick_list.append(ticks)

        for zoom in y_zooms:
            assert zoom[0] == 0
            add_y_ticks(range(*zoom))

        def zoom_out_anim():
            zoom_i = axes.y_axis.zoom_i
            assert zoom_i < len(axes.y_axis.tick_list)
            stretch =y_zooms[zoom_i][2]/y_zooms[zoom_i+1][2]
            axes.y_axis.remove(axes.y_axis.tick_list[zoom_i])
            anim = AnimationGroup(axes.y_axis.animate.stretch(stretch, 1, about_point= origin),
                            FadeOut(axes.y_axis.tick_list[zoom_i])
                        )
            axes.y_axis.zoom_i += 1
            return anim
        axes.y_axis.zoom_out_anim = zoom_out_anim
        axes.y_axis.zoom_i = 0

        # Zoom to the first level
        stretch =y_zooms[-1][2]/y_zooms[0][2]
        axes.y_axis.stretch(stretch, 1, about_point= origin),
        # self.play(axes.y_axis.animate.stretch(stretch, 1, about_point= origin),run_time=2)
        return axes


class TheGradient(ZoomedScene, SearchScene):
# contributed by TheoremofBeethoven, www.youtube.com/c/TheoremofBeethoven
    def __init__(self, **kwargs):
        ZoomedScene.__init__(
            self,
            zoom_factor=0.5,
            zoomed_display_height=4,
            zoomed_display_width=4,
            image_frame_stroke_width=20,
            zoomed_camera_config={
                "default_frame_stroke_width": 3,
                },
            **kwargs
        )

    def point_and_neighbors(self, scale_tracker, center_pos=[0,0]):
        # Some very strange behaviours here with the add_updater functions.
        # If I call add_updater in a for loop over a list of MObjects or a VGroup, something fails. (except sometimes it doesn't)
        # Seems like if I give the exact same update function to all the elements, then it's fine. Otherwise I need to unroll the full loop manually
        radius = 0.2
        circle_stroke_width=0
        center = Dot(center_pos, stroke_width=circle_stroke_width, radius=radius, color=WHITE)
        # up down left right
        neighbors = [Dot(stroke_width=circle_stroke_width, radius=radius) for _ in range(4)]
        neighbors = VGroup(*neighbors)
        neighbors[0].add_updater(lambda ob: ob.next_to(center, UP*scale_tracker.get_value()))
        neighbors[1].add_updater(lambda ob: ob.next_to(center, DOWN*scale_tracker.get_value()))
        neighbors[2].add_updater(lambda ob: ob.next_to(center, LEFT*scale_tracker.get_value()))
        neighbors[3].add_updater(lambda ob: ob.next_to(center, RIGHT*scale_tracker.get_value()))

        center.add_updater(lambda ob: ob.set(height=radius*scale_tracker.get_value()))
        for neighbor in neighbors: neighbor.add_updater(lambda ob: ob.set(height=radius*scale_tracker.get_value()))

        def is_best_neighbor(i):
            values = [f(*self.axes.p2c(neighbors[j].get_center())) for j in range(4)]
            return i == np.argmin(values)

        neighbors[0].add_updater(lambda ob, dt: ob.set_color( YELLOW if is_best_neighbor(0) else WHITE), call_updater=True)
        neighbors[1].add_updater(lambda ob, dt: ob.set_color( YELLOW if is_best_neighbor(1) else WHITE), call_updater=True)
        neighbors[2].add_updater(lambda ob, dt: ob.set_color( YELLOW if is_best_neighbor(2) else WHITE), call_updater=True)
        neighbors[3].add_updater(lambda ob, dt: ob.set_color( YELLOW if is_best_neighbor(3) else WHITE), call_updater=True)

        line_stroke_width=4
        lines = [ Line(stroke_width=line_stroke_width) for point in neighbors]
        lines = VGroup(*lines)
        lines[0].add_updater(lambda ob: ob.put_start_and_end_on(center.get_top(), neighbors[0].get_center()))
        lines[1].add_updater(lambda ob: ob.put_start_and_end_on(center.get_bottom(), neighbors[1].get_center()))
        lines[2].add_updater(lambda ob: ob.put_start_and_end_on(center.get_left(), neighbors[2].get_center()))
        lines[3].add_updater(lambda ob: ob.put_start_and_end_on(center.get_right(), neighbors[3].get_center()))

        lines[0].add_updater(lambda ob: ob.set_color( YELLOW if is_best_neighbor(0) else WHITE))
        lines[1].add_updater(lambda ob: ob.set_color( YELLOW if is_best_neighbor(1) else WHITE))
        lines[2].add_updater(lambda ob: ob.set_color( YELLOW if is_best_neighbor(2) else WHITE))
        lines[3].add_updater(lambda ob: ob.set_color( YELLOW if is_best_neighbor(3) else WHITE))

        for i, line in enumerate(lines):
            line.add_updater(lambda ob, dt: ob.set(stroke_width=line_stroke_width*scale_tracker.get_value()), call_updater=True )

        return center, neighbors, lines

    def construct(self):
        self.play(self.create_axes_animation)
        # Full screen color map
        self.play(FadeIn(self.elev_img))

        self.display_equation()
        self.wait(2)
        self.remove_equation()
        self.wait()
        zd_camera = self.zoomed_camera
        zd_camera_frame = zd_camera.frame
        zd_display = self.zoomed_display
        zd_display_frame = zd_display.display_frame

        center_coords = [-5, -3]
        center_pos = self.axes.c2p(*center_coords)

        zd_camera_frame.move_to(center_pos)

        scale_tracker = ValueTracker(1)
        center, neighbors, lines = self.point_and_neighbors(scale_tracker, center_pos=center_pos)
        self.play(FadeIn(neighbors, lines, center))

        # Every point has a best direction
        track_points = [ center_coords, [1,-3], [3, 5], [-4, 2], [-1.1, -5] ]
        path = VMobject()
        path.set_points_smoothly([*[self.axes.c2p(x,y) for x,y in track_points]])
        print(linear)
        self.play(MoveAlongPath(center, path), run_time=7, rate_func=rate_functions.ease_in_out_sine)

        # But the best direction also depends on the size of the step size.
        # Show best neighbor change as grid decreases (only 4 neighbors)
        self.play(scale_tracker.animate.set_value(2.5), run_time=3)
        self.play(scale_tracker.animate.set_value(0.5), run_time=3)
        # self.wait(2)
        # No matter which point we are looking at, the process will converge to a neighbor.
        # You can keep making the grid smaller, but the best direction will remain unchanged
        # Since this funciton is quite smooth, we don't need to make the step size very small before we converge


        self.play(FadeOut(neighbors,lines))
        # The maigc of differenciation, is that we can compute  don't need to evaluate the loss functions
        # In fact, we can do better than that. We can compute the best direction, something that would have been impossible
        # if we had to evaluate the value of all infinitely many directions
        dir_circ = Circle(color=WHITE, fill_opacity=0, stroke_width = 4)
        rad_tracker = ValueTracker(0.9)
        dir_circ.add_updater(lambda ob, dt: ob.move_to(center.get_center() ), call_updater=True)
        dir_circ.add_updater(lambda ob, dt: ob.set_height(rad_tracker.get_value()), call_updater=True)
        # self.play(FadeIn(dir_circ))
        def tip_pos():
            grad_scalar = 2.
            center_pos = np.array(center.get_center())
            grad = np.array(self.axes.c2p(*gradf(*self.axes.p2c(center_pos)))) # The grad is computed in coordinates, which might be scaled differently
            grad_tip = center_pos - grad_scalar*grad
            return grad_tip

        def gradf_from_pos(pos):
            x,y = self.axes.p2c(pos)
            return self.axes.c2p(-dfdx(x,y), -dfdy(x,y))


        arrow = Arrow()
        arrow.add_updater(lambda ob, dt: ob.become( Arrow(start=center.get_center(),
                                                        end=tip_pos(),
                                                        buff=0., color=WHITE)
                                                    ),
                            call_updater=True)
        self.play(FadeIn(arrow))

        track_points = [ track_points[-1], [1,-3], [3, 5], [-4, 2], [-1.1, -6] ]
        path = VMobject()
        path.set_points_smoothly([*[self.axes.c2p(x,y) for x,y in track_points]])
        self.play(MoveAlongPath(center, path), run_time=7, rate_func=rate_functions.ease_in_out_sine)

        self.play(rad_tracker.animate.set_value(0.8), run_time=2)
        self.play(rad_tracker.animate.set_value(0.2), run_time=2)
        self.play(FadeOut(center, arrow))

        v_field = ArrowVectorField(gradf_from_pos, color=WHITE, length_func=lambda x: x)
        # self.add(v_field.create())
        self.play(Create(v_field))
        self.wait()
        self.play(FadeOut(self.elev_img))
        self.play(FadeOut(v_field))

        stream_lines = StreamLines(
            gradf_from_pos,
            # color=YELLOW,
            x_range=[-7, 7, 1],
            y_range=[-4, 4, 1],
            stroke_width=3,
            virtual_time=8,  # use shorter lines
            max_anchors_per_line=5,  # better performance with fewer anchors
        )
        # self.play(stream_lines.create(), run_time=3)  # uses virtual_time as run_time
        # self.add(stream_lines)
        stream_lines.start_animation(warm_up=True, flow_speed=2.5, time_width=0.5)
        self.wait(3)
        self.play(stream_lines.end_animation())
        return


        # Show full vector field


        # Cut to me working out the derivative.
        # You don't actually need to manually compute it. We got frameworks...

        # To give you a bit of intution of how this is even possible we need to zoom into our function
        zd_rect = BackgroundRectangle(zd_display, fill_opacity=0, buff=MED_SMALL_BUFF)
        self.add_foreground_mobject(zd_rect)
        unfold_camera = UpdateFromFunc(zd_rect, lambda rect: rect.replace(zd_display))
        self.play(Create(zd_camera_frame))
        self.activate_zooming()
        self.play(self.get_zoomed_display_pop_out_animation())

        # This is because if you zoom in enough, any differentiable function becomes linear.
        # As the step size become smaller, the best direction might change.

        # This is because any differentiable function becomes linear if you zoom deep enough

        self.play(zd_camera_frame.animate.move_to([-2, -2.5,0]), run_time=3)
        self.play(zd_camera_frame.animate.move_to([-4, 2,0]), run_time=3)

        scale_factor = [0.3, 0.3, 0]
        self.play(
            zd_camera_frame.animate.scale(scale_factor),
        #    zd_display.animate.scale(scale_factor),
        )
        # self.play(self.get_zoomed_display_pop_out_animation(), rate_func=lambda t: smooth(1 - t))
        # self.play(Uncreate(zoomed_display_frame), FadeOut(frame))
