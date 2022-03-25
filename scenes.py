from manim import *
from colour import Color
from math import sin, cos, sqrt, exp

def poly(x, parameters):
    r = 0
    for i,p in enumerate(parameters): r+= p*(x**i)
    return r

def color_map(val):
    val = -val/4 + 80
    hsl_color =  np.float32([val/255,0.5,0.5])
    rgb_color = np.float32(Color(hsl=hsl_color).rgb) * 255
#    rgb_color = rgb_color.astype(np.uint8)
    in_bounds = [ (rgb_color[i] >= 0) and (rgb_color[i] <=255) for i in range(3)]
    for i in range(3):
        if in_bounds[i] == False: print('Warning, color', color, 'is out of bounds')
    return rgb_color

class ColorNumberLine(NumberLine):
    def __init__(self, *args, color_res=500, label_text=None, **kwargs):
        length = kwargs['length']
        super().__init__(*args, rotation=PI/2, **kwargs)

        screen_height = config.frame_height
        img_np = np.zeros([color_res, color_res//10, 3], np.uint8)
        for i in range(color_res):
            alpha = i / color_res
            color = np.array(interpolate_color(RED, DARK_BLUE, alpha).rgb)
            img_np[i,:, :] = color * 256
        img = ImageMobject(img_np, scale_to_resolution=color_res*(screen_height/length) ).set_z_index(-1)
        self.submobjects.append(img)
        if label_text:
            label = Text(label_text).scale(0.8)
            label.shift(DOWN*(length/2 + 0.3))
            self.submobjects.append(label)

    def number_to_color(self, x):
        pass

class TestColorNL(Scene):
    def construct(self):
        # nl = NumberLine(x_range=[0, 50, 10], length=4.5)
        nl = ColorNumberLine(x_range=[0, 50, 10], length=4.5, label_text='loss')
        # nl = ColorNumberLine(x_range=[0, 50, 10], length=4.5)
        self.add(nl)
        self.play(nl.animate.shift(LEFT*2))


class Images(Scene):
    def construct(self):
        screen_height = config.frame_height
        n, m = 50,20
        img_np = np.zeros([n,m, 3], np.uint8)
        for i in range(n):
            for j in range(m):
                alpha = i / 50.
                color = np.array(interpolate_color(RED, DARK_BLUE, alpha).rgb)
                img_np[i,j, :] = color * 256
        # If scale to resolution is 1, a single pixel occupies the entire height
        # setting it to 2 occupies half the height
        # If I want n pixels to occupy the entire height I should set it to n
        img_heigh = 6
        img = ImageMobject(img_np, scale_to_resolution=n*(screen_height/img_heigh) )
        line = Line(3*DOWN + LEFT*5, UP *3+ LEFT*5)
        # print(img_np)
        self.add(img)
        self.add(line)
        self.wait()


# TODO: refactor into single parent class

class Intro(Scene):
    def construct(self):
        Title = Text("Finding functions that fit data", t2c={'functions':BLUE, 'data':GREEN}).move_to(UP*3.2).set_z_index(2)
        title_background_rect = BackgroundRectangle(Title, fill_opacity=.7, buff=0.).set_z_index(1)
        self.add(title_background_rect)
        self.play(Write(Title), run_time=1)
        self.wait()

        axes = Axes(tips=False).scale(0.8)
        labels = self.axis_labels(axes)
        time_tracker = ValueTracker(0.)

        h_params = np.array([2, 1, -1/10, 0])
        def parameters_at_t():
            t = time_tracker.get_value()
            param_path_t = parameter_path(t)
            if t < 20: return param_path_t
            elif t < 25:
                alpha = (25 -t)/5
                return alpha*param_path_t + (1-alpha)*h_params
            elif t < 26:
                alpha = (26 -t)
                return alpha*h_params + (1-alpha)*np.array([-1,1,0,0])
            # elif t < 28: return np.array([-1,1,0,0])
            elif t < 32:
                s = t - 26
                return np.array([-cos(2*PI*s/5),cos(2*PI*s/10),0,0])
            elif t < 33:
                s = 32 - 26
                last = np.array([-cos(2*PI*s/5),cos(2*PI*s/10),0,0])
                alpha = 33 - t
                return alpha* last + (1- alpha)*np.array([-2, 0, 1/5, 0])
            elif t < 38:
                s = t-33
                return np.array([-2*cos(s/2), sin(s), 1/5*cos(s), 0])
            else: return h_params

        def parameter_path(t):
            return np.array([-2*cos(t/5),cos(-t/8), sin(t)/10, -sin(cos(t/5) + 1)/100])

        def f_at_t(x): return poly(x, parameters_at_t())
        def h(x): return poly(x, h_params)

        f_graph = axes.plot(f_at_t, color=BLUE)
        f_graph.add_updater(lambda ob: ob.become(axes.plot(f_at_t, color=BLUE)))
        h_graph = axes.plot(h, color=GREEN)

        data_x_vals = [-5, -1, 3, 4]
        data_h_xy_vals = [ (x,h(x)) for x in data_x_vals ]
        data_h_dots = [Dot(axes.c2p(x, y), color=GREEN) for x,y in data_h_xy_vals]
        data_h_dots = VGroup(*data_h_dots)

        self.play(Create(axes), Create(labels))
        self.play(Create(f_graph), FadeIn(data_h_dots, lag_ratio=0.1))
        self.play(Create(h_graph))
        self.play(Uncreate(h_graph))

        dt = 10
        self.play(time_tracker.animate.set_value(time_tracker.get_value() + dt), run_time=dt)

        data_f_xy_vals = [ (x,f_at_t(x)) for x in data_x_vals ]
        data_f_dots = [Dot(axes.c2p(x, y), color=BLUE) for x,y in data_f_xy_vals]
        data_f_dots = VGroup(*data_f_dots)
        err_bars = [Line(hd, fd, color=RED) for hd, fd in zip(data_h_dots,  data_f_dots)]
        err_bars = VGroup(*err_bars)

        self.play(FadeIn(data_f_dots, lag_ratio=0.1), FadeIn(err_bars, lag_ratio=0.1))

        # Make sure the f dots and error bars stay updated as t changes (or as the axes change position)
        for x, dot in zip(data_x_vals, data_f_dots):
            # Note how x=x needs to be passed in the lambda expresion. Otherwise all the lambda expresions use the last x in the list
            dot.add_updater(lambda ob, x=x: ob.become(Dot(axes.c2p(x, f_at_t(x)), color=BLUE)))
        for x, dot in zip(data_x_vals, data_h_dots):
            dot.add_updater(lambda ob, x=x: ob.become(Dot(axes.c2p(x, h(x)), color=GREEN)))
        for bar, hd, fd in zip(err_bars, data_h_dots, data_f_dots):
            bar.add_updater(lambda ob, hd=hd, fd=fd: ob.become(Line(hd, fd, color=RED)))


        dt = 5
        self.play(time_tracker.animate.set_value(time_tracker.get_value() + dt), run_time=dt)

        loss_line = NumberLine(x_range=[0, 100, 20], length=4.5, rotation=PI/2, include_numbers=False, label_direction=LEFT).shift(LEFT*5)
        loss_text = Text('loss').scale(0.8)
        loss_text.next_to(loss_line, DOWN)
        self.play(axes.animate.scale(0.8).shift(RIGHT*2), Create(loss_line), Write(loss_text))
        def loss_func():
            loss = 0
            for x in data_x_vals: loss += (h(x) - f_at_t(x))**2
            return loss/len(data_x_vals)
        loss_dot = Dot(loss_line.n2p(loss_func()), color=RED)
        self.play(FadeIn(loss_dot))
        loss_dot.add_updater(lambda ob: ob.become(Dot(loss_line.n2p(loss_func()), color=RED)))
        self.play(time_tracker.animate.set_value(time_tracker.get_value() + dt), run_time=dt)

        # We are at time 20. From now to t=25 f approaches h
        self.play(time_tracker.animate.set_value(time_tracker.get_value() + dt), run_time=dt)

        # self.clear()
        self.play(FadeOut(Title, loss_line, loss_text, data_f_dots, data_h_dots, loss_dot, title_background_rect))
        self.wait(1)
        Title = Text("Parametrization of function spaces").move_to(UP*3.2).set_z_index(2)
        title_background_rect = BackgroundRectangle(Title, fill_opacity=.7, buff=0.).set_z_index(1)
        self.play(Write(Title), run_time=1)
        self.add(title_background_rect)

        render_text = True

        # func_eq = MathTex("f(x)=").shift(LEFT*4+UP*2).scale(0.8)
        func_eq = self.poly_f_tex(parameters_at_t()[0:2]).shift(LEFT*4+UP*2).scale(0.8)
        param_eq =self.param_list_tex(parameters_at_t()[0:2]).shift(LEFT*4+UP).scale(0.8)
        func_eq.set_z_index(1)
        param_eq.set_z_index(1)
        func_rect = self.background_rect(func_eq)
        param_rect = self.background_rect(param_eq)
        self.add(func_rect, param_rect)
        if render_text:
            func_eq.add_updater(lambda ob, dt=0: ob.become(self.poly_f_tex(parameters_at_t()[0:2]).shift(LEFT*4+UP*2).scale(0.8)),
                                call_updater=True)
            param_eq.add_updater(lambda ob, dt=0: ob.become(self.param_list_tex(parameters_at_t()[0:2]).shift(LEFT*4+UP).scale(0.8)),
                                call_updater=True)
        dt = 1
        self.play(Write(func_eq), Write(param_eq),
                  time_tracker.animate.set_value(time_tracker.get_value() + dt),
                  run_time=dt)
        # t = 26

        # Now we take a path through linear functions
        dt = 6
        self.play(time_tracker.animate.set_value(time_tracker.get_value() + dt), run_time=dt)
        # t = 32

        #  Add one extra term. deg 2 polynomials
        func_eq2 = self.poly_f_tex(parameters_at_t()[0:3]).shift(LEFT*4+UP*2).scale(0.8)
        param_eq2 =self.param_list_tex(parameters_at_t()[0:3]).shift(LEFT*4+UP).scale(0.8)
        func_eq2.set_z_index(1)
        param_eq2.set_z_index(1)
        func_rect2 = self.background_rect(func_eq)
        param_rect2 = self.background_rect(param_rect)
        self.remove(func_rect, param_rect)
        self.add(func_rect2, param_rect2)

        # Something about the updaters breaks Transform of func_eq and param_eq
        # So we substitute them by brand new versions before running the Transform animation
        func_eq_ = self.poly_f_tex(parameters_at_t()[0:2]).shift(LEFT*4+UP*2).scale(0.8)
        param_eq_ =self.param_list_tex(parameters_at_t()[0:2]).shift(LEFT*4+UP).scale(0.8)
        self.remove(func_eq, param_eq)
        self.add(func_eq_, param_eq_)
        func_eq = func_eq_
        param_eq = param_eq_

        dt = 1
        self.play(Transform(func_eq_, func_eq2),
                Transform(param_eq_, param_eq2),
                time_tracker.animate.set_value(time_tracker.get_value() + dt), run_time=dt)

        if render_text:
            func_eq.add_updater(lambda ob, dt=0: ob.become(self.poly_f_tex(parameters_at_t()[0:3]).shift(LEFT*4+UP*2).scale(0.8)),
                                call_updater=True)
            param_eq.add_updater(lambda ob, dt=0: ob.become(self.param_list_tex(parameters_at_t()[0:3]).shift(LEFT*4+UP).scale(0.8)),
                                call_updater=True)
        # t = 33

        dt = 5
        self.play(time_tracker.animate.set_value(time_tracker.get_value() + dt), run_time=dt)

    def param_list_tex(self, params):
        text = ['\\text{parameters }=', '[']
        for i, p in enumerate(params):
            if i == len(params) - 1:
                text.append("{:.2f}".format(p))
            else:
                text.append("{:.2f}".format(p)+", " )
        text.append(']')
        return MathTex(''.join(text))

    def poly_f_tex(self, params):
        text = ['f(x)=', ]
        for i, p in enumerate(params):
            if i == 0: factor = "{:.2f}".format(p)
            elif i == 1: factor = "+{:.2f}".format(p) + "x"
            else: factor = "+{:.2f}".format(p) + "x^{}".format(i)
            text.append(factor)
        return MathTex(''.join(text))

    def background_rect(self, ob, z_index = 0, opacity=0.7):
        rect = BackgroundRectangle(ob, fill_opacity=.7, buff=0.).set_z_index(z_index)
        return rect

    def axis_labels(self, ax):
        label_x = ax.get_x_axis_label(label="x")
        label_y = ax.get_y_axis_label("f(x)")
        label_x.add_updater(lambda ob, ax=ax: ob.become( ax.get_x_axis_label(label="x").shift(DOWN*0.2).scale(0.8)))
        label_y.add_updater(lambda ob, ax=ax: ob.become( ax.get_y_axis_label(label="f(x)").shift(DOWN*0.2).scale(0.8)))
        labels = VGroup(label_x, label_y)
        return labels







class Intro2(Scene):
    def get_title(self, txt):
        title = Text(txt).move_to(UP*3.2).set_z_index(2)
        title_background_rect = BackgroundRectangle(title, fill_opacity=.7, buff=0.).set_z_index(1)
        write_anim = AnimationGroup(Write(title), Create(title_background_rect))
        return VGroup(title, title_background_rect), write_anim

        # self.add(title_background_rect)

    def get_func_dots(self, axes, f, color):
        data_xy_vals = [ (x,f(x)) for x in self.data_x_vals ]
        data_dots = [Dot(axes.c2p(x, y), color=color) for x,y in data_xy_vals]
        data_dots = VGroup(*data_dots)
        for x, dot in zip(self.data_x_vals, data_dots):
            dot.add_updater(lambda ob, x=x: ob.move_to(axes.c2p(x, f(x) ) ))
        return data_dots

    def get_dots_and_err_bars(self, axes, f1, f2):
        f1_dots = self.get_func_dots(axes, f1, BLUE)
        f2_dots = self.get_func_dots(axes, f2, GREEN)
        err_bars = [Line(d1, d2, color=RED) for d1, d2 in zip(f1_dots, f2_dots)]
        err_bars = VGroup(*err_bars)
        for bar, d1, d2 in zip(err_bars, f1_dots, f2_dots):
                bar.add_updater(lambda ob, d1=d1, d2=d2: ob.put_start_and_end_on(d1.get_center(), d2.get_center()))
        return f1_dots, f2_dots, err_bars

    def tick_time(self, dt):
        return self.time_tracker.animate.set_value(self.time_tracker.get_value() + dt)

    def play_dt(self, *args, dt=1, **kwargs):
        self.play(*args, self.tick_time(dt), rate_func=linear, run_time=dt, **kwargs)

    def param_list_tex(self, params):
        text = ['\\text{parameters }=', '[']
        for i, p in enumerate(params):
            if i == len(params) - 1:
                text.append("{:.2f}".format(p))
            else:
                text.append("{:.2f}".format(p)+", " )
        text.append(']')
        return MathTex(''.join(text))

    def poly_f_tex(self, params):
        text = ['f(x)=', ]
        for i, p in enumerate(params):
            if i == 0: factor = "{:.2f}".format(p)
            elif i == 1: factor = "+{:.2f}".format(p) + "x"
            else: factor = "+{:.2f}".format(p) + "x^{}".format(i)
            text.append(factor)
        return MathTex(''.join(text))

    def background_rect(self, ob, z_index = 0, opacity=0.7):
        rect = BackgroundRectangle(ob, fill_opacity=.7, buff=0.).set_z_index(z_index)
        return rect

    def axis_labels(self, ax, x_label='x', y_label='f(x)'):
        label_x = ax.get_x_axis_label(label="x")
        label_y = ax.get_y_axis_label("f(x)")
        label_x.add_updater(lambda ob, ax=ax: ob.become( ax.get_x_axis_label(label=x_label).shift(DOWN*0.2).scale(0.8)))
        label_y.add_updater(lambda ob, ax=ax: ob.become( ax.get_y_axis_label(label=y_label).shift(DOWN*0.2).scale(0.8)))
        labels = VGroup(label_x, label_y)
        return labels


    def construct(self):
        self.data_x_vals = [-2.8, -1.2, 0.4, 2]
        self.time_tracker = ValueTracker(0.)

        def func_2param(x, p):
            assert len(p) == 2
            p1, p2 = p[0], p[1]
            return poly(x, [0, p1, p2/20])
        h_params = np.array([1.2, -2.5])
        def wandering_parameter_path(t):
            return np.array([-1.*cos(t),2.5*sin(-t)])
            # return np.array([cos(t),sin(t)])
        def zigzag_parameter_path(t):
            pass
        def parameters_at_t():
            t = self.time_tracker.get_value()
            if t < 20: return wandering_parameter_path(t)
            # elif t < 40: return zigzag_parameter_path(t-20)
            return wandering_parameter_path(t)

        def f_at_t(x): return func_2param(x, parameters_at_t())
        def h(x): return func_2param(x, h_params)

        title, write_title_anim = self.get_title("Function spaces with 2 parameters")
        self.play(write_title_anim, run_time=1)
        self.wait()

        param_axes = Axes(x_range=[-3,3], y_range=[-3,3], x_length = 5, y_length = 5, tips=False)
        param_labels = self.axis_labels(param_axes, x_label='p1', y_label='p2')
        param_dot = Dot()
        param_dot.add_updater(lambda ob, dt=0: ob.move_to(param_axes.c2p(*parameters_at_t())),
                                call_updater=True
                                )
        # self.play(Create(axes, run_time = 1), Create(labels), Create(param_axes, run_time = 0.8), Create(param_labels), run_time = 1)
        self.play(Create(param_axes, run_time = 0.8), Create(param_labels), run_time = 1)
        self.play_dt( Create(param_dot), dt=0.2)
        self.play_dt(dt=2)

        axes = Axes(x_range=[-3,3], y_range=[-3,3], x_length = 5, y_length = 5, tips=False).shift(RIGHT*3)
        labels = self.axis_labels(axes)
        self.play_dt(Create(axes), Create(labels),
                param_axes.animate.shift(LEFT*3),
                dt=1)

        f_graph = axes.plot(f_at_t, color=BLUE)
        f_graph.add_updater(lambda ob: ob.become(axes.plot(f_at_t, color=BLUE)))
        h_graph = axes.plot(h, color=GREEN)

        param_dot = Dot()
        param_dot.add_updater(lambda ob, dt=0: ob.move_to(param_axes.c2p(*parameters_at_t())),
                                call_updater=True)
        self.play_dt(Create(f_graph), Create(param_dot),dt=1)
        self.play_dt(dt=2)
        data_f_dots, data_h_dots, err_bars = self.get_dots_and_err_bars(axes, f_at_t, h)

        loss_line = NumberLine(x_range=[0, 50, 10], length=4.5, rotation=PI/2, include_numbers=False, label_direction=LEFT).shift(RIGHT*6)
        loss_text = Text('loss').scale(0.8)
        loss_text.next_to(loss_line, DOWN)
        def loss_func():
            return sum([(h(x) - f_at_t(x))**2 for x in self.data_x_vals])/len(self.data_x_vals)
        loss_dot = Dot(loss_line.n2p(loss_func()), color=RED)
        loss_dot.add_updater(lambda ob: ob.become(Dot(loss_line.n2p(loss_func()), color=RED)))

        number_bar_np = np.empty([50,5, 3], dtype=np.uint8)
        for i in range(number_bar_np.shape[0]):
            # color = interpolate_color()
            # number_bar_np[i, :, :] =
            number_bar_np[i, :, :] = color_map(i)[None]
            print('Color ', color_map(i)[None].shape)
            print(color_map(i)[None])
        color_bar = ImageMobject(number_bar_np, 80).move_to(loss_line.get_center()).set_z_index(-1)

        arrow = Arrow(start = RIGHT, end = 3*RIGHT)
        self.play_dt(axes.animate.shift(LEFT*1.),
                    param_axes.animate.shift(LEFT*1.),
                    Create(loss_line), Write(loss_text),
                    FadeIn(data_f_dots, lag_ratio=0.1),
                    FadeIn(data_h_dots, lag_ratio=0.1),
                    FadeIn(err_bars, lag_ratio=0.1),
                    FadeIn(loss_dot),
                    FadeIn(color_bar),
                    dt=1)
        self.play_dt(dt=5)
        return



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

# Base class
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

    def fadeout_points_animation(self, point_indices=None):
        print('fadeout ', point_indices)
        if  point_indices is None: point_indices = list(range(len(self.grid_points)))
        fadeout_points_animation = [FadeOut(self.grid_points[i]) for i in point_indices]
        return AnimationGroup(*reversed(fadeout_points_animation), lag_ratio = self.grid_lag)

    def fadeout_points(self, point_indices=None):
        anim = self.fadeout_points_animation(point_indices)
        self.play(anim)

    def fadein_points_animation(self, point_indices=None, color_by_elevation=False, run_time=1):
        if  point_indices is None: point_indices = list(range(len(self.grid_points)))
        if color_by_elevation:
            fadein_points_animation = [FadeIn(self.grid_points[i].set_fill(self.grid_colors[i])) for i in point_indices]
        else:
            fadein_points_animation = [FadeIn(self.grid_points[i].set_fill(WHITE)) for i in point_indices]
        return AnimationGroup(*reversed(fadein_points_animation), run_time=run_time, lag_ratio = self.grid_lag)

    def fadein_points(self, point_indices=None, color_by_elevation=False, run_time=1):
        anim = self.fadein_points_animation(point_indices, color_by_elevation, run_time)
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
        self.elev_img.fade(0.8)
        self.play(FadeIn(self.elev_img))

        # Display grid of grey dots
        self.fadein_points(color_by_elevation=False)
        self.wait()

        # self.play(self.elev_img.animate.fade(0.8))
        # Perform the measurement 1 by 1. The dots pick up the color.
        self.color_points(color_by_elevation=True)
        # Hightlight the loest dot in the screen
        self.darken_points_except_best()
        self.wait(2)
        self.fadeout_points()

class LocalSearch(SearchScene):
    def construct(self):
        self.play(self.create_axes_animation)
        self.elev_img.fade(0.8)
        self.play(FadeIn(self.elev_img))

        # Display grid of grey dots
        # Perform the measurement 1 by 1. The dots pick up the color.
        # points = self.sample_random_points(3)
        # points = [57, 1, 15]
        # Bad local optima
        points = [78]
        print('points ', points)

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
            self.play(self.fadein_points_animation(neighbors, True), FadeIn(*arrows, run_time=0.4))
            # self.play(FadeIn(*arrows, run_time=0.3))
            # self.fadein_points(neighbors, run_time=0.5)
            self.play(FadeOut(*arrows, run_time=0.4))
            # self.color_points(neighbors, dark=False, run_time=0.5)
            points.extend(neighbors)

        best = self.darken_points_except_best(points)
        best_point = points[best]


class AFewTrialsLocalSearch(SearchScene):
    def construct(self):
        self.play(self.create_axes_animation)
        self.elev_img.fade(0.8)
        self.play(FadeIn(self.elev_img))
        initial_points = [61, 28]

        for i in range(len(initial_points)):
            # Display grid of grey dots
            # Perform the measurement 1 by 1. The dots pick up the color.
            points = [initial_points[i]]
            self.fadein_points(points, color_by_elevation=True)

            best_point = None
            while True:
                best = self.darken_points_except_best(points)
                if points[best] == best_point: break
                best_point = points[best]
                neighbors = self.get_neighbors(best_point)
                arrows = self.get_arrows(best_point, neighbors)
                # If a neighbor is already in the dataset ignore it
                neighbors = [n for n in neighbors if n not in points]
                self.play(self.fadein_points_animation(neighbors, True), FadeIn(*arrows, run_time=0.3))
                self.play(FadeOut(*arrows, run_time=0.3))
                # self.color_points(neighbors, dark=False, run_time=0.5)
                points.extend(neighbors)
            best = self.darken_points_except_best(points)
            self.fadeout_points(points)


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
        # self.add(*axes.y_axis.label_list)

        # self.play(axes.y_axis.zoom_out_anim(), run_time=2)
        axes.y_axis.zoom_out_anim()

        return
        # Reference ponits
        # Grains of sand on earth
        # Number of atoms in the universe
        # Number of chess positions

        for i in range(0, 15):
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
        axes.y_axis.label_list = []

        def formatted_label(y):
            if y >= 1000: label_str = str(y//1000) + 'K'
            elif y >= 1000000: label_str = str(y//1000000) + 'M'
            else:  label_str = str(y)
            label = axes.y_axis._create_label_tex(label_str)
            label.scale(0.7)
            return label

        def add_y_ticks(y_values):
            ticks = VGroup()
            labels = VGroup()
            for y in y_values[1:]:
                ticks.add(axes.y_axis.get_tick(y))
                label = formatted_label(y)
                label.y = y
                label.add_updater(lambda ob: ob.next_to(axes.y_axis.n2p(ob.y), direction=LEFT, buff=0.1))
                labels.add(label)
            axes.y_axis.add(ticks)
            # axes.y_axis.add(labels)

            axes.y_axis.tick_list.append(ticks)
            axes.y_axis.label_list.append(labels)

        for zoom in y_zooms:
            assert zoom[0] == 0
            add_y_ticks(range(*zoom))

        def zoom_out_anim():
            zoom_i = axes.y_axis.zoom_i
            assert zoom_i < len(axes.y_axis.tick_list)
            stretch =y_zooms[zoom_i][2]/y_zooms[zoom_i+1][2]
            anim = AnimationGroup(axes.y_axis.animate.stretch(stretch, 1, about_point=origin),
                            FadeOut(axes.y_axis.tick_list[zoom_i]),
                            # FadeOut(axes.y_axis.label_list[zoom_i]),
                        )
            self.play(anim)
            # axes.y_axis.remove(axes.y_axis.tick_list[zoom_i])
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
