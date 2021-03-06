from manim import *
from colour import Color
import math
from math import sin, cos, sqrt, exp, log, log2, floor, ceil

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
        self.length = kwargs['length']
        self.x_range = kwargs['x_range']
        super().__init__(*args, rotation=PI/2, **kwargs)

        screen_height = config.frame_height
        img_np = np.zeros([color_res, color_res//10, 3], np.uint8)
        for i in range(color_res):
            alpha = i / color_res
            color = np.array(interpolate_color(RED, DARK_BLUE, alpha).rgb)
            img_np[i,:, :] = color * 256
        self.color_bar = ImageMobject(img_np, scale_to_resolution=color_res*(screen_height/self.length) ).set_z_index(-1)
        # self.submobjects.append(img)
        if label_text:
            label = Text(label_text).scale(0.8)
            label.shift(DOWN*(self.length/2 + 0.3))
            self.submobjects.append(label)

    def number_to_color(self, x):
            x_max = self.x_range[0]
            x_min = self.x_range[1]
            alpha = (x - x_min)/(x_max - x_min)
            # assert alpha>=0 and alpha<= 1, "Number is {} but the range is {} to {}".format(x, self.x_range[0], self.x_range[1])
            if alpha<0 or alpha>1:
                print("ColorNumberLine WARNING: Number is {} but the range is {} to {}".format(x, self.x_range[0], self.x_range[1]))
            alpha = np.clip(alpha, 0,1)
            # color = np.array(interpolate_color(RED, DARK_BLUE, alpha).rgb)
            color = interpolate_color(RED, DARK_BLUE, alpha)
            return color


class FuncSpaceScene(Scene):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.axis_leg_size = 4
        self.rng = [-self.axis_leg_size, self.axis_leg_size]
        self.data_x_vals = [-2.8, -1.2, 0.4, 2]
        self.time_tracker = ValueTracker(0.)

    def parameters_at_t(self): raise Exception('Needs to be implemented by child chass')
    def f_at_t(self, x): raise Exception('Needs to be implemented by child chass')
    def h(self, x): raise Exception('Needs to be implemented by child chass')
    def loss_func(self, f): raise Exception('Needs to be implemented by child chass')
    def param_loss(self, params): raise Exception('Needs to be implemented by child chass')

    def get_title(self, txt):
        if type(txt) == str:
            title = Text(txt)
        else:
            title = txt
        title.move_to(UP*3.2).set_z_index(2)
        title_background_rect = BackgroundRectangle(title, fill_opacity=.7, buff=0.).set_z_index(1)
        write_anim = AnimationGroup(Write(title), Create(title_background_rect))
        return VGroup(title, title_background_rect), write_anim
        # self.add(title_background_rect)

    def get_func_dots(self, axes, f, color):
        data_xy_vals = [ (x,f(x)) for x in self.data_x_vals ]
        data_dots = [Dot(axes.c2p(x, y), color=color).set_z_index(1) for x,y in data_xy_vals]
        data_dots = VGroup(*data_dots)
        for x, dot in zip(self.data_x_vals, data_dots):
            dot.add_updater(lambda ob, x=x: ob.move_to(axes.c2p(x, f(x) ) ))
        return data_dots

    def get_dots_and_err_bars(self, axes, f1, f2):
        f1_dots = self.get_func_dots(axes, f1, BLUE)
        f2_dots = self.get_func_dots(axes, f2, GREEN)
        err_bars = [Line(d1, d2, color=RED).set_z_index(0) for d1, d2 in zip(f1_dots, f2_dots)]
        err_bars = VGroup(*err_bars)
        for bar, d1, d2 in zip(err_bars, f1_dots, f2_dots):
            def update_bar(ob, d1=d1, d2=d2):
                v1, v2 = d1.get_center(), d2.get_center()
                # We hide the bar if there is no error
                if np.all(np.abs(v1-v2)<0.01): ob.set_fill(opacity=0.)
                else: ob.put_start_and_end_on(v1, v2).set_fill(opacity=1.)
            bar.add_updater(update_bar)
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
        label_x.add_updater(lambda ob, ax=ax: ob.become( ax.get_x_axis_label(label=x_label).shift(DOWN*0.4).scale(0.8)))
        label_y.add_updater(lambda ob, ax=ax: ob.become( ax.get_y_axis_label(label=y_label).shift(LEFT*0.4 + DOWN*0.).scale(0.8)))
        labels = VGroup(label_x, label_y)
        return labels

    def color_map(self, res=20):
        screen_height = config.frame_height
        img_np = np.zeros([res, res,  3], np.uint8)
        for i in range(res):
            for j in range(res):
                x = -self.axis_leg_size + j/(res -1) *2*self.axis_leg_size
                y = self.axis_leg_size - i/(res-1) *2*self.axis_leg_size
                l = self.param_loss([x,y])
                color = np.array(self.loss_line.number_to_color(l).rgb)
                img_np[i,j, :] = color * 256
        img = ImageMobject(img_np, scale_to_resolution=res*(screen_height/(self.param_x_length)) ).set_z_index(-1)
        img.move_to(self.param_axes.get_center())
        return img


# TODO: refactor into single parent class
class IntroFuncSpaces(FuncSpaceScene):
    h_params = np.array([2, 1, -1/10, 0])
    def wandering_parameters(self, t):
        return np.array([-2*cos(t/5),cos(-t/8), sin(t)/10, -sin(cos(t/5) + 1)/100])
    def parameters_at_t(self):
        t = self.time_tracker.get_value()
        wandering_parameters = self.wandering_parameters(t)
        linear_parameters = np.array([-1,1,0,0])
        quadratic_parameters = np.array([-2, 0, 1/5, 0])
        if self.parameter_stage == 'wandering': return wandering_parameters
        elif self.parameter_stage == 'good_fit':
            if self.good_fit_t == None: self.good_fit_t = t
            good_fit_transition_time = 4.
            hold_factor = 0.2
            alpha = (t - self.good_fit_t)/good_fit_transition_time
            # print('good fit t', self.good_fit_t, t, t - self.good_fit_t)
            if alpha < 1.:
                self.last_good_fit = alpha*self.h_params + (1-alpha)*wandering_parameters
            elif alpha >= 1 and alpha < 1+ hold_factor:
                self.last_good_fit = self.h_params
            else:
            # elif alpha >= 1+hold_factor:
                betta = alpha - (1+hold_factor)
                betta = min(1, betta)
                self.last_good_fit = betta* wandering_parameters + (1-betta)*self.h_params
            return self.last_good_fit
        elif self.parameter_stage == 'linear_wandering':
            if self.linear_wandering_t == None: self.linear_wandering_t = t
            # interpolate to a linear function
            linear_transition_length = 2
            alpha = (t-self.linear_wandering_t)/linear_transition_length
            # print('linear wandering alpha', 1/9)
            if alpha < 1: return alpha*linear_parameters + (1-alpha)* self.last_good_fit
            s= t - self.linear_wandering_t - linear_transition_length
            # We store the linear parameters so we can use then in the next transition interpolation
            self.last_linear = np.array([-cos(2*PI*s/5),cos(2*PI*s/10),0,0])
            return self.last_linear
        elif self.parameter_stage == 'quadratic_wandering':
            if self.quadratic_wandering_t == None: self.quadratic_wandering_t = t
            # interpolate to a quadratic function
            quadratic_transition_length = 2
            alpha = (t-self.quadratic_wandering_t)/quadratic_transition_length
            if alpha < 1:
                return alpha*quadratic_parameters + (1- alpha)*self.last_linear
            s= t - self.quadratic_wandering_t - quadratic_transition_length
            return np.array([-2*cos(s/2), sin(s), 1/5*cos(s), 0])
        else: raise Exception('Not valid parameter_stage', self.parameter_stage)

    def f_at_t(self, x): return poly(x, self.parameters_at_t())

    def h(self, x): return poly(x, self.h_params)

    def loss_func(self, f):
        loss = sum([(self.h(x) - f(x))**2 for x in self.data_x_vals])/len(self.data_x_vals)
        return log2(1+ loss) # Purely for asthetic reasons

    def construct(self):
        self.parameter_stage = 'wandering'
        self.good_fit_t = None
        self.linear_wandering_t = None
        self.quadratic_wandering_t = None

        Title = Text("Finding functions that fit data", t2c={'functions':BLUE, 'data':GREEN})
        title, write_title_anim = self.get_title(Title)
        self.play(write_title_anim, run_time=1)
        self.wait()

        axes = Axes(tips=False).scale(0.8).shift(DOWN*0.5)
        labels = self.axis_labels(axes, x_label='x', y_label='f(x)')
        self.play(Create(axes), Write(labels))

        f_graph = axes.plot(lambda x: self.f_at_t(x), color=BLUE)
        f_graph.add_updater(lambda ob: ob.become(axes.plot(self.f_at_t, color=BLUE)))
        h_graph = axes.plot(lambda x: self.h(x), color=GREEN)

        self.data_x_vals = [-5, -1, 3, 4]
        data_f_dots, data_h_dots, err_bars = self.get_dots_and_err_bars(axes, self.f_at_t, self.h)

        self.play_dt(Create(f_graph), FadeIn(data_h_dots, lag_ratio=0.1))
        self.play_dt(dt=5)
        self.play_dt(Create(h_graph))
        self.play_dt(dt=1)
        # self.play_dt(Uncreate(h_graph))
        self.play_dt(FadeOut(h_graph))
        self.play_dt(dt=5)
        self.play_dt( FadeIn(err_bars, lag_ratio=0.1), FadeIn(data_f_dots, lag_ratio=0.1))
        self.play_dt(dt=5)
        self.parameter_stage = 'good_fit'
        self.play_dt(dt=5)


        max_loss = 8
        self.loss_line = ColorNumberLine(x_range=[0, max_loss, 2], length=4.5, label_text='loss')
        loss_line = self.loss_line
        loss_line.shift(RIGHT*5+ DOWN*0.5)
        loss_line.color_bar.shift(RIGHT*5+ DOWN*0.5)
        loss_dot = Dot(loss_line.n2p(self.loss_func(self.f_at_t)))
        # loss_dot.add_updater(lambda ob: ob.become(Dot(loss_line.n2p(self.loss_func(self.f_at_t)), radius=0.1)))
        loss_dot.add_updater(lambda dot: dot.move_to(loss_line.n2p(self.loss_func(self.f_at_t))))
        self.play_dt(Create(loss_line), FadeIn(loss_line.color_bar),
                    axes.animate.shift(LEFT*2.5).scale(0.8),
                    FadeIn(loss_dot),
                    dt=1)
        self.play_dt(dt=5)

        self.parameter_stage = 'linear_wandering'
        self.play_dt(FadeOut(Title, loss_line, loss_line.color_bar, err_bars, data_f_dots, data_h_dots, loss_dot))
        self.play_dt(dt=1)
        Title = Text("Parametrization of function spaces")
        title, write_title_anim = self.get_title(Title)
        self.play_dt(write_title_anim, dt=1)
        f_eq, p_eq = self.param_equs(2)
        self.play_dt(Write(f_eq), Write(p_eq), dt=1)
        self.play_dt(dt=5)

        self.parameter_stage = 'quadratic_wandering'
        self.play_transform_equs(f_eq, p_eq, 3)
        self.play_dt(dt=2)
        self.play_dt(dt=3)
        return

    def param_equs(self, n):
        text_shift = RIGHT*4 -UP*1
        up_shift = 1.2
        params = self.parameters_at_t()[0:n]
        p_eq  = VMobject()
        f_eq  = VMobject()
        p_eq.add_updater(lambda ob, dt=0:
                ob.become(self.param_list_tex_(self.parameters_at_t()[0:n]).shift(text_shift + UP*up_shift)),
                            call_updater=True)
        f_eq.add_updater(lambda ob, dt=0:
                ob.become(self.poly_f_tex_(self.parameters_at_t()[0:n]).shift(text_shift)),
                            call_updater=True)
        # eq_group = VGroup(p_eq, f_eq)
        return f_eq, p_eq

    def play_transform_equs(self, f_eq, p_eq, new_n):
        f_eq.updaters.pop()
        p_eq.updaters.pop()
        f_eq2, p_eq2 = self.param_equs(new_n)
        f_up2 = f_eq2.updaters.pop()
        p_up2 = p_eq2.updaters.pop()
        self.play_dt(Transform(f_eq, f_eq2),Transform(p_eq, p_eq2), dt=1)
        f_eq.updaters.append(f_up2)
        p_eq.updaters.append(p_up2)


    def param_list_tex_(self, params):
        text = ['\\text{parameters }=', '[']
        for i, p in enumerate(params):
            if i == len(params) - 1:
                text.append("{:.2f}".format(p))
            else:
                text.append("{:.2f}".format(p)+", " )
        text.append(']')
        tex = MathTex(''.join(text)).scale(0.7)
        return tex

    def poly_f_tex_(self, params):
        text = ['f(x)=', ]
        def signed_num(p):
            if p>=0: return "+{:.2f}".format(p)
            else: return "{:.2f}".format(p)
        for i, p in enumerate(params):
            if i == 0: factor = "{:.2f}".format(p)
            elif i == 1: factor = signed_num(p) + "x"
            else: factor = signed_num(p) + "x^{}".format(i)
            text.append(factor)
        tex = MathTex(''.join(text)).scale(0.7)
        # rect = self.background_rect(tex)
        # tex.submobjects.append(rect).shift(text_shift).scale(0.8)        return tex
        return tex


class RandomParamPath:
    def __init__(self, param_shape, point_num=10, rng=1):
        points = []
        for _ in range(point_num):
            params = rng* np.random.random(param_shape) - rng/2
            points.append(params)
        self.points = np.stack(points)
        self.curve = bezier(self.points)

    def __call__(self, t):
        return self.curve(t)


class FuncSpaceZoo(Scene):
    def poly(self, x):
        t = self.t_tracker.get_value()
        poly_params = self.poly_param_curve(t)
        res = 0
        for i in range(poly_params.size): res += poly_params[i] * x**i
        return res

    def harmonic(self, x):
        t = self.t_tracker.get_value()
        # harm_params = self.harm_param_path(t)
        harm_params = self.harm_param_curve(t)
        res = 0
        for i in range(harm_params.size//2): res += harm_params[i] * cos(i*x)
        for i in range(harm_params.size//2, harm_params.size): res += harm_params[i] * sin(i*x)
        return res

    def nn(self, x):
        t = self.t_tracker.get_value()
        # harm_params = self.harm_param_path(t)
        params = self.nn_param_curve(t)
        N = self.nn_N
        assert  len(params) == N + N**2 + N, len(params)
        A_1 = params[:N]
        A_2 = params[N:N**2+N].reshape([N,N])
        A_3 = params[N**2+N:]
        if self.X:
            print(A_1)
            print(A_2)
            print(A_3)
            self.X = False
        h = np.tanh(x * A_1)
        h = np.tanh(A_2.dot(h))
        res = np.tanh(A_3.dot(h))
        return res

    def separator(self, title):
        line = Line(5*LEFT, 5*RIGHT).set_z_index(1)
        title = Text(title).scale(0.7).set_z_index(2)
        box = BackgroundRectangle(title, fill_opacity=1., buff=0.1).set_z_index(1)
        group = VGroup(line, title, box)
        return group, AnimationGroup(Create(line), Write(title), FadeIn(box))

    def dt_anim(self, dt):
        t = self.t_tracker.get_value()
        return self.t_tracker.animate.set_value(t + dt)

    def construct(self):
        self.X = True
        self.t_tracker = ValueTracker(0)
        # poly_path_points = np.array([[0.5,2,-1/4], [-1,0, 1/5]])
        poly_path_points = np.array([[0.5,0.6,0.02, 0.01],
                                    [1/2,0, -1/17, +0.02]])
        harm_path_points = np.array([[1/2, 0.8, 1/3, 1/9, 0, -1/9],
                                    [1/2, 1/10, 1/9, -1/2, 1/18, 1/8]])
        # nn_path_points = np.array([[1/3,-1/6,-1/10,1,2,1/9,-1/9,1/5]])
        self.nn_N = 4
        nn_path_points = np.random.random([3,2*self.nn_N + self.nn_N**2])

        self.poly_param_curve = bezier(poly_path_points)
        self.harm_param_curve = bezier(harm_path_points)
        self.nn_param_curve = bezier(nn_path_points)

        # a natural way of rescaling nubmers to make nicer lookng polynomials
        ax_x_length = 5
        ax_y_length =  2
        ax_x_range = [-4, 4]
        ax_y_range = [-2, 2]
        up_shift = 2.4
        right_shift = 3.
        ax_scale = 1.
        text_scale = 0.7
        separator_shift = 1.2

        # POLY
        self.poly_axes = Axes(x_length=ax_x_length, y_length=ax_y_length, x_range = ax_x_range, y_range=ax_y_range, tips=False).scale(ax_scale).shift(UP*up_shift + RIGHT*right_shift)
        poly_eq = MathTex(r'p_i &\in \mathbb{R} \\ f(x) &= \sum_{i=0}^n p_i x^i').scale(text_scale).shift(UP*up_shift - RIGHT*right_shift)
        p_sep, p_sep_anim = self.separator('Polynomial')
        p_sep.shift(UP*(up_shift+separator_shift) )
        poly_plot = self.poly_axes.plot(lambda x: self.poly(x), color=BLUE)
        self.play(Create(self.poly_axes),
                    Write(poly_eq), p_sep_anim,
                    Create(poly_plot),
                    rate_func=linear)
        poly_plot.add_updater(lambda ob: ob.become(self.poly_axes.plot(lambda x: self.poly(x), color=BLUE)))

        # HARMONIC
        self.harm_axes = Axes(x_length=ax_x_length, y_length=ax_y_length, x_range = ax_x_range, y_range=ax_y_range, tips=False).scale(ax_scale).shift(RIGHT*right_shift)
        harm_eq = MathTex(r'a_i, b_i &\in \mathbb{R} \\ f(x) = \sum_{i=0}^n a_i &cos(ix) + b_i sin(ix)').scale(text_scale).shift(- RIGHT*right_shift)
        h_sep, h_sep_anim = self.separator('Harmonic')
        h_sep.shift(UP*(separator_shift) )
        harm_plot = self.harm_axes.plot(lambda x: self.harmonic(x), color=BLUE)
        self.play(Create(self.harm_axes),
                    Write(harm_eq), h_sep_anim,
                    Create(harm_plot),
                    self.dt_anim(0.2),
                    rate_func=linear,
                    run_time=1)
        harm_plot.add_updater(lambda ob: ob.become(self.harm_axes.plot(lambda x: self.harmonic(x), color=BLUE)))

        # Neural Network
        self.nn_axes = Axes(x_length=ax_x_length, y_length=ax_y_length, x_range = ax_x_range, y_range=ax_y_range, tips=False).scale(ax_scale).shift(-UP*up_shift+RIGHT*right_shift)
        nn_eq = MathTex(r'A_1 \in \mathbb{R}^{n,1}, &A_2 \in \mathbb{R}^{n,n}, A_3 \in \mathbb{R}^{1,n} \\  f(x) &= A_3 \sigma( A_2 \sigma(A_1 x))').scale(text_scale).shift(-UP*up_shift - RIGHT*right_shift)
        n_sep, n_sep_anim = self.separator('3 Layer Neural Network')
        n_sep.shift(UP*(-up_shift+separator_shift) )
        nn_plot = self.nn_axes.plot(lambda x: self.nn(x), color=BLUE)
        self.play(Create(self.nn_axes),
                    Write(nn_eq), n_sep_anim,
                    Create(nn_plot),
                    self.dt_anim(0.2),
                    rate_func=linear,
                    run_time=1)
        nn_plot.add_updater(lambda ob: ob.become(self.nn_axes.plot(lambda x: self.nn(x), color=BLUE)))
        self.play(self.dt_anim(1.5), run_time=3, rate_func=linear)



class IntroColorMap(FuncSpaceScene):
    def func_2param(self, x, p):
        assert len(p) == 2
        p1, p2 = p[1], p[0]
        return poly(x, [0, p1/3, p2/20])

    def wandering_parameters(self, t):
        return np.array([-1.*cos(t),2.5*sin(-t)])

    def parameters_at_t(self):
        t = self.time_tracker.get_value()
        wandering_parameters = self.wandering_parameters(t)
        if self.parameter_stage == 'wandering':
            return wandering_parameters
        elif self.parameter_stage == 'transition':
            if self.transition_t == None:
                self.transition_t = t
                self.param_at_transition_t = wandering_parameters
            alpha = (t - self.transition_t)/self.transition_length
            transition_params = alpha*self.param_grid_list[0] + (1-alpha)*self.param_at_transition_t
            # return self.param_grid_list[0]
            return transition_params
        elif self.parameter_stage == 'zigzag':
            if self.zigzag_t == None: self.zigzag_t = t
            t = t-self.zigzag_t
            curr_ind = floor(t/self.time_per_dot)
            next_ind = ceil(t/self.time_per_dot)
            if curr_ind >= len(self.param_grid_list): curr_ind = len(self.param_grid_list) -1
            if next_ind >= len(self.param_grid_list): next_ind = len(self.param_grid_list) -1
            self.dot_grid[curr_ind].set_fill(opacity=1.)
            alpha = t/self.time_per_dot %1
            return alpha*self.param_grid_list[next_ind] + (1-alpha)*self.param_grid_list[curr_ind]
        elif self.parameter_stage == 'transition2':
            if self.transition2_t == None:
                self.transition2_t = t
            alpha = (t - self.transition2_t)/self.transition_length
            transition_params = alpha*self.wandering_parameters(self.transition2_t+self.transition_length) + (1-alpha)*self.param_grid_list[-1]
            # return self.param_grid_list[0]
            return transition_params
        else: raise Exception('Not valid parameter_stage')

    def f_at_t(self, x):
        return self.func_2param(x, self.parameters_at_t())

    h_params = np.array([1.2, -1.5])

    def h(self, x): return self.func_2param(x, self.h_params)

    def loss_func(self, f):
        loss = sum([(self.h(x) - f(x))**2 for x in self.data_x_vals])/len(self.data_x_vals)
        return log2(1+ loss) # Purely for asthetic reasons

    def param_loss(self, params):
        return self.loss_func(lambda x: self.func_2param(x, params))

    def make_parameter_grid(self, grid_size=7, rng= 4- 0.3, opacity=1.):
        param_grid_list = []
        grid_rng = rng
        for y in range(grid_size+1):
            for x in range(grid_size+1):
                if (-1)**y == -1: x = grid_size -x
                # x,y range over 0 -> grid_size -1
                p1 = -grid_rng + 2*grid_rng*x/grid_size
                p2 = grid_rng - 2*grid_rng*y/grid_size
                param_grid_list.append( np.array([p1,p2]) )
        self.param_grid_list = param_grid_list

        grid_losses = []
        self.dot_grid = []
        for i, params in enumerate(self.param_grid_list):
            grid_losses.append(self.param_loss(params))
            dot = Dot(self.param_axes.c2p(*params),
                        radius = 0.1,
                        fill_opacity=opacity,
                        color=self.loss_line.number_to_color(self.param_loss(params)))
            self.dot_grid.append(dot)
            self.add(dot)
        print('max loss in grid ', max(grid_losses))
        print('min loss in grid ', min(grid_losses))


    def construct(self):
        self.parameter_stage = 'wandering'
        self.transition_t = None
        self.zigzag_t = None
        self.transition2_t = None
        self.transition_length = 2
        self.time_per_dot = 0.2
        self.param_x_length = 5
        self.param_y_length = 5

        # The main function family of the scene. Has 2 parameters
        title, write_title_anim = self.get_title("Function spaces with 2 parameters")
        self.play(write_title_anim, run_time=1)
        self.wait()

        param_axes = Axes(x_range=self.rng, y_range=self.rng, x_length=self.param_x_length, y_length=self.param_y_length, tips=False)
        param_axes.shift(DOWN*0.5)
        self.param_axes = param_axes

        # TODO: add to submobjects automatically
        param_labels = self.axis_labels(param_axes, x_label='p1', y_label='p2')
        param_dot = Dot()
        param_dot.add_updater(lambda ob, dt=0: ob.move_to(param_axes.c2p(*self.parameters_at_t())),
                                call_updater=True
                              )
        # self.play(Create(axes, run_time = 1), Create(labels), Create(param_axes, run_time = 0.8), Create(param_labels), run_time = 1)
        self.play(Create(param_axes, run_time = 0.8), Create(param_labels), run_time = 1)
        self.play_dt( Create(param_dot), dt=0.2)
        self.play_dt(dt=2)

        axes = Axes(x_range=self.rng, y_range=self.rng, x_length = 5, y_length = 5, tips=False).shift(DOWN*0.5+RIGHT*3)
        labels = self.axis_labels(axes)
        self.play_dt(Create(axes), Create(labels),
                param_axes.animate.shift(LEFT*3),
                dt=1)

        f_at_t = lambda x: self.f_at_t(x)
        # without making this lambda funciton there is an error from pickling a lock
        f_graph = axes.plot(f_at_t, color=BLUE)
        f_graph.add_updater(lambda ob: ob.become(axes.plot(self.f_at_t, color=BLUE)))
        h_graph = axes.plot(self.h, color=GREEN)

        self.play_dt(Create(f_graph))
        self.play_dt(dt=2)
        data_f_dots, data_h_dots, err_bars = self.get_dots_and_err_bars(axes, self.f_at_t, self.h)

        max_loss = 5
        self.loss_line = ColorNumberLine(x_range=[0, max_loss, 1], length=4.5, label_text='loss')
        loss_line = self.loss_line

        loss_line.shift(RIGHT*6 + DOWN*0.3)
        loss_line.color_bar.shift(RIGHT*6 + DOWN*0.3)
        loss_dot = Dot(loss_line.n2p(self.loss_func(self.f_at_t)))
        # loss_dot.add_updater(lambda ob: ob.become(Dot(loss_line.n2p(self.loss_func(self.f_at_t)), radius=0.1)))
        loss_dot.add_updater(lambda dot: dot.move_to(loss_line.n2p(self.loss_func(self.f_at_t))))

        self.play_dt(axes.animate.shift(LEFT*1.),
                    param_axes.animate.shift(LEFT*1.),
                    Create(loss_line), FadeIn(loss_line.color_bar),
                    FadeIn(err_bars, lag_ratio=0.1),
                    FadeIn(data_f_dots, lag_ratio=0.1),
                    FadeIn(data_h_dots, lag_ratio=0.1),
                    FadeIn(loss_dot),
                    dt=1)
        param_dot.add_updater(lambda ob: ob.set_color(loss_line.number_to_color(self.loss_func(self.f_at_t))))
        self.play_dt(dt=5)
        # constructs the grid of parameters that will cover the param axes
        self.make_parameter_grid(opacity=0.)
        self.parameter_stage = 'transition'
        self.play_dt(dt=self.transition_length)
        self.parameter_stage = 'zigzag'
        zigzag_length = self.time_per_dot * len(self.dot_grid)
        self.play_dt(dt=zigzag_length)
        color_map_img = self.color_map()
        param_dot.updaters.pop() # Stops the color from updating
        param_dot.set_fill(color=WHITE)
        self.parameter_stage = 'transition2'
        self.play_dt(FadeIn(color_map_img), dt=1/2)
        self.play_dt(dt=self.transition_length-1/2)
        # self.play_dt(dt=self.transition_length)
        self.parameter_stage = 'wandering'
        self.play_dt(dt=10)
        return









##############################################################################

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
