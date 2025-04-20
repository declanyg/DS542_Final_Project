from manim import *
import math

class DefaultTemplate(Scene):
    def construct(self):
        circle = Circle()  # create a circle
        circle.set_fill(PINK, opacity=0.5)  # set color and transparency

        square = Square()  # create a square
        square.flip(RIGHT)  # flip horizontally
        square.rotate(-3 * TAU / 8)  # rotate a certain amount

        self.play(Create(square))  # animate the creation of the square
        self.play(Transform(square, circle))  # interpolate the square into the circle
        self.play(FadeOut(square))  # fade out animation


class GradientDescentGraph(Scene):
    def construct(self):
        graph_func = lambda x: math.sin(2*x)+0.4*x**2
        d_graph_func = lambda x: 2*math.cos(2*x) + 2*x
        axes = Axes(x_range = (-7, 7), y_range = (-2, 6), tips = True)
        graph = axes.plot(graph_func, color = BLUE)
        
        self.play(Write(axes))
        self.play(Write(graph), run_time = 2)

        ball_x = ValueTracker(3.5)
        ball = always_redraw(lambda: Dot(
            axes.coords_to_point(ball_x.get_value(), graph_func(ball_x.get_value())),
            color=RED
        ))

        x_min, x_max = axes.x_range[:2]

        # tangent = always_redraw(
        #     lambda: TangentLine(
        #         graph,
        #         alpha=(ball_x.get_value() - x_min) / (x_max - x_min),
        #         color=YELLOW,
        #         length=4
        #     )
        # )

        self.add(ball)
        self.play(FadeIn(ball))
        self.wait(2)

        self.play(ball_x.animate.set_value(1.91873), run_time=4, rate_func=linear)

        self.wait(5)

        self.play(ball_x.animate.set_value(-0.65322), run_time=4, rate_func=linear)