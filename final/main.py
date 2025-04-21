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

class Gridworld(Scene):
    def construct(self):
        grid_size = 5
        square_size = 1

        def grid_to_pos(x, y):
            return LEFT * (grid_size / 2 - 0.5 - x) * square_size + DOWN * (grid_size / 2 - 0.5 - y) * square_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                square = Square(side_length=square_size).move_to(grid_to_pos(i, j))
                self.add(square)
        

        start_cell = Square(side_length=square_size).move_to(grid_to_pos(0, 0)).set_fill(GREEN, 0.5)
        goal_cell = Square(side_length=square_size).move_to(grid_to_pos(4, 4)).set_fill(BLUE, 0.5)
        lava_cells = [(1, 2), (3, 1)]
        lava_squares = [Square(side_length=square_size).move_to(grid_to_pos(x, y)).set_fill(RED, 0.5) for x, y in lava_cells]

        self.add(start_cell, goal_cell, *lava_squares)

        agent = Circle(radius=0.2, color=WHITE, fill_opacity=1).move_to(grid_to_pos(0, 0))
        self.add(agent)

        path = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (3, 2), (4, 2), (4, 3), (4, 4)]

        for coord in path[1:]:
            self.play(agent.animate.move_to(grid_to_pos(*coord)), run_time=0.5)

        self.wait(1)

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