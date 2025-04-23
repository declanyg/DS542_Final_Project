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

        coin_positions = [(0, 4), (2, 1), (3, 2), (4, 2), (1, 3)]
        coins = {}

        for coord in coin_positions:
            coin = Circle(radius=0.2, color=GOLD_E, fill_opacity=1).move_to(grid_to_pos(*coord))
            self.add(coin)
            coins[coord] = coin

        agent = Circle(radius=0.2, color=WHITE, fill_opacity=1).move_to(grid_to_pos(0, 0))
        self.add(agent)

        self.wait(4)
        highlight = Square(side_length=square_size + 0.1, color=YELLOW).move_to(grid_to_pos(4, 4))
        self.add(highlight)
        self.play(Create(highlight))
        self.play(FadeOut(highlight))

        lava_highlights = [
            Square(side_length=square_size + 0.1, color=YELLOW).move_to(grid_to_pos(x, y))
            for x, y in lava_cells
        ]
        self.add(*lava_highlights)
        self.play(*[Create(highlight) for highlight in lava_highlights])
        self.play(*[FadeOut(highlight) for highlight in lava_highlights])

        coin_highlights = [
            Square(side_length=square_size + 0.1, color=YELLOW).move_to(grid_to_pos(x, y))
            for x, y in coin_positions
        ]
        self.add(*coin_highlights)
        self.play(*[Create(highlight) for highlight in coin_highlights])
        self.play(*[FadeOut(highlight) for highlight in coin_highlights])

        path = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (3, 2), (4, 2), (4, 3), (4, 4)]

        for coord in path[1:]:
            self.play(agent.animate.move_to(grid_to_pos(*coord)), run_time=0.5)
            if coord in coins:
                self.remove(coins[coord])
                del coins[coord]

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

class GRPOText(Scene):
    def construct(self):
        title = Text("GRPO (Group Relative Policy Optimization)", font_size=48)
        self.play(Write(title))

class ReinforcementLearningText(Scene):
    def construct(self):
        rl_text = Text("Reinforcement Learning", font_size=48)
        self.play(Write(rl_text))

        self.wait(2)

        ml_text = Text("Machine Learning", font_size=48)
        ml_text.to_edge(UP)
        self.play(Write(ml_text))
        self.wait(1)

        rl_text_target = rl_text.copy().scale(0.8)
        rl_text_target.to_edge(RIGHT).shift(DOWN * 0.5)

        self.play(Transform(rl_text, rl_text_target))

        ml_box = SurroundingRectangle(ml_text, color=BLUE, corner_radius=0.2)
        rl_box = SurroundingRectangle(rl_text, color=BLUE, corner_radius=0.2)
        self.play(Create(ml_box), Create(rl_box))

        branch_rl = Arrow(ml_box.get_bottom(), rl_box.get_top())
        self.play(Create(branch_rl))
        self.wait()

        sl_text = Text("Supervised Learning...", font_size=48).scale(0.8)
        sl_text.to_edge(LEFT).shift(DOWN * 0.5)

        sl_box = SurroundingRectangle(sl_text, color=BLUE, corner_radius=0.2)
        self.play(Write(sl_text), Create(sl_box))
        branch_sl = Arrow(ml_box.get_bottom(), sl_box.get_top())
        self.play(Create(branch_sl))
        

class GRPO(Scene):
    def construct(self):
        title = Text("GRPO (Group Relative Policy Optimization)", font_size=24).to_edge(UP)
        self.play(Write(title))
        
        # Main Equation
        equation = MathTex(
            r"J_{\text{GRPO}}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(O \mid q)} \left[",
            r"\frac{1}{G} \sum_{i=1}^G \min \left(",
            r"\frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)} A_i,",
            r"\text{clip} \left(",
            r"\frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)},",
            r"1 - \epsilon, 1 + \epsilon",
            r"\right) A_i",
            r"\right)",
            r"- \beta D_{KL}(\pi_\theta \| \pi_{\text{ref}})",
            r"\right]"
        ).scale(0.5)

        # Animate equation
        self.play(Write(equation[0]))  # Expectation
        self.wait(0.5)
        self.play(Write(equation[1]))  # Left bracket
        self.play(Write(equation[2]))  # Group average
        self.play(Write(equation[3]))  # min(
        self.play(Write(equation[4]))  # Policy ratio * advantage
        self.play(Write(equation[5]))  # Clipped term
        self.play(Write(equation[6]))  # )
        self.play(Write(equation[7]))  # KL penalty
        self.play(Write(equation[8]))  # Right bracket