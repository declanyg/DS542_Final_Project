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

class ReinforcementLearningConcepts(Scene):
    def construct(self):
        title = Text("Reinforcement Learning Concepts", font_size=48)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        # Create a list of concepts
        concepts = ["- Agent", "- Environment", "- Rewards", "- Policy"]

        list_text = VGroup(*[
            Text(concept, font_size=36) for concept in concepts
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.3)

        self.play(LaggedStart(*[Write(t) for t in list_text], lag_ratio=0.8))
        self.wait(2)

class GridworldConceptsDemo(Scene):
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

        self.wait(1)
        highlight = Square(side_length=square_size + 0.1, color=YELLOW).move_to(grid_to_pos(0, 0))
        self.add(highlight)
        self.play(Create(highlight))
        self.play(FadeOut(highlight))

        self.wait(1)
        highlight = Square(side_length=grid_size + 0.1, color=YELLOW)
        self.add(highlight)
        self.play(Create(highlight))
        self.play(FadeOut(highlight))

        self.wait(2)
        arrow_len = 0.4
        agent_pos = grid_to_pos(0, 0)
        arrows = VGroup(
            Arrow(agent_pos + UP * 0.5, agent_pos + UP * (0.5 + arrow_len), buff=0, color=YELLOW, stroke_width=8),
            Arrow(agent_pos + DOWN * 0.5, agent_pos + DOWN * (0.5 + arrow_len), buff=0, color=YELLOW, stroke_width=8),
            Arrow(agent_pos + LEFT * 0.5, agent_pos + LEFT * (0.5 + arrow_len), buff=0, color=YELLOW, stroke_width=8),
            Arrow(agent_pos + RIGHT * 0.5, agent_pos + RIGHT * (0.5 + arrow_len), buff=0, color=YELLOW, stroke_width=8)
        )

        self.play(LaggedStart(*[Create(arrow) for arrow in arrows], lag_ratio=0.3))
        self.play(LaggedStart(*[FadeOut(arrow) for arrow in arrows], lag_ratio=0.0))

        path = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (3, 2), (4, 2), (4, 3), (4, 4)]

        for i in range(1, len(path)):
            prev_coord = path[i - 1]
            curr_coord = path[i]

            start = grid_to_pos(*prev_coord)
            end = grid_to_pos(*curr_coord)
            arrow = Arrow(start, end, buff=0.1, color=YELLOW, stroke_width=6)
            self.play(Create(arrow))
            self.play(FadeOut(arrow))
            self.play(agent.animate.move_to(end))

            if curr_coord in coins:
                self.remove(coins[curr_coord])
                score_text = Text("+0.1", font_size=36, color=GREEN).move_to(end+ UP)
                self.play(Write(score_text))
                self.play(FadeOut(score_text))
                del coins[curr_coord]
            
            if curr_coord == (4, 4):
                score_text = Text("+1", font_size=36, color=GREEN).move_to(grid_to_pos(4, 4)+ UP)
                self.play(Write(score_text))
                self.play(FadeOut(score_text))
        
        # self.camera.frame.save_state()
        # self.play(self.camera.frame.animate.shift(RIGHT * 3))
        
        # text_on_right = Text("Reward", font_size=36).shift(RIGHT * 4)  # Position it on the right side
        # self.play(Write(text_on_right))

        # self.play(Restore(self.camera.frame))

class GridworldPolicyDemo(Scene):
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

        coin_positions = [(0, 4), (2, 1), (4, 2), (1, 3)]
        coins = {}

        for coord in coin_positions:
            coin = Circle(radius=0.2, color=GOLD_E, fill_opacity=1).move_to(grid_to_pos(*coord))
            self.add(coin)
            coins[coord] = coin

        agent = Circle(radius=0.2, color=WHITE, fill_opacity=1).move_to(grid_to_pos(3, 2))
        self.add(agent)

        self.wait(12)

        agent_pos = grid_to_pos(3, 2)
        arrow_len = 0.4
        arrows = VGroup(
            Arrow(agent_pos + UP * 0.5, agent_pos + UP * (0.5 + arrow_len), buff=0, color=GREEN, stroke_width=8),
            Arrow(agent_pos + LEFT * 0.5, agent_pos + LEFT * (0.5 + arrow_len), buff=0, color=GREEN, stroke_width=8),
            Arrow(agent_pos + RIGHT * 0.5, agent_pos + RIGHT * (0.5 + arrow_len), buff=0, color=GREEN, stroke_width=8)
        )

        self.play(LaggedStart(*[Create(arrow) for arrow in arrows], lag_ratio=0.3))
        self.play(LaggedStart(*[FadeOut(arrow) for arrow in arrows], lag_ratio=0.0))

        self.wait(1)

        self.play(agent.animate.move_to(grid_to_pos(3,1)))
        self.play(agent.animate.set_fill(RED).set_stroke(RED, width=2))
        score_text = Text("-1", font_size=36, color=RED).move_to(grid_to_pos(3,1)+ UP)
        self.play(Write(score_text))
        self.play(FadeOut(score_text))
        self.wait(1)
        self.play(agent.animate.move_to(grid_to_pos(0,0)).set_fill(WHITE).set_stroke(WHITE, width=2))

        path = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 4), (2, 4), (3, 4), (4, 4)]

        for i in range(1, len(path)):
            prev_coord = path[i - 1]
            curr_coord = path[i]

            end = grid_to_pos(*curr_coord)
            self.play(agent.animate.move_to(end))

            if curr_coord in coins:
                self.remove(coins[curr_coord])
                score_text = Text("+0.1", font_size=36, color=GREEN).move_to(end+ UP)
                self.play(Write(score_text))
                self.play(FadeOut(score_text))
                del coins[curr_coord]
            
            if curr_coord == (4, 4):
                score_text = Text("+1", font_size=36, color=GREEN).move_to(grid_to_pos(4, 4)+ UP)
                self.play(Write(score_text))
                self.play(FadeOut(score_text))

class GridWorldMDP(Scene):
    def construct(self):

        def get_mini_grid(position):
            grid_size = 5
            square_size = 0.3  # smaller so it can be used as a mini visual
            grid_group = VGroup()

            def grid_to_pos(x, y):
                return LEFT * (grid_size / 2 - 0.5 - x) * square_size + DOWN * (grid_size / 2 - 0.5 - y) * square_size

            # Add grid squares
            for i in range(grid_size):
                for j in range(grid_size):
                    square = Square(side_length=square_size).move_to(grid_to_pos(i, j))
                    grid_group.add(square)

            # Start and goal cells
            start_cell = Square(side_length=square_size).move_to(grid_to_pos(0, 0)).set_fill(GREEN, 0.5)
            goal_cell = Square(side_length=square_size).move_to(grid_to_pos(4, 4)).set_fill(BLUE, 0.5)
            grid_group.add(start_cell, goal_cell)

            # Lava
            lava_cells = [(1, 2), (3, 1)]
            for x, y in lava_cells:
                lava = Square(side_length=square_size).move_to(grid_to_pos(x, y)).set_fill(RED, 0.5)
                grid_group.add(lava)

            # Coins
            coin_positions = [(0, 4), (2, 1), (3, 2), (4, 2), (1, 3)]
            for coord in coin_positions:
                coin = Circle(radius=0.08, color=GOLD_E, fill_opacity=1).move_to(grid_to_pos(*coord))
                grid_group.add(coin)

            # Agent
            agent = Circle(radius=0.08, color=WHITE, fill_opacity=1).move_to(grid_to_pos(*position))
            grid_group.add(agent)

            return grid_group

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

        # Show transition probabilities for the first move (e.g., moving right from (0, 0) to (0, 1))
        prob_label_right = Text("P(Right) = 0.7", font_size=24).shift(RIGHT * 4 + UP * 1.5)
        prob_label_up = Text("P(Up) = 0.3", font_size=24).shift(RIGHT * 4 + UP * 0.5)
        prob_label_left = Text("P(Left) = 0", font_size=24).shift(RIGHT * 4 + DOWN * 0.5)
        prob_label_down = Text("P(Down) = 0", font_size=24).shift(RIGHT * 4 + DOWN * 1.5)

        self.play(Write(prob_label_right), Write(prob_label_up), Write(prob_label_left), Write(prob_label_down))

        self.play(agent.animate.move_to(grid_to_pos(1, 0)), run_time=0.5)

        self.wait(2)

        self.play(FadeOut(*self.mobjects))

        title = Text("Markov Decision Process (MDP)", font_size=48)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        # Create a list of concepts
        concepts = ["- Current State", "- Possible Actions ", "- Transition Probabilities", "- Reward Function"]

        list_text = VGroup(*[
            Text(concept, font_size=36) for concept in concepts
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.3)

        self.play(LaggedStart(*[Write(t) for t in list_text], lag_ratio=0.8))
        self.wait(2)

        self.play(FadeOut(*self.mobjects))

        title = Text("Policy (π)", font_size=48)
        title.to_edge(UP)
        self.play(Write(title))

        self.wait(1)
        equation = MathTex(r"\pi(s) = a")

        self.play(Write(equation))

        self.wait(2)

        mini_grid = get_mini_grid((0, 0)).next_to(equation, DOWN)
        self.play(Transform(equation, MathTex(r"\pi((0,0)) = RIGHT")), FadeIn(mini_grid))
        self.wait(1)
        self.play(Transform(equation, MathTex(r"\pi((4,3)) = UP")), Transform(mini_grid, get_mini_grid((4, 3)).next_to(equation, DOWN)))
        self.wait(1)
        self.play(Transform(equation, MathTex(r"\pi((3,4)) = RIGHT")), Transform(mini_grid, get_mini_grid((3, 4)).next_to(equation, DOWN)))
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

class ReinforcementLearningIntro(Scene):
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
            r"\frac{\pi_\theta(o_i \mid q)}{",
            r"\pi_{\theta_{\text{old}}}",
            r"(",
            r"o_i ",
            r"\mid ",
            r"q",
            r")} "
            r"A_i",
            r",\text{clip} \left(",
            r"\frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)},",
            r"1 - \epsilon, 1 + \epsilon",
            r"\right) A_i",
            r"\right)",
            r"- \beta D_{KL}(\pi_\theta \| \pi_{\text{ref}})",
            r"\right]"
        ).scale(0.5)

        # Animate equation
        self.play(Write(equation), runtime=1)

        self.play(equation.animate.shift(UP * 1.5))

        #Transforming the q into question 
        original_q = equation[7].copy()
        self.play(equation[7].animate.move_to(ORIGIN).scale(1.75), runtime=0.25)
        self.play(Transform(equation[7], Text("question", font_size=24, color=WHITE)))
        self.wait(7)
        self.play(Transform(equation[7], Text("What is Deepseek?", font_size=24, color=WHITE)))
        self.play(Transform(equation[7], original_q))

        #Transforming the o_i into output 
        original_oi = equation[5].copy()
        self.play(equation[5].animate.move_to(ORIGIN).scale(1.75), runtime=0.25)
        self.play(Transform(equation[5], Text("output", font_size=24, color=WHITE)))
        self.wait(1)
        #Highlight old policy
        self.play(Indicate(equation[3], color=YELLOW))
        self.wait(5)
        #continue o_i stuff
        self.play(Transform(equation[5], Text("A Chinese AI company", font_size=24, color=WHITE)))
        self.wait(0.25)
        self.play(Transform(equation[5], Text("A chatbot", font_size=24, color=WHITE)))
        self.play(Transform(equation[5], Text("A blue whale", font_size=24, color=WHITE)))
        self.wait(1)
        self.play(Transform(equation[5], original_oi))

        #Shifting equation back to middle
        self.play(equation.animate.shift(DOWN * 1.5))
        self.wait(0.5)

class GRPOScene2(Scene):
    def construct(self):
        title = Text("GRPO (Group Relative Policy Optimization)", font_size=24).to_edge(UP)
        self.add(title)

        # Main Equation
        equation = MathTex(
            r"J_{\text{GRPO}}(\theta) = ",
            r"\mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(O \mid q)}",
            r"\left[\frac{1}{G} \sum_{i=1}^G \min \left(",
            r"\frac{\pi_\theta(o_i \mid q)}{",
            r"\pi_{\theta_{\text{old}}}",
            r"(",
            r"o_i ",
            r"\mid ",
            r"q",
            r")} "
            r"A_i",
            r",\text{clip} \left(",
            r"\frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)},",
            r"1 - \epsilon, 1 + \epsilon",
            r"\right) A_i",
            r"\right)",
            r"- \beta D_{KL}(\pi_\theta \| \pi_{\text{ref}})",
            r"\right]"
        ).scale(0.5)
        self.add(equation)

        self.play(Indicate(equation[1], color=YELLOW))
        self.wait(3)

class GRPOScene3(Scene):
    def construct(self):
        title = Text("GRPO (Group Relative Policy Optimization)", font_size=24).to_edge(UP)
        self.add(title)

        # Main Equation
        equation = MathTex(
            r"J_{\text{GRPO}}(\theta) = ",
            r"\mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(O \mid q)}",
            r"\left[",
            r"\frac{1}{G}\sum_{i=1}^G",
            r"\min \left(",
            r"\frac{\pi_\theta(o_i \mid q)}{",
            r"\pi_{\theta_{\text{old}}}",
            r"(",
            r"o_i ",
            r"\mid ",
            r"q",
            r")} "
            r"A_i",
            r",\text{clip} \left(",
            r"\frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)},",
            r"1 - \epsilon, 1 + \epsilon",
            r"\right) A_i",
            r"\right)",
            r"- \beta D_{KL}(\pi_\theta \| \pi_{\text{ref}})",
            r"\right]"
        ).scale(0.5)
        self.play(Write(equation), run_time=1)

        self.play(Indicate(equation[3], color=YELLOW))
        self.wait(6)

        min_expr = VGroup(
        equation[4],  # \min \left(
        equation[5],  # \frac{\pi_\theta(o_i \mid q)}{
        equation[6],  # \pi_{\theta_{\text{old}}}
        equation[7],  # (
        equation[8],  # o_i
        equation[9],  # \mid
        equation[10], # q
        equation[11], # ) A_i
        equation[12], # ,\text{clip} \left(
        equation[13], # \frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)},
        equation[14], # 1 - \epsilon, 1 + \epsilon
        equation[15], # \right) A_i
        )   

        self.play(Indicate(min_expr, color=YELLOW))
        self.play(Indicate(equation[17], color=YELLOW))

        self.play(Indicate(min_expr, color=YELLOW, run_time=11))


class GRPOScene4(Scene):
    def construct(self):
        title = Text("GRPO (Group Relative Policy Optimization)", font_size=24).to_edge(UP)
        self.add(title)

        # Main Equation
        equation = MathTex(
            r"J_{\text{GRPO}}(\theta) = ",
            r"\mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(O \mid q)}",
            r"\left[",
            r"\frac{1}{G}\sum_{i=1}^G",
            r"\min \left(",
            r"\frac{\pi_\theta(o_i \mid q)}{",
            r"\pi_{\theta_{\text{old}}}(o_i \mid q)}"
            r"A_i",
            r",\text{clip} \left(",
            r"\frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)},",
            r"1 - \epsilon, 1 + \epsilon",
            r"\right)",
            r"A_i",
            r"\right)",
            r"- \beta D_{KL}(\pi_\theta \| \pi_{\text{ref}})",
            r"\right]"
        ).scale(0.5)
        self.play(Write(equation), run_time=1)
        self.play(equation.animate.shift(UP * 1.5))

        original_advantage = equation[11].copy()
        self.play(equation[11].animate.move_to(ORIGIN).scale(1.8))
        self.wait(0.5)
        self.play(Transform(
            equation[11],
            MathTex(
                r"A_i = \frac{r_i - \text{mean}\left(\{r_1, r_2, \ldots, r_G\}\right)}{\text{std}\left(\{r_1, r_2, \ldots, r_G\}\right)}",
                color=WHITE
            )
        ))
        self.wait(4)
        self.play(Transform(equation[11], original_advantage))
        self.wait(2)

class GRPORegularization(Scene):
    def construct(self):
        title = Text("GRPO (Group Relative Policy Optimization)", font_size=24).to_edge(UP)
        self.add(title)

        # Main Equation
        equation = MathTex(
            r"J_{\text{GRPO}}(\theta) = ",
            r"\mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(O \mid q)}",
            r"\left[",
            r"\frac{1}{G}\sum_{i=1}^G",
            r"\min \left(",
            r"\frac{\pi_\theta(o_i \mid q)}{",
            r"\pi_{\theta_{\text{old}}}(o_i \mid q)}"
            r"A_i",
            r",\text{clip} \left(",
            r"\frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)},",
            r"1 - \epsilon, 1 + \epsilon",
            r"\right)",
            r"A_i",
            r"\right)-",
            r"\beta D_{KL}(\pi_\theta \| \pi_{\text{ref}})",
            r"\right]"
        ).scale(0.5)
        self.play(Write(equation), run_time=1)
        self.play(equation.animate.shift(UP * 1.5))

        original_advantage = equation[13].copy()
        self.play(equation[13].animate.move_to(ORIGIN).scale(1.8))
        self.wait(5)
        self.play(Transform(
            equation[13],
            MathTex(
                r"\mathbb{D}_{KL}(",
                r"\pi_\theta \| \pi_{\text{ref}}",
                r") = ",
                r"\frac{\pi_{\text{ref}}(o_i \mid q)}{\pi_\theta(o_i \mid q)}",
                r"- \log \left(",
                r"\frac{\pi_{\text{ref}}(o_i \mid q)}{\pi_\theta(o_i \mid q)}",
                r"\right)",
                r"- 1",
                color=WHITE
            )
        ))
        self.wait(16)
        self.play(Transform(equation[13], original_advantage))
        self.wait(1)

        ref_policy = MathTex("\pi_{\\text{ref}}", color=WHITE).shift(RIGHT * 3)
        new_policy = MathTex("\pi_{\\theta}", color=WHITE).shift(LEFT * 3)

        self.play(FadeIn(ref_policy), FadeIn(new_policy))

        self.play(
            new_policy.animate.shift(RIGHT * 5),
            run_time=1.5
        )

        self.wait(2)

        self.play(FadeOut(ref_policy), FadeOut(new_policy))

        self.wait(1)

class QuestionDistribution(Scene):
    def construct(self):
        dist = Axes(
            x_range=[0, 10, 1], y_range=[0, 1, 0.2],
            axis_config={"include_tip": False},
        )
        curve = dist.plot(lambda x: 0.2 * np.exp(-0.2 * x), color=BLUE)
        label = Text("P(Q): Distribution over Questions").to_edge(UP)

        self.play(Create(dist), Create(curve), FadeIn(label))

        for i in range(3):
            x_val = i * 2 + 1 
            prob_val = 0.2 * np.exp(-0.2 * x_val)
            question = Text(f"Q{i+1}", font_size=36).move_to(dist.coords_to_point(x_val, -0.1))
            prob_text = Text(f"{prob_val:.2f}", font_size=24).move_to(dist.coords_to_point(x_val, prob_val + 0.05))

            self.play(FadeIn(question))
            self.play(FadeIn(prob_text))
            self.wait(0.3)
        self.wait(2)

class OutcomeGivenQuestion(Scene):
    def construct(self):
        q = Text("Q1", font_size=36).move_to(LEFT * 3)
        dist_box = Square(color=GREEN).scale(1).next_to(q, RIGHT, buff=1.5)
        outcomes = VGroup(*[Text(f"o{i+1}", font_size=24) for i in range(3)])
        outcomes.arrange(DOWN, buff=0.3).move_to(dist_box)

        arrow = Arrow(start=q.get_right(), end=dist_box.get_left(), buff=0.1)

        label = MathTex(r"\pi_{\theta_{\text{old}}}(O \mid \text{Q1})", font_size=24).next_to(dist_box, UP)

        self.play(FadeIn(q), Create(arrow), Create(dist_box), FadeIn(outcomes), FadeIn(label))
        self.wait(0.5)
        sampled = outcomes[1].copy().set_color(YELLOW).shift(DOWN * 2)
        self.play(sampled.animate.move_to(RIGHT * 3), run_time=1.0)
        
        self.play(Transform(q, Text("Blueberry \n colour?", font_size=36).move_to(LEFT * 4)))
        self.play(Transform(sampled, Text("Blue", font_size=36).move_to(RIGHT * 3)))

        self.wait(2)

class ClipObjective(Scene):
    def construct(self):
        epsilon = 0.2
        A = 1.0
        r_min = 0.5
        r_max = 1.5

        axes = Axes(
            x_range=[r_min, r_max, 0.1],
            y_range=[0, 2, 0.2],
            x_length=8,
            y_length=4,
            axis_config={"include_numbers": True},
        )
        axes_labels = axes.get_axis_labels(x_label="ratio \, r", y_label="Objective")

        self.play(Create(axes), Write(axes_labels))

        unclipped = axes.plot(lambda r: r * A, color=BLUE, x_range=[r_min, r_max])
        unclipped_label = axes.get_graph_label(unclipped, label="r \\times A", x_val=1.45, direction=UP)

        def clipped_func(r):
            clipped_r = np.clip(r, 1 - epsilon, 1 + epsilon)
            return clipped_r * A

        clipped = axes.plot(clipped_func, color=RED)
        clipped_label = axes.get_graph_label(clipped, label="clip(r) \\times A", x_val=1.45, direction=DOWN)

        epsilon_lines = VGroup(
            axes.get_vertical_line(axes.input_to_graph_point(1 - epsilon, unclipped), color=YELLOW, line_func=DashedLine),
            axes.get_vertical_line(axes.input_to_graph_point(1 + epsilon, unclipped), color=YELLOW, line_func=DashedLine),
        )
        epsilon_labels = VGroup(
            MathTex("1 - \epsilon", font_size=24).next_to(epsilon_lines[0], UP),
            MathTex("1 + \epsilon", font_size=24).next_to(epsilon_lines[1], UP)
        )

        self.play(Create(unclipped), Write(unclipped_label))
        self.play(Create(clipped), Write(clipped_label))
        self.play(Create(epsilon_lines), Write(epsilon_labels))

        shaded_area = axes.get_area(clipped, x_range=[r_min, r_max], color=RED, opacity=0.3)
        self.play(FadeIn(shaded_area))
        self.wait(2)

        caption = MathTex("min(r \\times A, clip(r) \\times A)", font_size=32).to_edge(DOWN)
        self.play(Write(caption))
        self.wait(3)

class ClippingPreventsOutliers(Scene):
    def construct(self):
        epsilon = 0.2
        A = 1.0
        r_min, r_max = 0.5, 1.7

        axes = Axes(
            x_range=[r_min, r_max, 0.1],
            y_range=[0, 2, 0.2],
            x_length=8,
            y_length=4,
            axis_config={"include_numbers": True},
        )
        axes_labels = axes.get_axis_labels("ratio", "Objective")
        self.play(Create(axes), Write(axes_labels))

        unclipped = axes.plot(lambda r: r * A, color=BLUE)
        unclipped_label = axes.get_graph_label(unclipped, label="r · A", x_val=1.65, direction=UP)

        def clipped_func(r): return np.clip(r, 1 - epsilon, 1 + epsilon) * A
        clipped = axes.plot(clipped_func, color=RED)
        clipped_label = axes.get_graph_label(clipped, label="clip(r) · A", x_val=1.65, direction=DOWN)

        self.play(Create(unclipped), Write(unclipped_label))
        self.play(Create(clipped), Write(clipped_label))

        left_line = axes.get_vertical_line(axes.input_to_graph_point(1 - epsilon, unclipped), color=YELLOW, line_func=DashedLine)
        right_line = axes.get_vertical_line(axes.input_to_graph_point(1 + epsilon, unclipped), color=YELLOW, line_func=DashedLine)
        epsilon_lines = VGroup(left_line, right_line)

        epsilon_labels = VGroup(
            MathTex("1 - \\epsilon").next_to(left_line, UP),
            MathTex("1 + \\epsilon").next_to(right_line, UP)
        )

        self.play(Create(epsilon_lines), Write(epsilon_labels))

        outlier_area = VGroup(
            axes.get_area(unclipped, x_range=[1 + epsilon, r_max], color=BLUE, opacity=0.2),
            axes.get_area(unclipped, x_range=[r_min, 1 - epsilon], color=BLUE, opacity=0.2)
        )
        outlier_label = Text("Outlier Region", font_size=24).next_to(axes.c2p(1.55, 1.55), UP)
        self.play(FadeIn(outlier_area), FadeIn(outlier_label))

        dot = Dot(color=WHITE).move_to(axes.c2p(1.0, A))
        self.play(FadeIn(dot))

        self.play(dot.animate.move_to(axes.c2p(1.6, 1.6)), run_time=2)
        self.play(Indicate(clipped_label, color=RED))
        self.wait(1)

class QuickGRPOSummary(Scene):
    def construct(self):
        title = Text("GRPO (Group Relative Policy Optimization)", font_size=24).to_edge(UP)
        self.add(title)

        # Main Equation
        equation = MathTex(
            r"J_{\text{GRPO}}(\theta) = ",
            r"\mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(O \mid q)}",
            r"\left[\frac{1}{G} \sum_{i=1}^G \min \left(",
            r"\frac{\pi_\theta(o_i \mid q)}{",
            r"\pi_{\theta_{\text{old}}}",
            r"(",
            r"o_i ",
            r"\mid ",
            r"q",
            r")} "
            r"A_i",
            r",\text{clip} \left(",
            r"\frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)},",
            r"1 - \epsilon, 1 + \epsilon",
            r"\right) A_i",
            r"\right)",
            r"- \beta D_{KL}(\pi_\theta \| \pi_{\text{ref}})",
            r"\right]"
        ).scale(0.5)
        self.add(equation.shift(UP * 1.5))


        summary1 = Text("Expected average over questions & old outputs", font_size=32)
        summary2 = Text("Clipped policy ratio × Group Advantage", font_size=32)
        summary3 = Text("− KL Divergence Penalty", font_size=32)

        self.play(Write(summary1, run_time=2), Indicate(equation[2], color=YELLOW))
        self.wait(2.5)
        self.play(FadeOut(summary1))

        self.play(Write(summary2, run_time=2), Indicate(VGroup(*equation[3:15]), color=YELLOW))
        self.play(FadeOut(summary2))

        self.wait(1)
        self.play(Write(summary3, run_time=1), Indicate(equation[15], color=RED))
        self.play(FadeOut(summary3))
        
        self.wait(3)
