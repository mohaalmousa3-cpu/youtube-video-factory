"""Phase 1 proof-of-concept scene: a stick figure that raises one arm.
Real scenes will be generated per script beat later — this just proves the
AnimationProvider path (Manim -> mp4) works end to end."""
from manim import Circle, Create, DOWN, Line, LEFT, ORIGIN, RIGHT, Scene, UP, VGroup


class StickmanDemo(Scene):
    def construct(self):
        head = Circle(radius=0.4).move_to(UP * 1.6)
        body = Line(UP * 1.2, DOWN * 0.5)
        left_leg = Line(DOWN * 0.5, DOWN * 1.3 + LEFT * 0.4)
        right_leg = Line(DOWN * 0.5, DOWN * 1.3 + RIGHT * 0.4)
        left_arm = Line(UP * 0.9, DOWN * 0.2 + LEFT * 0.7)
        right_arm = Line(UP * 0.9, ORIGIN)

        stickman = VGroup(head, body, left_leg, right_leg, left_arm, right_arm)
        self.play(*[Create(m) for m in stickman])
        self.play(right_arm.animate.put_start_and_end_on(UP * 0.9, UP * 1.6 + RIGHT * 0.8), run_time=0.6)
        self.wait(0.5)
