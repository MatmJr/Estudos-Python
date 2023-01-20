from manim import *
import numpy as np
from math import gamma


class CreateGraph(Scene):
    def construct(self):
        axes = Axes(
            x_range=[0.1, 5],
            y_range=[0, 8],
            axis_config={"color": BLUE},
        )

        # Create Graph
        graph = axes.plot(lambda x: 1, color=WHITE)
        graph_label = axes.get_graph_label(graph, label='O(1)')

        graph1 = axes.plot(lambda x: np.log(x+1), color=WHITE)
        graph_label1 = axes.get_graph_label(graph1, label='O(log(n)')

        graph2 = axes.plot(lambda x: x, color=WHITE)
        graph_label2 = axes.get_graph_label(graph2, label='O(n)')

        graph3 = axes.plot(lambda x: x*np.log(x)+0.3, color=WHITE)
        graph_label3 = axes.get_graph_label(graph3, label='O(n log(n))')

        graph4 = axes.plot(lambda x: x**2, color=WHITE)
        graph_label4 = axes.get_graph_label(graph4, label='O(n^{2})')

        graph5 = axes.plot(lambda x: x**3, color=WHITE)
        graph_label5 = axes.get_graph_label(graph5, label='O(n^{3})')

        graph6 = axes.plot(lambda x: 4**x-1, color=WHITE)
        graph_label6 = axes.get_graph_label(graph6, label='O(2^{n})')

        graph7 = axes.plot(lambda x: gamma(x+4)-6.8, color=WHITE)
        graph_label7 = axes.get_graph_label(graph6, label='O(n!)')

        # Display graph
        self.play(Create(axes), Create(graph), Write(graph_label))
        self.wait(2)
        self.play(ReplacementTransform(graph, graph1),
                  FadeTransform(graph_label, graph_label1))
        self.wait(2)
        self.play(ReplacementTransform(graph1, graph2),
                  FadeTransform(graph_label1, graph_label2))
        self.wait(2)
        self.play(ReplacementTransform(graph2, graph3),
                  FadeTransform(graph_label2, graph_label3))
        self.wait(2)
        self.play(ReplacementTransform(graph3, graph4),
                  FadeTransform(graph_label3, graph_label4))
        self.wait(2)
        self.play(ReplacementTransform(graph4, graph5),
                  FadeTransform(graph_label4, graph_label5))
        self.wait(2)
        self.play(ReplacementTransform(graph5, graph6),
                  FadeTransform(graph_label5, graph_label6))
        self.wait(2)
        self.play(ReplacementTransform(graph6, graph7),
                  FadeTransform(graph_label6, graph_label7))
