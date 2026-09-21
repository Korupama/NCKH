"""Metric football-pitch drawing primitives for Stage 4 QA exports."""

from __future__ import annotations

from matplotlib.patches import Arc, Circle, Rectangle


def draw_metric_pitch(
    ax,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    line_color: str = "#4a4a4a",
    line_width: float = 1.2,
    pitch_facecolor: str = "#f7fbf5",
    show_axes_guides: bool = False,
    show_goals: bool = True,
) -> None:
    """Draw a regulation pitch in Stage 4's pitch-centred metric frame.

    ``X`` runs goal-to-goal and ``Y`` runs touchline-to-touchline.  The
    drawing is deliberately independent of pose data so it can be rendered
    first, behind all track markers and labels.
    """

    half_length = pitch_length / 2.0
    half_width = pitch_width / 2.0
    penalty_depth = 16.5
    penalty_half_width = 40.32 / 2.0
    goal_area_depth = 5.5
    goal_area_half_width = 18.32 / 2.0
    centre_circle_radius = 9.15
    penalty_mark_distance = 11.0
    goal_half_width = 7.32 / 2.0
    goal_depth = 2.0  # A visual aid; this is not a modelled goal volume.

    ax.set_facecolor(pitch_facecolor)

    def add_rectangle(x: float, y: float, width: float, height: float) -> None:
        ax.add_patch(
            Rectangle(
                (x, y),
                width,
                height,
                fill=False,
                edgecolor=line_color,
                linewidth=line_width,
                zorder=1,
            )
        )

    # Boundary, halfway line, and centre markings.
    add_rectangle(-half_length, -half_width, pitch_length, pitch_width)
    ax.plot([0.0, 0.0], [-half_width, half_width], color=line_color, linewidth=line_width, zorder=1)
    ax.add_patch(
        Circle(
            (0.0, 0.0),
            centre_circle_radius,
            fill=False,
            edgecolor=line_color,
            linewidth=line_width,
            zorder=1,
        )
    )
    ax.scatter([0.0], [0.0], s=18, color=line_color, zorder=2)

    # Penalty and goal areas at both ends.
    add_rectangle(-half_length, -penalty_half_width, penalty_depth, 2.0 * penalty_half_width)
    add_rectangle(half_length - penalty_depth, -penalty_half_width, penalty_depth, 2.0 * penalty_half_width)
    add_rectangle(-half_length, -goal_area_half_width, goal_area_depth, 2.0 * goal_area_half_width)
    add_rectangle(half_length - goal_area_depth, -goal_area_half_width, goal_area_depth, 2.0 * goal_area_half_width)

    left_penalty_mark = (-half_length + penalty_mark_distance, 0.0)
    right_penalty_mark = (half_length - penalty_mark_distance, 0.0)
    ax.scatter(
        [left_penalty_mark[0], right_penalty_mark[0]],
        [left_penalty_mark[1], right_penalty_mark[1]],
        s=18,
        color=line_color,
        zorder=2,
    )
    ax.add_patch(
        Arc(
            left_penalty_mark,
            width=2.0 * centre_circle_radius,
            height=2.0 * centre_circle_radius,
            theta1=310,
            theta2=50,
            color=line_color,
            linewidth=line_width,
            zorder=1,
        )
    )
    ax.add_patch(
        Arc(
            right_penalty_mark,
            width=2.0 * centre_circle_radius,
            height=2.0 * centre_circle_radius,
            theta1=130,
            theta2=230,
            color=line_color,
            linewidth=line_width,
            zorder=1,
        )
    )

    if show_goals:
        add_rectangle(-half_length - goal_depth, -goal_half_width, goal_depth, 2.0 * goal_half_width)
        add_rectangle(half_length, -goal_half_width, goal_depth, 2.0 * goal_half_width)

    if show_axes_guides:
        ax.axhline(0.0, color="#999999", linewidth=0.8, linestyle="--", alpha=0.5, zorder=0)
        ax.axvline(0.0, color="#999999", linewidth=0.8, linestyle="--", alpha=0.5, zorder=0)

    ax.set_xlim(-half_length - 3.0, half_length + 3.0)
    ax.set_ylim(-half_width - 2.0, half_width + 2.0)
    ax.set_aspect("equal", adjustable="box")
