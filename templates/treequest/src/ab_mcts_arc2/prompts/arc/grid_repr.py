from ab_mcts_arc2.data_types import Grid


def list_format(grid: Grid) -> str:
    shape_lst = []
    now = grid
    while isinstance(now, list):
        shape_lst.append(str(len(now)))
        if len(now) == 0:
            break
        now = now[0]

    if len(shape_lst) == 0:
        shape_str = "None"
    else:
        shape_str = "(" + ", ".join(shape_lst) + ")"

    if grid is None:
        grid_str = "None"
    elif len(grid) == 0:
        grid_str = "[[]]\n"
    else:
        grid_str = "[\n"
        for row in grid:
            grid_str += "    " + str(row) + ",\n"
        grid_str += "]\n"

    return f"""
Shape: {shape_str}
{grid_str}
"""
