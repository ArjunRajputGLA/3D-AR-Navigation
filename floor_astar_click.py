import cv2
import numpy as np
import matplotlib.pyplot as plt
import heapq
import math
from collections import deque
from matplotlib.animation import FuncAnimation

# -----------------------------
# A* Algorithm
# -----------------------------
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid, start, goal):
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        r, c = current
        neighbors = [
            (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1),
            (r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)
        ]

        for nr, nc in neighbors:
            if 0 <= nr < rows and 0 <= nc < cols:
                if grid[nr, nc] == 1:
                    continue

                move_cost = math.sqrt(2) if (nr != r and nc != c) else 1.0
                tentative_g = g_score[current] + move_cost
                neighbor = (nr, nc)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))

    return None


# -----------------------------
# Connected component + snapping
# -----------------------------
def largest_walkable_component(grid):
    rows, cols = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    best_component = []
    best_size = 0

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 0 and not visited[r, c]:
                q = deque([(r, c)])
                visited[r, c] = True
                component = [(r, c)]

                while q:
                    cr, cc = q.popleft()
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if grid[nr, nc] == 0 and not visited[nr, nc]:
                                visited[nr, nc] = True
                                q.append((nr, nc))
                                component.append((nr, nc))

                if len(component) > best_size:
                    best_size = len(component)
                    best_component = component

    mask = np.zeros_like(grid, dtype=bool)
    for (r, c) in best_component:
        mask[r, c] = True
    return mask


def snap_to_walkable(grid, pt, mask=None, max_radius=300):
    r0, c0 = pt
    rows, cols = grid.shape

    def valid(r, c):
        if not (0 <= r < rows and 0 <= c < cols):
            return False
        if grid[r, c] != 0:
            return False
        if mask is not None and not mask[r, c]:
            return False
        return True

    if valid(r0, c0):
        return (r0, c0)

    for rad in range(1, max_radius + 1):
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                r = r0 + dr
                c = c0 + dc
                if valid(r, c):
                    return (r, c)

    return pt


# -----------------------------
# Build obstacle-aware grid
# -----------------------------
def build_walkable_grid_smart(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    bin_img = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5
    )

    inv = 255 - bin_img

    # thick walls
    wall_kernel = np.ones((7, 7), np.uint8)
    walls = cv2.erode(inv, wall_kernel, iterations=1)
    walls = cv2.dilate(walls, wall_kernel, iterations=1)

    # thin obstacles (furniture/text)
    small = cv2.subtract(inv, walls)
    small = cv2.morphologyEx(small, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    obstacles = cv2.bitwise_or(walls, small)

    # safety buffer
    obstacles = cv2.dilate(obstacles, np.ones((5, 5), np.uint8), iterations=1)

    grid = np.where(obstacles > 0, 1, 0).astype(np.uint8)

    # connect corridors
    walkable = (grid == 0).astype(np.uint8) * 255
    walkable = cv2.morphologyEx(walkable, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1)
    grid = np.where(walkable == 255, 0, 1).astype(np.uint8)

    return grid


# -----------------------------
# Smoothing (string pulling)
# -----------------------------
def line_is_clear(grid, p1, p2):
    r1, c1 = p1
    r2, c2 = p2
    steps = int(max(abs(r2 - r1), abs(c2 - c1)))
    if steps == 0:
        return True
    for i in range(steps + 1):
        t = i / steps
        r = int(round(r1 + (r2 - r1) * t))
        c = int(round(c1 + (c2 - c1) * t))
        if grid[r, c] == 1:
            return False
    return True


def smooth_path(grid, path):
    if not path or len(path) < 3:
        return path

    smoothed = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            if line_is_clear(grid, path[i], path[j]):
                break
            j -= 1
        smoothed.append(path[j])
        i = j
    return smoothed


# -----------------------------
# Interpolation for animation
# -----------------------------
def interpolate_path(points, step=2.0):
    dense = []
    for i in range(len(points) - 1):
        r1, c1 = points[i]
        r2, c2 = points[i + 1]
        dist = math.sqrt((r2 - r1) ** 2 + (c2 - c1) ** 2)
        n = max(2, int(dist / step))
        for k in range(n):
            t = k / (n - 1)
            r = r1 + (r2 - r1) * t
            c = c1 + (c2 - c1) * t
            dense.append((r, c))
    return dense


# -----------------------------
# Turn-by-turn directions
# -----------------------------
def angle_between(v1, v2):
    x1, y1 = v1
    x2, y2 = v2
    a1 = math.atan2(y1, x1)
    a2 = math.atan2(y2, x2)
    ang = math.degrees(a2 - a1)
    while ang > 180:
        ang -= 360
    while ang < -180:
        ang += 360
    return ang


def path_directions(smoothed_path, meters_per_pixel, turn_threshold_deg=25):
    if len(smoothed_path) < 2:
        return []

    pts = [(p[1], p[0]) for p in smoothed_path]  # (x,y)

    def dist_m(a, b):
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        return math.sqrt(dx * dx + dy * dy) * meters_per_pixel

    instructions = []
    total_straight = 0.0

    for i in range(1, len(pts) - 1):
        prev = pts[i - 1]
        curr = pts[i]
        nxt = pts[i + 1]

        v1 = (curr[0] - prev[0], curr[1] - prev[1])
        v2 = (nxt[0] - curr[0], nxt[1] - curr[1])

        total_straight += dist_m(prev, curr)
        ang = angle_between(v1, v2)

        if abs(ang) >= turn_threshold_deg:
            if total_straight > 0.2:
                instructions.append(f"Go straight {total_straight:.1f} m")
            instructions.append("Turn left" if ang > 0 else "Turn right")
            total_straight = 0.0

    total_straight += dist_m(pts[-2], pts[-1])
    if total_straight > 0.2:
        instructions.append(f"Go straight {total_straight:.1f} m")

    instructions.append("You have arrived ✅")
    return instructions


# -----------------------------
# MAIN (wrapped in function)
# -----------------------------
def main():
    IMAGE_PATH = "floor_map.png"

    img_bgr = cv2.imread(IMAGE_PATH)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]

    # Click start + goal
    clicks = []

    def onclick(event):
        if event.xdata is None or event.ydata is None:
            return
        x, y = int(event.xdata), int(event.ydata)
        clicks.append((y, x))
        if len(clicks) == 1:
            print(f"Start selected at (row={y}, col={x})")
        elif len(clicks) == 2:
            print(f"Goal selected at (row={y}, col={x})")
            plt.close()

    fig, ax = plt.subplots(figsize=(7, 10))
    ax.imshow(img_rgb)
    ax.set_title("Click START point, then click GOAL point")
    plt.axis("off")
    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()

    if len(clicks) < 2:
        raise Exception("You must click 2 points.")

    raw_start, raw_goal = clicks[0], clicks[1]

    # Backend grid + A*
    grid = build_walkable_grid_smart(img_bgr)
    mask = largest_walkable_component(grid)

    start = snap_to_walkable(grid, raw_start, mask=mask)
    goal = snap_to_walkable(grid, raw_goal, mask=mask)

    path = astar(grid, start, goal)
    if path is None:
        print("❌ No path found. Try clicking in open corridor/living area.")
        return

    smoothed = smooth_path(grid, path)

    # scale: 7000mm = 7m
    REAL_WIDTH_METERS = 7.0
    meters_per_pixel = REAL_WIDTH_METERS / W

    instructions = path_directions(smoothed, meters_per_pixel)

    print("\n========= TURN BY TURN DIRECTIONS =========")
    for i, ins in enumerate(instructions, 1):
        print(f"{i}. {ins}")
    print("==========================================\n")

    dense = interpolate_path(smoothed, step=2.0)

    # Animation plot
    fig, ax = plt.subplots(figsize=(7, 10))
    ax.imshow(img_rgb)
    ax.set_title("Indoor Navigation Animation")
    ax.axis("off")

    path_xy = np.array([(p[1], p[0]) for p in smoothed], dtype=np.int32)
    ax.plot(path_xy[:, 0], path_xy[:, 1], linewidth=3)

    ax.scatter([start[1]], [start[0]], s=150, marker="o", label="Start")
    ax.scatter([goal[1]], [goal[0]], s=150, marker="X", label="Goal")

    dot = ax.scatter([], [], s=120)
    text = ax.text(
        20, 40, "",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="black")
    )

    segment_idx = 0
    instruction_idx = 0

    def update(frame):
        nonlocal segment_idx, instruction_idx

        r, c = dense[frame]
        dot.set_offsets([[c, r]])

        if instruction_idx < len(instructions):
            text.set_text(f"Step {instruction_idx+1}/{len(instructions)}:\n{instructions[instruction_idx]}")

        # advance when reaching next waypoint
        if segment_idx < len(smoothed) - 1:
            target = smoothed[segment_idx + 1]
            if abs(r - target[0]) < 4 and abs(c - target[1]) < 4:
                segment_idx += 1
                if instruction_idx < len(instructions) - 1:
                    instruction_idx += 1

        return dot, text

    FuncAnimation(fig, update, frames=len(dense), interval=30, blit=False, repeat=False)
    plt.show()


if __name__ == "__main__":
    main()
