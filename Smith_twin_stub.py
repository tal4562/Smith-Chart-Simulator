import skrf as rf
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import ttk

# === Globals for circles ===
pi, Radius = np.pi, 1.11827  # Smith chart radius on the plot
t = np.linspace(0, 2 * pi, 1000)
sine, cosine = np.sin(t), np.cos(t)

# === Globals for figure ===
fig, ax, gamma, gamma_print, Z, Z_print,Ydif = None, None, np.ndarray, str, complex, str, np.array([complex,complex])
plotted_points, plotted_circles,plotted_arcs = [], [], [] # global list to store all plotted points, circles and arcs

# === Remove Functions ===
# Remove circle function
def pop_first_circle() -> None:
    """Remove the first plotted circle."""
    if not plotted_circles:
        print("No circles to remove.")
        return

    circle_artist = plotted_circles.pop(0)
    try:
        circle_artist.remove()
    except Exception as e:
        print(f"⚠️ Could not remove circle: {e}")

    plt.draw()


def pop_first_arc() -> None:
    """Remove the first plotted circle."""
    if not plotted_arcs:
        print("No arcs to remove.")
        return

    arc_artist = plotted_arcs.pop(0)
    try:
        arc_artist.remove()
    except Exception as e:
        print(f"⚠️ Could not remove arc: {e}")

    # Safely update legend
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels)
    else:
        # Remove legend if no handles
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    plt.draw()


# Remove text, line and point
def pop_first_point() -> None:
    """Remove the first plotted point, line, and text from the axes and update legend safely."""
    global plotted_points
    if not plotted_points:
        print("No points to remove.")
        return

    point_z, line_z, text_z, point_lambda, text_lambda = plotted_points.pop(0)

    for artist in [point_z, line_z, text_z, point_lambda, text_lambda]:
        try:
            artist.remove()
        except Exception as e:
            print(f"⚠️ Failed to remove {artist}: {e}")

    # Safely update legend
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels)
    else:
        # Remove legend if no handles
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    plt.draw()

### allows to insert 1/2 and not only 0.5
def parse_fraction(s):
    s = s.strip()
    if "/" in s:
        num, den = s.split("/", 1)
        return float(num) / float(den)
    else:
        return float(s)


# Throws warnings as needed
def input_func() -> None:
    global fig, ax, gamma, Z, Z_print, plotted_circles, plotted_points
    if fig is None or ax is None:
        print("⚠️ Please open a Smith chart first.")
        return
    if gamma is None:
        print("First plot an impedance.")
        return


#################################### Main Functions ####################################
# First Function: Converts a normalized impedance → Γ
def impedance_to_gamma(z_in: complex) -> list[np.ndarray | str]:
    gama = (z_in - 1) / (z_in + 1)
    theta = np.angle(gama)
    theta = theta + 2 * pi if theta < 0 else theta
    return [np.array([np.real(gama), np.imag(gama), theta, np.abs(gama)]), str(np.round(gama, 3))]


# Second Function: Converts Γ → normalized impedance
def gamma_to_impedance(gama: complex) -> list[complex | str]:
    Z_in_norm = (1 + gama) / (1 - gama)
    return [Z_in_norm, str(np.round(Z_in_norm, 3))]


# Third Function: Rotate Γ toward generator. The generator is located at z = l < 0 and the load is at z = 0.
# To rotate towards the load insert z = l > 0
def rotate_gamma(gamma_ar, l: float) -> list[np.ndarray | str]:
    # since abs(last_gamma) = constant we simply gain a phase.
    # converting from wavelengths to radians is as follows:
    # 0.5 [wavelength] = 2 * pi [rad] - > angle / 0.5 [wavelength] = angle / 2 * pi [rad]
    # angle [rad] = 4 * pi * the distance l
    gamma_rot = gamma_ar[1] * np.exp(1j * (gamma_ar[0] + 4 * np.pi * l))
    return [np.array([gamma_rot.real, gamma_rot.imag, np.angle(gamma_rot), gamma_ar[1]]), str(np.round(gamma_rot, 3))]


# Fourth Function: Plot the clockwise arc and returns the arc length
def clock_wise_arc(start: float, end: float, arc_num: int, col: str, name: str, arcs=None) -> float:
    if arcs is None:
        arcs = plotted_arcs

    # --- CLOCKWISE rotation handling ---
    start = start % (2 * np.pi)
    end = end % (2 * np.pi)

    # clockwise angular distance (always positive)
    ang_cw = (start - end) % (2 * np.pi)

    # create angle array going clockwise (decreasing angle)
    t1 = np.linspace(start, start - ang_cw, 100, endpoint=True)

    # --- Compute wavelength equivalent ---
    ang2wl = ang_cw / (4 * np.pi)

    # --- Plot the clockwise arc ---
    (arc,) = ax.plot(arc_num * 0.4 * np.cos(t1), arc_num * 0.4 * np.sin(t1), color=col, linestyle='--', linewidth=3,
                     label=name + ' ' + str(np.round(ang2wl, 4)) + 'λ')
    arcs.append(arc)
    ax.legend()

    return ang2wl


# Fifth Function: plot Γ and its corresponding wavelength
def plot_gamma_point(ax, gama: np.ndarray, art: list[str], points=None) -> None:
    #  art = dot shape, line color, point name
    # gama = gama real, gama imaginary, gama phase
    if points is None:
        points = plotted_points

    # Converts the angle from radians to wavelength
    ang_graph = 0.25 * (1 - gama[2] / pi)
    ang_graph = 0.5 + ang_graph if ang_graph < 0 else ang_graph
    ang_graph = str(np.round(ang_graph, 4)) + 'λ'

    # Dynamic alignment depending on position angle
    ha = 'left' if np.cos(gama[2]) > 0 else 'right'
    va = 'bottom' if np.sin(gama[2]) > 0 else 'top'

    # Plots the point on the chart
    (point_z,) = ax.plot(gama[0], gama[1], markersize=8, marker=art[0], color=art[1], label=art[2] + ' : ' + Z_print)
    # Line from origin to circle radius direction
    (line_z,) = ax.plot([0, Radius * np.cos(gama[2])], [0, Radius * np.sin(gama[2])], color=art[1], linestyle='--')
    # Text label near the point
    text_z = ax.text(gama[0] + 0.02, gama[1] + 0.02, art[2], fontsize=14, weight='bold')

    # Plots the corresponding wavelength
    (point_lambda,) = ax.plot(Radius * np.cos(gama[2]), Radius * np.sin(gama[2]), markersize=5)
    text_lambda = ax.text(1.05 * Radius * np.cos(gama[2]), 1.05 * Radius * np.sin(gama[2]), ang_graph, fontsize=12,
                          weight='bold', ha=ha, va=va)

    # Append the point so it can be removed if needed
    points.append((point_z, line_z, text_z, point_lambda, text_lambda))

    # Prints to screen the
    print('Γ_' + art[2] + ' = ' + gamma_print + ', l = ' + ang_graph + ', ' + art[2] + ' = ' + Z_print)
    ax.legend()


# Sixth function: intersection of 2 circles
def circle_intersections_4nums(xc1, yc1, r1, xc2, r2) -> np.ndarray:
    """
    Find intersections of two circles:
    Circle1: center (xc1, yc1), radius r1
    Circle2: center (xc2, 0), radius r2
    Returns x1, y1, x2, y2 (both intersection points)
    """
    # Distance between centers
    dx = xc2 - xc1
    dy = -yc1  # second circle center y = 0
    d = np.sqrt(dx ** 2 + dy ** 2)

    # No intersection
    if d > r1 + r2 or d < abs(r1 - r2):
        print('No intersection')
        return np.zeros(4)

    # Distance from first circle center to line connecting intersections
    a = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
    h = np.sqrt(r1 ** 2 - a ** 2)

    # Point along line connecting centers
    x3 = xc1 + a * dx / d
    y3 = yc1 + a * dy / d

    # Intersection offsets
    rx = -dy * (h / d)
    ry = dx * (h / d)

    # Two intersection points
    x1, y1 = x3 + rx, y3 + ry
    x2, y2 = x3 - rx, y3 - ry

    return np.array([x1, y1, x2, y2])


#################################### Chart Functions ####################################
# === 0. Show Smith chart ===
def show_chart() -> None:
    print('A new Diagram Has Been Opened')
    global fig, ax
    plt.close('all')
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal', adjustable='box')

    if mode_Z.get():
        print('Current mode is impedance (Z).')
    else:
        print('Current mode is admittance (Y).')

    if mode_SC.get():
        print('Trying to match a short circuit.')
    else:
        print('Trying to match an open circuit.')

    # Load Smith chart
    background = plt.imread("Smith_Chart_600_dpi_square.PNG")
    ax.imshow(background, extent=(-1.157, 1.157, -1.157, 1.157), aspect='equal', origin='upper')

    # Smith chart overlay
    rf.plotting.smith(smithR=1, ax=ax, chart_type='z', draw_labels=True, border=False, ref_imm=1.0)

    # Outer wavelength circle
    ax.plot(Radius * cosine, Radius * sine, color='blue', linewidth=2)

    plt.tight_layout()
    plt.show(block=False)


# === 1. Plot an impedance on the chart ===
def on_plot_gamma() -> None:
    global fig, ax, gamma, gamma_print, Z, Z_print, plotted_circles
    if fig is None or ax is None:
        print("⚠️ Please open a Smith chart first.")
        return
    try:
        R, X = parse_fraction(entry_real.get()), parse_fraction(entry_imag.get())
    except ValueError:
        print("⚠️ Invalid input: enter numeric R, X")
        return

    print('Option One: New Load')

    # Calcs gamma and Z
    gamma, gamma_print = impedance_to_gamma(R + 1j * X)
    Z, Z_print = gamma_to_impedance(gamma[0] + gamma[1] * 1j)

    # plot the circle with radius = abs(gamma)
    cir, = ax.plot(gamma[3] * cosine, gamma[3] * sine, color='blue', linewidth=2)
    plotted_circles.append(cir)

    # plot Z_L / Y_L
    plot_gamma_point(ax, gamma[0:3], ['o', 'red', 'Z_L' if mode_Z.get() else 'Y_L'])

    plt.draw()


# === 2. Plot the rotated impedance on the chart ===
def on_rotate_gamma() -> None:
    global fig, ax, gamma, gamma_print, Z, Z_print
    print('Option Two: Rotation')
    input_func()
    try:
        d = parse_fraction(d_val.get())
    except ValueError:
        print("⚠️ Invalid input: enter a numeric d (the distance of the load from the first stub)")
        return

    # Rotates gamma and finds new Z
    gamma, gamma_print = rotate_gamma(gamma[2:], d)
    Z, Z_print = gamma_to_impedance(gamma[0] + 1j * gamma[1])

    # Plots new Z
    plot_gamma_point(ax, gamma[0:3], ['^', 'blue', 'Z_L_rot' if mode_Z.get() else 'Y_L_rot'])

    plt.draw()


# === 3. Plot the mirrored impedance on the chart ===
def on_mirror_gamma() -> None:
    global fig, ax, gamma, gamma_print, Z, Z_print, plotted_circles
    input_func()
    # Toggle mode_Z

    # negate gamma
    gamma[0], gamma[1], gamma[2] = -gamma[0], -gamma[1], np.arctan2(-gamma[1], -gamma[0])
    gamma[2] = 2 * pi + gamma[2] if gamma[2] < 0 else gamma[2]
    gamma_print = str(np.round(gamma[0] + 1j * gamma[1], 3))
    r_test = (1-0.3333) / 2
    x0_test = (1+0.3333) / 2
    if ((gamma[0] - x0_test) **2 + (gamma[1] ** 2)) < r_test ** 2:
        print('If Y_L inside this circle, can not match. This is the forbidden zone')
        plot_gamma_point(ax, gamma[0:3], ['D', 'green', 'Z_L_mir' if mode_Z.get() else 'Y_L_mir'])
        cir, = ax.plot(r_test * cosine + x0_test, r_test * sine, color='red', linewidth=3, linestyle = '--')
        plotted_circles.append(cir)
        plt.draw()
        return

    mode_Z.set(not mode_Z.get())



    if mode_Z.get():
        print('Option Three: Printing and working with admittance Z -> Y.')
    else:
        print('Option Three: Printing and working with impedance Y -> Z.')

    # plots new Z
    Z, Z_print = gamma_to_impedance(gamma[0] + 1j * gamma[1])
    plot_gamma_point(ax, gamma[0:3], ['D', 'green', 'Z_L_mir' if mode_Z.get() else 'Y_L_mir'])

    plt.draw()


# === 4. Plot Y1, rotated unity circles and find their intersections ===
def on_create_aux_circles_and_find_intersection():
    global fig, ax, gamma, gamma_print, Z, Z_print, Ydif, plotted_circles
    input_func()
    try:
        d2 = parse_fraction(d2_var.get()) # the distance between the stubs
    except ValueError:
        print("Invalid d2 value. Enter a numeric value.")
        return

    tmp_gamma, tmp_gamma_print, tmp_Z, tmp_Z_print = gamma, gamma_print, Z, Z_print

    # Plot the circle for Y1
    x_Y1 = (1 - gamma[0] ** 2 - gamma[1] ** 2) / (2 * (1 - gamma[0]))
    r_y1 = abs(1 - x_Y1)
    (cir,) = ax.plot(x_Y1 + r_y1 * cosine, r_y1 * sine,color = 'blue' , linewidth = 2)
    plotted_circles.append(cir)
    print(f'Y1 circle: centered at: ({x_Y1:.3f},0)')

    # Plot the rotated unity circle
    uni_par,_ = rotate_gamma(np.array([0,0.5]),d2)
    (unity_circle_line), = ax.plot(0.5 * cosine + uni_par[0], 0.5 * sine + uni_par[1], linewidth=2, color='blue')
    plotted_circles.append(unity_circle_line)
    print(f'Unity circle rotated by {d2}: center = ({uni_par[0]:.3f},{uni_par[1]:.3f})')

    # Find and plot intersection points
    p_int = circle_intersections_4nums(uni_par[0],uni_par[1],0.5,x_Y1,r_y1)

    # plot the intersection points
    for i in range(2):
        Z, Z_print = gamma_to_impedance(p_int[2*i] + 1j * p_int[2*i + 1])
        gamma, gamma_print = impedance_to_gamma(Z)
        plot_gamma_point(ax,gamma[0:3],['*', 'm', ('Z_int' if mode_Z.get() else 'Y_int') + str(i+1)])
        Ydif[i] = (Z - tmp_Z)

    gamma, gamma_print, Z, Z_print = tmp_gamma, tmp_gamma_print, tmp_Z, tmp_Z_print

    plt.draw()


# === 5. Plot Y_intersection - Y_L  this gives L1 ===
def on_subtract():
    global fig, ax, gamma, gamma_print,Z, Z_print,plotted_arcs, Ydif
    input_func()
    print('Step 5: Subtract to Find L1, the length of the first stub.')

    tmp_gamma, tmp_gamma_print, tmp_Z, tmp_Z_print = gamma, gamma_print, Z, Z_print

    for i in range(2):
        gamma, gamma_print = impedance_to_gamma(Ydif[i])
        Z, Z_print = gamma_to_impedance(gamma[0] + 1j * gamma[1])
        plot_gamma_point(ax, gamma[0:3],['X', 'orange','Ydif' + str(i + 1)])
        ang_cw = clock_wise_arc(0 if mode_SC.get() else pi, gamma[2],1+i,'y','L1 arc ' +str(i+1) + ' :' )
        print('The length of the first stub L1: ' + str(np.round(ang_cw, 4)) + 'λ')
    gamma, gamma_print, Z, Z_print = tmp_gamma, tmp_gamma_print, tmp_Z, tmp_Z_print

    plt.draw()


# === 6. Plot and rotate unity back this gives L2 ===
def on_rotate_unity_back():
    global fig, ax, gamma, gamma_print,Z, Z_print,plotted_arcs, Ydif
    input_func()
    try:
        d2 =  -parse_fraction(d2_var.get())
    except ValueError:
        print("Invalid d2 value. Enter a numeric value from step 4.")
        return
    print('Step 6: Rotate unity back to place to find L2.')

    tmp_gamma, tmp_gamma_print, tmp_Z, tmp_Z_print = gamma, gamma_print, Z, Z_print

    for i in range(2):
        # rotated points on the unity circle
        gamma, gamma_print = impedance_to_gamma(Ydif[i] + tmp_Z)
        gamma, gamma_print = rotate_gamma(gamma[2:4],d2)
        Z, Z_print = gamma_to_impedance(gamma[0] + 1j * gamma[1])
        plot_gamma_point(ax,gamma[0:3],['*','b','Y2_rot' +str(i+1)])

        # now stubs
        gamma, gamma_print = impedance_to_gamma(-1j *Z.imag)
        Z, Z_print = gamma_to_impedance(gamma[0] + 1j * gamma[1])
        plot_gamma_point(ax,gamma[0:3],['o','k','stub'+str(i+1)])

        # now arcs = stub lengths
        ang_cw = clock_wise_arc(0 if mode_SC.get() else pi, gamma[2],1+i,'y','L2 arc ' +str(i+1) + ' :' )
        print('The length of the first stub L1: ' + str(np.round(ang_cw, 4)) + 'λ')

    gamma, gamma_print, Z, Z_print = tmp_gamma, tmp_gamma_print, tmp_Z, tmp_Z_print

    plt.draw()


def on_remove_point():
    global fig, ax, plotted_points
    if fig is None or ax is None:
        print("⚠️ Please open a Smith chart first.")
        return
    if not plotted_points:
        print('Nothing to remove')
        return
    pop_first_point()
    plt.draw()


def on_remove_circle():
    global fig, ax, plotted_circles
    if fig is None or ax is None:
        print("⚠️ Please open a Smith chart first.")
        return
    if not plotted_circles:
        print('Nothing to remove')
        return
    pop_first_circle()
    plt.draw()


def on_remove_arc():
    global fig, ax, plotted_circles
    if fig is None or ax is None:
        print("⚠️ Please open a Smith chart first.")
        return
    if not plotted_arcs:
        print('Nothing to remove')
        return
    pop_first_arc()
    plt.draw()


#################################### GUI setup ####################################
root = tk.Tk()
mode_Z = tk.BooleanVar(value=True)
mode_SC = tk.BooleanVar(value=True)

# Main frame
frame = ttk.Frame(root, padding=20)
frame.grid(row=0, column=0, sticky="w")

# Prevent columns from stretching
for i in range(6):
    frame.columnconfigure(i, weight=0)

# 0. Show chart label and button
ttk.Label(frame, text="0. Open a New Chart", font=("Segoe UI", 12, "bold")) \
    .grid(column=0, row=0, sticky="w", pady=5)
ttk.Button(frame, text="New Chart", command=show_chart) \
    .grid(column=1, row=0, sticky="w", padx=(20, 0), pady=5)

# Checkboxes
ttk.Checkbutton(frame, text="Short Circuit Mode", variable=mode_SC) \
    .grid(column=0, row=1, sticky="w", padx=(5, 0), pady=5)
ttk.Checkbutton(frame, text="Z Mode", variable=mode_Z) \
    .grid(column=1, row=1, sticky="w", padx=(5, 0), pady=5)

# 1. Plot ZL
ttk.Label(frame, text="1. Enter Load Impedance (Z = R + jX):", font=("Segoe UI", 12, "bold")) \
    .grid(column=0, row=2, columnspan=5, sticky="w", pady=(15, 5))
zl_frame = ttk.Frame(frame)
zl_frame.grid(row=3, column=0, columnspan=5, sticky="w", pady=5)
ttk.Label(zl_frame, text="R:").grid(column=0, row=0, sticky="e")
entry_real = ttk.Entry(zl_frame, width=6)
entry_real.grid(column=1, row=0, sticky="w", padx=(2, 10))
ttk.Label(zl_frame, text="X:").grid(column=2, row=0, sticky="e")
entry_imag = ttk.Entry(zl_frame, width=6)
entry_imag.grid(column=3, row=0, sticky="w", padx=(2, 15))
ttk.Button(zl_frame, text="Plot Impedance", command=on_plot_gamma)\
    .grid(column=4, row=0, sticky="w")

# 2. Rotate gamma
ttk.Label(frame, text="2. Enter Rotation Distance:", font=("Segoe UI", 12, "bold")) \
    .grid(column=0, row=4, columnspan=5, sticky="w", pady=(15, 5))
rotate_frame = ttk.Frame(frame)
rotate_frame.grid(row=5, column=0, columnspan=5, sticky="w", pady=5)
ttk.Label(rotate_frame, text="d1:").grid(column=0, row=0, sticky="e")
d_val = ttk.Entry(rotate_frame, width=6)
d_val.grid(column=1, row=0, sticky="w", padx=(2, 15))
ttk.Button(rotate_frame, text="Rotate d1", command=on_rotate_gamma).grid(column=2, row=0, sticky="w")

# 3. Mirror gamma
mirror_frame = ttk.Frame(frame)
mirror_frame.grid(row=6, column=0, columnspan=5, sticky="w", pady=(15, 5))
ttk.Label(mirror_frame, text="3. Mirror Γ:", font=("Segoe UI", 12, "bold")).grid(column=0, row=0, sticky="w")
ttk.Button(mirror_frame, text="Mirror", command=on_mirror_gamma).grid(column=1, row=0, sticky="w", padx=5)

# 4. Plot auxiliary circles / distance between stubs
ttk.Label(frame, text="4. Enter the distance between the stubs:", font=("Segoe UI", 12, "bold")) \
    .grid(column=0, row=7, columnspan=5, sticky="w", pady=(15, 5))
stub_frame = ttk.Frame(frame)
stub_frame.grid(row=8, column=0, columnspan=5, sticky="w", pady=5)
ttk.Label(stub_frame, text="d2 > 0:").grid(column=0, row=0, sticky="e")
d2_var = ttk.Entry(stub_frame, width=6)
d2_var.grid(column=1, row=0, sticky="w", padx=(2, 15))
ttk.Button(stub_frame, text="Rotate d2", command=on_create_aux_circles_and_find_intersection).grid(column=2, row=0, sticky="w")


# 5. Subtract for L1
mirror_frame = ttk.Frame(frame)
mirror_frame.grid(row=9, column=0, columnspan=5, sticky="w", pady=(15, 5))
ttk.Label(mirror_frame, text="5. Subtract For L1:", font=("Segoe UI", 12, "bold")).grid(column=0, row=0, sticky="w")
ttk.Button(mirror_frame, text="Subtract", command=on_subtract).grid(column=1, row=0, sticky="w", padx=5)


# 6. Rotate back for L2
mirror_frame = ttk.Frame(frame)
mirror_frame.grid(row=12, column=0, columnspan=5, sticky="w", pady=(15, 5))
ttk.Label(mirror_frame, text="6. Rotate Unity Back:", font=("Segoe UI", 12, "bold")).grid(column=0, row=0, sticky="w")
ttk.Button(mirror_frame, text="Rotate Back", command=on_rotate_unity_back).grid(column=1, row=0, sticky="w", padx=5)


# Clear a point2
row_frame = ttk.Frame(frame)
row_frame.grid(row=16, column=0, columnspan=5, sticky="w", pady=5)
ttk.Label(row_frame, text="7. Clear a Point: ",font=("Segoe UI", 12, "bold")).grid(column=0, row=0, sticky="e")
ttk.Button(row_frame, text="Clear Point", command=on_remove_point)\
    .grid(column=4, row=0, sticky="w")


row_frame = ttk.Frame(frame)
row_frame.grid(row=18, column=0, columnspan=5, sticky="w", pady=5)
ttk.Label(row_frame, text="8. Clear a Circle: ",font=("Segoe UI", 12, "bold")).grid(column=0, row=0, sticky="e")
ttk.Button(row_frame, text="Clear Circle", command=on_remove_circle)\
    .grid(column=4, row=0, sticky="w")

row_frame = ttk.Frame(frame)
row_frame.grid(row=19, column=0, columnspan=5, sticky="w", pady=5)
ttk.Label(row_frame, text="9. Clear an Arc: ",font=("Segoe UI", 12, "bold")).grid(column=0, row=0, sticky="e")
ttk.Button(row_frame, text="Clear Arc", command=on_remove_arc)\
    .grid(column=4, row=0, sticky="w")

root.mainloop()

