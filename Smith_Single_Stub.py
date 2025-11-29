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
fig, ax, gamma, gamma_print, Z, Z_print = None, None, np.ndarray, str, complex, str
plotted_points, plotted_circles,plotted_arcs = [], [], [] # global list to store all plotted points, circles and arcs
Zint = [0, 0]


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


# Remove an arc
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
def parse_fraction(s: str) -> float:
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


#################################### Chart Functions ####################################
# === 0. Show Smith chart ===
def show_chart() -> None:
    print('A new Diagram Has Been Opened')
    global fig, ax, plotted_points, plotted_circles,plotted_arcs
    plt.close('all')
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal', adjustable='box')
    plotted_points, plotted_circles, plotted_arcs = [], [], []

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
    global fig, ax, gamma, gamma_print, Z, Z_print
    input_func()

    # negate gamma
    gamma[0], gamma[1], gamma[2] = -gamma[0], -gamma[1], np.arctan2(-gamma[1], -gamma[0])
    gamma[2] = 2 * pi + gamma[2] if gamma[2] < 0 else gamma[2]
    gamma_print = str(np.round(gamma[0] + 1j * gamma[1], 3))

    # Toggle mode_Z
    mode_Z.set(not mode_Z.get())

    if mode_Z.get():
        print('Option Three: Printing and working with admittance Z -> Y.')
    else:
        print('Option Three: Printing and working with impedance Y -> Z.')

    # plots new Z
    Z, Z_print = gamma_to_impedance(gamma[0] + 1j * gamma[1])
    plot_gamma_point(ax, gamma[0:3], ['D', 'green', 'Z_L_mir' if mode_Z.get() else 'Y_L_mir'])

    plt.draw()


# === 4. Plot and print the matched passive element ===
def on_match_without_stub() -> None:
    global fig, ax, gamma, gamma_print, Z, Z_print, plotted_circles, plotted_arcs
    input_func()
    print('Option Four: Matching without a stub. Make sure you are working with impedance (Z).')

    tmp_gamma, tmp_gamma_print, tmp_Z, tmp_Z_print = gamma, gamma_print, Z, Z_print

    # plots the Unity circle 1 + Xj
    (cir,) = ax.plot(0.5 * cosine + 0.5, 0.5 * sine, color='b', linewidth=2)
    plotted_circles.append(cir)

    # Find intersection points with 1 + Xj
    x_int = gamma[3] ** 2
    y_int1 = np.sqrt(x_int - x_int ** 2)
    for i, y in enumerate([y_int1, -y_int1]):
        i += 1
        # --- Compute gamma point ---
        gamma = np.array([x_int, y, np.arctan2(y, x_int), np.sqrt(x_int ** 2 + y ** 2)])
        gamma[2] = gamma[2] + 2 * np.pi if gamma[2] < 0 else gamma[2]

        # Plot the intersection point
        Z, Z_print = gamma_to_impedance(gamma[0] + 1j * gamma[1])
        plot_gamma_point(ax, gamma[0:3], ['*', 'm', 'Z_int' + str(i)])

        ang2wl = clock_wise_arc(tmp_gamma[2], gamma[2], i, 'y', 'arc ')
        print(f"The length of the matching line (yellow arc length): {ang2wl}λ, "
              f"Imaginary component: {np.round(-Z.imag, 3)}")

    gamma, gamma_print, Z, Z_print = tmp_gamma, tmp_gamma_print, tmp_Z, tmp_Z_print

    plt.draw()


# === 5. Plot and print the stub distance ===
def on_match_with_stub() -> None:
    global fig, ax, gamma, gamma_print, Z, Z_print, plotted_circles, Zint
    input_func()
    tmp_gamma, tmp_gamma_print, tmp_Z, tmp_Z_print = gamma, gamma_print, Z, Z_print

    print('Option 5: matching with ' + ('a short circuit.' if mode_SC.get() else 'an open circuit.'))
    try:
        Zc_in_value = parse_fraction(Zc_in.get())  # the value to match, the input impedance of the generator
        Zc_line_value = parse_fraction(Zc_line.get())  # the value of the connected line
    except ValueError:
        print("⚠️ Invalid input: enter numeric Zc.")
        return
    if mode_Z.get():
        print('Current mode is impedance (Z), swap to admittance (Y).')

    # finds the real part to match
    ratio = Zc_line_value / Zc_in_value
    gamma_abs = (ratio - 1) / (1 + ratio)
    r = (1 - gamma_abs) / 2
    x0 = (1 + gamma_abs) / 2

    # plots the constant real circle, ratio + Aj
    (cir,) = ax.plot(r * cosine + x0, sine * r, color='green', linewidth=2)
    plotted_circles.append(cir)

    if abs(x0) < 1e-12 and abs(tmp_gamma[3] - r) < 1e-12:
        raise ValueError("Circles coincide: infinitely many intersections")

    # find the intersection points of the circles with real part abs(gamma) and ratio
    x = (tmp_gamma[3] ** 2 - r ** 2 + x0 ** 2) / (2.0 * x0)
    y_int = np.sqrt(tmp_gamma[3] ** 2 - x ** 2)

    if y_int < 0:
        print('No real intersection.')
        raise ValueError("No real intersection: y < 0")

    # plot the intersection points
    for i, y in enumerate([y_int, -y_int]):
        i += 1
        gamma = np.array([x, y, np.arctan2(y, x), np.sqrt(x ** 2 + y ** 2)])
        gamma[2] = gamma[2] + 2 * np.pi if gamma[2] < 0 else gamma[2]

        # Plot the intersection points
        Z, Z_print = gamma_to_impedance(gamma[0] + 1j * gamma[1])
        Zint[i - 1] = - 1j * Z.imag
        plot_gamma_point(ax, gamma[0:3], ['*', 'm', 'Y_int' + str(i)])

        ang2wl = clock_wise_arc(tmp_gamma[2], gamma[2], i, 'y', 'arc ')
        ang2wl = np.round(ang2wl, 4)
        print('The stub distance from the load d (yellow arc length) : ' + str(ang2wl) + 'λ')

    plt.draw()


# === 6. Plot and print the stubs values and their lengths ===
def on_plot_stubs() -> None:
    global fig, ax, gamma, gamma_print, Z, Z_print, Zint
    input_func()
    tmp_gamma, tmp_gamma_print, tmp_Z, tmp_Z_print = gamma, gamma_print, Z, Z_print

    print('Option 6: length of the' + (' short circuit stub.' if mode_SC.get() else ' open circuit stub.'))
    try:
        Zc_in_value = parse_fraction(Zc_in.get())  # Get the input from Zc_in field
        Zc_line_value = parse_fraction(Zc_line.get())  # Get the input from Zc_line field
    except ValueError:
        print("⚠️ Invalid input: enter numeric Zc.")
        return

    for i, inter in enumerate(Zint):
        i += 1
        wtf, _ = impedance_to_gamma(inter * Zc_in_value / Zc_line_value)
        Z, Z_print = gamma_to_impedance(wtf[0] + 1j * wtf[1])

        plot_gamma_point(ax, wtf[0:3], ['X', 'c', 'stub ' + str(i)])
        a = - wtf[2] if mode_SC.get() else pi - wtf[2]
        radd = i / 3
        clock_wise_arc(0 if mode_SC.get() else pi, wtf[2], radd, 'k', 'stub arc ' + str(i) + ' : ')
        a = (a / (4 * np.pi)) % (0.5)
        print('Length of the ' + (
            'short circuit ' if mode_SC.get() else 'open circuit ') + 'stub (black arc length) = ' + str(
            np.round(a, 4)))

    gamma, gamma_print, Z, Z_print = tmp_gamma, tmp_gamma_print, tmp_Z, tmp_Z_print

    plt.draw()


# === 7. clear the first plotted point ===
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


# === 8. clear the first plotted circle ===
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


# === 7. clear the first plotted arc ===
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
frame = ttk.Frame(root, padding=20)
frame.grid(row=0, column=0, sticky="w")

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

# Checkbox for Short Circuit Mode
ttk.Checkbutton(frame, text="Short Circuit Mode", variable=mode_SC) \
    .grid(column=0, row=1, sticky="w", padx=(5, 0), pady=5)
# Checkbox for Z Mode
ttk.Checkbutton(frame, text="Z Mode", variable=mode_Z) \
    .grid(column=1, row=1, sticky="w", padx=(5, 0), pady=5)

# 1. Plot ZL label and button
ttk.Label(frame, text="1. Enter Load Impedance (Z = R + jX):", font=("Segoe UI", 12, "bold")) \
    .grid(column=0, row=2, columnspan=5, sticky="w", pady=(15, 5))
# R
row_frame = ttk.Frame(frame)
row_frame.grid(row=4, column=0, columnspan=5, sticky="w", pady=5)
ttk.Label(row_frame, text="R:").grid(column=0, row=0, sticky="e")
entry_real = ttk.Entry(row_frame, width=6)
entry_real.grid(column=1, row=0, sticky="w", padx=(2, 10))
# X
ttk.Label(row_frame, text="X:").grid(column=2, row=0, sticky="e")
entry_imag = ttk.Entry(row_frame, width=6)
entry_imag.grid(column=3, row=0, sticky="w", padx=(2, 15))
ttk.Button(row_frame, text="Plot Impedance", command=on_plot_gamma).grid(column=4, row=0, sticky="w")

# 2. Rotate gamma
ttk.Label(frame, text="2. Enter rotation distance:", font=("Segoe UI", 12, "bold")) \
    .grid(column=0, row=5, columnspan=5, sticky="w", pady=(15, 5))
# Create a sub-frame for the row of widgets
row_frame = ttk.Frame(frame)
row_frame.grid(row=6, column=0, columnspan=5, sticky="w", pady=5)
# d input
ttk.Label(row_frame, text="d:").grid(column=0, row=0, sticky="e")
d_val = ttk.Entry(row_frame, width=6)
d_val.grid(column=1, row=0, sticky="w", padx=(2, 15))
# Rotate button
ttk.Button(row_frame, text="Rotate", command=on_rotate_gamma).grid(column=2, row=0, sticky="w")

# 3. Mirror gamma
mirror_row = ttk.Frame(frame)
mirror_row.grid(row=7, column=0, columnspan=5, sticky="w", pady=(15, 5))
ttk.Label(mirror_row, text="3. Mirror Γ:", font=("Segoe UI", 12, "bold")).grid(column=0, row=0, sticky="w")
ttk.Button(mirror_row, text="Mirror", command=on_mirror_gamma).grid(column=1, row=0, sticky="w", padx=5)

# 4. Match without stub
mirror_row = ttk.Frame(frame)
mirror_row.grid(row=9, column=0, columnspan=5, sticky="w", pady=(15, 5))
ttk.Label(mirror_row, text="4. Match Without a Stub:", font=("Segoe UI", 12, "bold")).grid(column=0, row=0, sticky="w")
ttk.Button(mirror_row, text="Match Zero", command=on_match_without_stub).grid(column=1, row=0, sticky="w", padx=5)

# 5. Match with a stub
ttk.Label(frame, text="5. Match With a Stub:", font=("Segoe UI", 12, "bold")) \
    .grid(column=0, row=11, columnspan=5, sticky="w", pady=(15, 5))

# Create a new row_frame for Z input and button
row_frame = ttk.Frame(frame)
row_frame.grid(row=12, column=0, columnspan=5, sticky="w", pady=5)

# Z in
ttk.Label(row_frame, text="Z in:").grid(column=0, row=0, sticky="e")
Zc_in = ttk.Entry(row_frame, width=6)
Zc_in.grid(column=1, row=0, sticky="w", padx=(2, 10))
Zc_in.insert(0, "50")  # set default value

# Zc line
ttk.Label(row_frame, text="Zc line:").grid(column=2, row=0, sticky="e")
Zc_line = ttk.Entry(row_frame, width=6)
Zc_line.grid(column=3, row=0, sticky="w", padx=(2, 15))
Zc_line.insert(0, "50")  # set default value

# Button
ttk.Button(row_frame, text="Match One", command=on_match_with_stub) \
    .grid(column=4, row=0, sticky="w")

# 6. length of stubs
mirror_row = ttk.Frame(frame)
mirror_row.grid(row=13, column=0, columnspan=5, sticky="w", pady=(15, 5))
ttk.Label(mirror_row, text="6. Length of Stubs", font=("Segoe UI", 12, "bold")).grid(column=0, row=0, sticky="w")
ttk.Button(mirror_row, text="Find length", command=on_plot_stubs).grid(column=1, row=0, sticky="w", padx=5)

# 7. Clear a point
row_frame = ttk.Frame(frame)
row_frame.grid(row=14, column=0, columnspan=5, sticky="w", pady=5)
ttk.Label(row_frame, text="7. Clear a Point: ", font=("Segoe UI", 12, "bold")).grid(column=0, row=0, sticky="e")
ttk.Button(row_frame, text="Clear Point", command=on_remove_point) \
    .grid(column=4, row=0, sticky="w")

# 8. Clear a circle
row_frame = ttk.Frame(frame)
row_frame.grid(row=15, column=0, columnspan=5, sticky="w", pady=5)
ttk.Label(row_frame, text="8. Clear a Circle: ", font=("Segoe UI", 12, "bold")).grid(column=0, row=0, sticky="e")
ttk.Button(row_frame, text="Clear Circle", command=on_remove_circle) \
    .grid(column=4, row=0, sticky="w")

# 9. Clear an arc
row_frame = ttk.Frame(frame)
row_frame.grid(row=16, column=0, columnspan=5, sticky="w", pady=5)
ttk.Label(row_frame, text="9. Clear an Arc: ", font=("Segoe UI", 12, "bold")).grid(column=0, row=0, sticky="e")
ttk.Button(row_frame, text="Clear Arc", command=on_remove_arc) \
    .grid(column=4, row=0, sticky="w")

root.mainloop()
