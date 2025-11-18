## Python GUI for impedance matching with the Smith chart

This project uses the skrf, matplotlib, numpy, and tkinter libraries to create a GUI that plots on the Smith chart.

## Installation

To run the code, you need to install the following Python libraries. You can install them using pip:

```bash
pip install skrf matplotlib numpy
```
This project also requires Tkinter. Some systems do not include it by default, so you may need to install it manually.

A smith chart in the form of a png has also been provided.


## Notation Convention

This project uses the notation that is illustrated in the figure below.

<img width="740" height="322" alt="T_line_convention" src="https://github.com/user-attachments/assets/db98a0fa-d9f9-44e3-b161-98edca705acd" />

The load impedance is at z = 0.

The generator is at z = - l.

The reflected wave has a negative sign in its exponential term, while the forward-propagating wave has a positive sign.

