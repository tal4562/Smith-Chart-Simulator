## Python GUI for impedance matching with the Smith chart

This project uses the skrf, matplotlib, numpy, and tkinter libraries to create a GUI that plots on the Smith chart.

Two examples are included.

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

## Single Stub Matching
**In both files the code saves the last impedance and gamma for mirror and rotation.**

 In case you closed the figure just open a new chart you should remain on the currect gamma and Z.

The GUI for matching with a single stub or without a stub:

<img width="317" height="602" alt="image" src="https://github.com/user-attachments/assets/285bc06b-d4b7-4eeb-8e20-d6f6b04337a2" />

Note the two check boxes on the top. by default they are initizlied as impedance input and a short circuit stub.

Toggle them at needed depending on your use.

0. Opens a new chart.
1. Enter the normalized impedance in the form Z = R + Xj.
2. Rotation - by convention rotation towards the generator requires d < 0 and towards the load d > 0.
3. Mirror - flips gamma by 180 to swap from impedance to admittance and vice versa.
4. Match without a stub - finds the intersection with the circle 1 + Aj and the required lump passive element to match.
5. Match with a stub - Z in is the impedance that we wish to match to. Zc line is the line impedance that is connected to the load.
   default values are 50 Ohms. find the intersection with the circle 1 + Xj and returns the stub's distance from the load.
6. Length of stubs - finds L the length of the stub.

## Double Stub Matching

The GUI for matching with two stubs:

<img width="333" height="607" alt="image" src="https://github.com/user-attachments/assets/c2e9199d-ef0b-484c-9ad4-e3064d930294" />

mostly the same as in the single case.

4. Rotates by the distance between the stubs d2 > 0.

5. Subtracts the intersections to find the imaginary component to match and returns L1, the length of the first stub.
6. Rotates the 1 + Xj circle back to its place to find L2 the length of the second stub.

