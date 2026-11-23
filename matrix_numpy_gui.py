"""
==============================================================================
 3x3 MATRIX CALCULATOR  -  NumPy VERSION
==============================================================================
 Uses NumPy arrays and NumPy's linear-algebra tools (np.linalg.det, np.linalg.inv,
 np.linalg.eig, ...) wherever NumPy offers them, plus Tkinter for the GUI.

 HOW TO FIND AN OPERATION
 ------------------------
 Every operation sits between two comment lines like:
        # ===== START OF ADDITION OPERATION =====
        # ===== END OF ADDITION OPERATION =====
 Each block contains the whole operation (reading the inputs + the maths +
 the text that is shown in the output box).

 Run it:   python matrix_numpy_gui.py   (needs:  pip install numpy)
==============================================================================
"""

import math
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from fractions import Fraction

N = 3            # size of the matrices (3x3)
EPS = 1e-9       # numbers smaller than this are treated as zero


# ==============================================================================
#  SHARED HELPERS  (used by many operations - not an operation themselves)
# ==============================================================================

def fmt(x, nd=4):
    """Turn a number (real or complex) into neat text, e.g. 2.0 -> '2'."""
    if isinstance(x, complex):
        re, im = x.real, x.imag
        if abs(im) < 1e-9:
            return fmt(re, nd)
        im_txt = fmt(abs(im), nd) + "i"
        if abs(re) < 1e-9:
            return ("-" if im < 0 else "") + im_txt
        return f"{fmt(re, nd)} {'+' if im >= 0 else '-'} {im_txt}"
    s = f"{float(x):.{nd}f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def parse_number(text):
    """Read a number from a text box. Accepts 3, -2.5 and fractions like 1/2."""
    text = text.strip()
    if text == "":
        raise ValueError("A box is empty - please fill every cell.")
    try:
        value = float(text)
    except ValueError:
        try:
            value = float(Fraction(text))
        except (ValueError, ZeroDivisionError):
            raise ValueError(f"'{text}' is not a valid number.")
    if not math.isfinite(value):
        raise ValueError(f"'{text}' is not a valid number.")
    return value


def mat_str(M, indent="  "):
    """Turn a matrix (NumPy array or list) into aligned text."""
    cells = [[fmt(v) for v in row] for row in np.asarray(M).tolist()]
    w = max(len(c) for row in cells for c in row)
    return "\n".join(indent + "[ " + "  ".join(c.rjust(w) for c in row) + " ]"
                     for row in cells)


def vec_str(v):
    return "[ " + ",  ".join(fmt(x) for x in np.asarray(v).tolist()) + " ]"


def clean(M, tol=1e-12):
    """Replace tiny rounding noise (like 3e-17) by exact 0."""
    M = np.array(M)
    return np.where(np.abs(M) < tol, 0, M)       # np.where keeps M, but replaces tiny numbers by 0


def reduce_matrix(M, jordan, tol=EPS):
    """
    Core row-reduction engine used by Gauss-Jordan elimination.
      jordan=False -> Gaussian elimination (row echelon form)
      jordan=True  -> Gauss-Jordan elimination (reduced row echelon form)
    Returns (result_matrix, list_of_steps, pivot_columns).
    NumPy lets us work on whole rows at once, e.g.  A[i] -= f * A[r].
    """
    A = np.array(M, dtype=float)
    rows, cols = A.shape
    steps = [("Start", clean(A))]
    pivots = []
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        p = r + int(np.argmax(np.abs(A[r:, c])))      # biggest number in column
        if abs(A[p, c]) < tol:
            continue                                   # no pivot in this column
        if p != r:
            A[[r, p]] = A[[p, r]]
            steps.append((f"Swap R{r+1} <-> R{p+1}", clean(A)))
        if jordan and abs(A[r, c] - 1) > tol:          # make the pivot equal to 1
            f = A[r, c]
            A[r] = np.divide(A[r], f)
            steps.append((f"R{r+1} = R{r+1} / {fmt(f)}", clean(A)))
        targets = range(rows) if jordan else range(r + 1, rows)
        for i in targets:                              # make zeros in the column
            if i != r and abs(A[i, c]) > tol:
                f = A[i, c] / A[r, c]
                A[i] = np.subtract(A[i], np.multiply(f, A[r]))
                A[i, c] = 0.0
                steps.append((f"R{i+1} = R{i+1} - ({fmt(f)}) * R{r+1}", clean(A)))
        pivots.append(c)
        r += 1
    return clean(A), steps, pivots


def steps_text(steps):
    out = []
    for k, (label, M) in enumerate(steps):
        out.append(f"Step {k}: {label}")
        out.append(mat_str(M))
        out.append("")
    return "\n".join(out)


# ==============================================================================
#  OPERATIONS   (each returns  (text_to_display, result_matrix_or_None) )
# ==============================================================================

# ===== START OF ADDITION OPERATION =====
def op_addition(app):
    A, B = app.get_A(), app.get_B()
    R = np.add(A, B)                           # np.add adds position by position
    return "A + B =\n" + mat_str(R), R
# ===== END OF ADDITION OPERATION =====


# ===== START OF SUBTRACTION OPERATION =====
def op_subtraction(app):
    A, B = app.get_A(), app.get_B()
    R = np.subtract(A, B)                      # np.subtract subtracts position by position
    return "A - B =\n" + mat_str(R), R
# ===== END OF SUBTRACTION OPERATION =====


# ===== START OF SCALAR MULTIPLICATION OPERATION =====
def op_scalar_multiplication(app):
    A, k = app.get_A(), app.get_k()
    R = np.multiply(k, A)                      # np.multiply multiplies every number by k
    return f"{fmt(k)} * A =\n" + mat_str(R), R
# ===== END OF SCALAR MULTIPLICATION OPERATION =====


# ===== START OF MATRIX MULTIPLICATION OPERATION =====
def op_matrix_multiplication(app):
    A, B = app.get_A(), app.get_B()
    R = np.matmul(A, B)                         # np.matmul = real matrix product (not element-wise)
    return "A x B =\n" + mat_str(R), R
# ===== END OF MATRIX MULTIPLICATION OPERATION =====


# ===== START OF TRANSPOSE OPERATION =====
def op_transpose(app):
    A = app.get_A()
    R = np.transpose(A)                         # np.transpose: rows become columns
    return "Transpose of A =\n" + mat_str(R), R
# ===== END OF TRANSPOSE OPERATION =====


# ===== START OF DETERMINANT OPERATION =====
def minor_submatrix(M, i, j):
    """Cross out row i and column j -> leftover 2x2 matrix."""
    return np.delete(np.delete(M, i, axis=0), j, axis=1)


def op_determinant(app):
    A = app.get_A()
    d = np.linalg.det(A)                        # NumPy determinant
    m = [np.linalg.det(minor_submatrix(A, 0, j)) for j in range(N)]
    text = ("Expansion along the first row:\n"
            f"  det(A) = ({fmt(A[0, 0])})({fmt(m[0])}) - ({fmt(A[0, 1])})({fmt(m[1])})"
            f" + ({fmt(A[0, 2])})({fmt(m[2])})\n"
            f"  det(A) = {fmt(d)}")
    return text, None
# ===== END OF DETERMINANT OPERATION =====


# ===== START OF INVERSE OPERATION =====
def inverse_matrix(M):
    """Inverse with NumPy. Returns None if the matrix is singular."""
    if abs(np.linalg.det(M)) < EPS:
        return None
    return np.linalg.inv(M)


def op_inverse(app):
    A = app.get_A()
    d = np.linalg.det(A)
    inv = inverse_matrix(A)
    if inv is None:
        raise ValueError("det(A) = 0, so A has no inverse (it is singular).")
    text = (f"det(A) = {fmt(d)}\n\nA^-1 =\n" + mat_str(inv) +
            "\n\nCheck: A x A^-1 =\n" + mat_str(clean(np.matmul(A, inv), 1e-9)))
    return text, inv
# ===== END OF INVERSE OPERATION =====


# ===== START OF TRACE OPERATION =====
def op_trace(app):
    A = app.get_A()
    t = np.trace(A)                             # sum of the main diagonal
    return (f"Trace(A) = {fmt(A[0, 0])} + {fmt(A[1, 1])} + {fmt(A[2, 2])}"
            f" = {fmt(t)}"), None
# ===== END OF TRACE OPERATION =====


# ===== START OF RANK OPERATION =====
def op_rank(app):
    A = app.get_A()
    r = np.linalg.matrix_rank(A)                # NumPy computes the rank (via SVD)
    return f"Rank(A) = {r}", None
# ===== END OF RANK OPERATION =====


# ===== START OF POWER OPERATION =====
def op_power(app):
    A, n = app.get_A(), app.get_n()
    if abs(n) > 1000:
        raise ValueError("Please use a power between -1000 and 1000.")
    try:
        R = np.linalg.matrix_power(A, n)        # negative n uses the inverse
    except np.linalg.LinAlgError:
        raise ValueError("A is singular, so a negative power is impossible.")
    return f"A^{n} =\n" + mat_str(R), R
# ===== END OF POWER OPERATION =====


# ===== START OF ROW OPERATION =====
def op_row_operation(app):
    A = app.get_A()
    kind, i, j, k = app.get_rowcol()
    if kind == "swap":
        if i == j:
            raise ValueError("To swap rows, Ri and Rj must be different.")
        A[[i, j]] = A[[j, i]]                   # swap two whole rows
        what = f"R{i+1} <-> R{j+1}"
    elif kind == "scale":
        if abs(k) < EPS:
            raise ValueError("k must not be 0 when scaling a row.")
        A[i] = np.multiply(k, A[i])
        what = f"R{i+1} = {fmt(k)} * R{i+1}"
    else:  # add a multiple of another row
        if i == j:
            raise ValueError("To add a multiple of a row, Ri and Rj must be different.")
        A[i] = np.add(A[i], np.multiply(k, A[j]))
        what = f"R{i+1} = R{i+1} + ({fmt(k)}) * R{j+1}"
    return f"Row operation  {what}\n\nResult:\n" + mat_str(A), A
# ===== END OF ROW OPERATION =====


# ===== START OF GAUSS-JORDAN ELIMINATION OPERATION =====
def op_gauss_jordan(app):
    A = app.get_A()
    aug = app.use_augmented()
    M = np.column_stack([A, app.get_b()]) if aug else A     # [A | b] if ticked
    R, steps, pivots = reduce_matrix(M, True)                # forward + backward elimination
    text = ("GAUSS-JORDAN ELIMINATION (reduced row echelon form)"
            + ("  on [A | b]" if aug else "") + "\n\n" + steps_text(steps))
    text += "Final reduced row echelon form:\n" + mat_str(R)
    if aug:
        if N in pivots:
            text += "\n\nThe system has NO solution (inconsistent)."
        elif pivots == [0, 1, 2]:
            text += "\n\nSolution (last column):  x = " + vec_str(R[:, N])
        else:
            text += "\n\nThe system has infinitely many solutions."
    return text, (None if aug else R)
# ===== END OF GAUSS-JORDAN ELIMINATION OPERATION =====


# ===== START OF EIGENVALUES AND EIGENVECTORS OPERATION =====
def eigen_pairs(A):
    """Use NumPy's eig. Returns (eigenvalues, eigenvectors-as-columns) in neat form."""
    w, V = np.linalg.eig(A)
    w = np.real_if_close(w, tol=1e6)             # drop tiny imaginary noise
    V = np.real_if_close(V, tol=1e6)
    order = np.lexsort((-w.imag, -w.real))   # sort: biggest real part first
    w, V = w[order], V[:, order]
    V = V.copy()
    for c in range(N):                           # make the first entry positive
        k = next((i for i in range(N) if abs(V[i, c]) > 1e-9), 0)
        V[:, c] = np.divide(V[:, c], V[k, c] / np.abs(V[k, c]))
    return w, V


def op_eigen(app):
    A = app.get_A()
    w, V = eigen_pairs(A)
    lines = ["Eigenvalues and (unit-length) eigenvectors from np.linalg.eig:", ""]
    for i in range(N):
        lines.append(f"λ{i+1} = {fmt(w[i])}")
        lines.append("    eigenvector: " + vec_str(V[:, i]))
        lines.append("")
    if np.iscomplexobj(w):
        lines.append("Note: complex eigenvalues come in conjugate pairs for a real matrix.")
    return "\n".join(lines), None
# ===== END OF EIGENVALUES AND EIGENVECTORS OPERATION =====


# ===== START OF LU DECOMPOSITION OPERATION =====
def op_lu(app):
    A = app.get_A()
    U = A.copy()
    L = np.zeros((N, N))
    P = np.eye(N)
    for k in range(N):
        p = k + int(np.argmax(np.abs(U[k:, k])))             # partial pivoting
        if p != k:
            U[[k, p]] = U[[p, k]]
            P[[k, p]] = P[[p, k]]
            L[[k, p]] = L[[p, k]]
        if abs(U[k, k]) < EPS:
            continue
        for i in range(k + 1, N):
            L[i, k] = U[i, k] / U[k, k]                      # multiplier
            U[i] = np.subtract(U[i], np.multiply(L[i, k], U[k]))
            U[i, k] = 0.0
    L = np.add(L, np.eye(N))
    text = ("P A = L U   (P = row-swap matrix, L = lower, U = upper triangular)\n\n"
            "P =\n" + mat_str(P) + "\n\nL =\n" + mat_str(L) +
            "\n\nU =\n" + mat_str(clean(U)) +
            "\n\nCheck: P A =\n" + mat_str(np.matmul(P, A)) +
            "\nCheck: L U =\n" + mat_str(clean(np.matmul(L, U), 1e-9)))
    return text, None
# ===== END OF LU DECOMPOSITION OPERATION =====


# ===== START OF INVERSE METHOD (SOLVE Ax = b) OPERATION =====
def op_inverse_method(app):
    A, b = app.get_A(), app.get_b()
    inv = inverse_matrix(A)
    if inv is None:
        raise ValueError("det(A) = 0, so the inverse method cannot be used.")
    x = np.matmul(inv, b)                        # x = A^-1 b
    text = ("x = A^-1 b\n\nA^-1 =\n" + mat_str(inv) +
            "\n\nSolution:  x = " + vec_str(x) +
            "\nCheck: A x = " + vec_str(clean(np.dot(A, x), 1e-9)) +
            "\nnp.linalg.solve(A, b) gives:  x = " + vec_str(np.linalg.solve(A, b)))
    return text, None
# ===== END OF INVERSE METHOD (SOLVE Ax = b) OPERATION =====


# ==============================================================================
#  BUTTON TABLE  (name on the button -> operation function above)
# ==============================================================================
# (group title, number of button columns, [(button text, function), ...])   # 15 operations in total
BUTTON_GROUPS = [
    ("Basic", 1, [("Addition  A + B", op_addition),
                  ("Subtraction  A - B", op_subtraction),
                  ("Scalar multiplication  k·A", op_scalar_multiplication),
                  ("Matrix multiplication  A×B", op_matrix_multiplication),
                  ("Transpose", op_transpose)]),
    ("Properties", 2, [("Determinant", op_determinant),
                       ("Inverse", op_inverse),
                       ("Trace", op_trace),
                       ("Rank", op_rank),
                       ("Power  A^n", op_power)]),
    ("Row work", 1, [("Row operation", op_row_operation),
                     ("Gauss-Jordan elimination", op_gauss_jordan)]),
    ("Decompositions", 1, [("Eigenvalues & eigenvectors", op_eigen),
                           ("LU decomposition", op_lu)]),
    ("Solve  A x = b", 1, [("Inverse method", op_inverse_method)]),
]

DEFAULT_A = [[1, 2, 3], [0, 1, 4], [5, 6, 0]]
DEFAULT_B = [[1, 0, 2], [0, 1, 3], [4, 0, 1]]
DEFAULT_b = [1, 2, 3]


# ==============================================================================
#  THE GUI  (window, input boxes, buttons, output box)
# ==============================================================================
class MatrixApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("3x3 Matrix Calculator - NumPy version")
        self.geometry("1150x780")
        self.minsize(950, 640)
        self.last_matrix = None                      # last matrix result
        self.aug_var = tk.BooleanVar(value=False)    # "use [A | b]" checkbox

        left = ttk.Frame(self, padding=8)
        left.grid(row=0, column=0, sticky="ns")
        right = ttk.Frame(self, padding=8)
        right.grid(row=0, column=1, sticky="nsew")
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self.build_inputs(left)
        self.build_buttons(right)
        self.build_output(right)
        self.show("Fill in the matrices on the left, then press any button.")

    # ---------- building the window ----------
    def make_grid(self, parent, values):
        entries = []
        for i in range(N):
            row = []
            for j in range(N):
                e = ttk.Entry(parent, width=7, justify="center")
                e.grid(row=i, column=j, padx=2, pady=2)
                e.insert(0, str(values[i][j]))
                row.append(e)
            entries.append(row)
        return entries

    def build_inputs(self, parent):
        box = ttk.LabelFrame(parent, text="Matrix A", padding=6)
        box.pack(fill="x", pady=3)
        self.A_entries = self.make_grid(box, DEFAULT_A)

        box = ttk.LabelFrame(parent, text="Matrix B", padding=6)
        box.pack(fill="x", pady=3)
        self.B_entries = self.make_grid(box, DEFAULT_B)

        box = ttk.LabelFrame(parent, text="Vector b  (for A x = b)", padding=6)
        box.pack(fill="x", pady=3)
        self.b_entries = []
        for j in range(N):
            e = ttk.Entry(box, width=7, justify="center")
            e.grid(row=0, column=j, padx=2, pady=2)
            e.insert(0, str(DEFAULT_b[j]))
            self.b_entries.append(e)

        box = ttk.LabelFrame(parent, text="Scalar k  and  power n", padding=6)
        box.pack(fill="x", pady=3)
        ttk.Label(box, text="k =").grid(row=0, column=0)
        self.k_entry = ttk.Entry(box, width=7, justify="center")
        self.k_entry.insert(0, "2")
        self.k_entry.grid(row=0, column=1, padx=4)
        ttk.Label(box, text="n =").grid(row=0, column=2)
        self.n_entry = ttk.Entry(box, width=7, justify="center")
        self.n_entry.insert(0, "2")
        self.n_entry.grid(row=0, column=3, padx=4)

        box = ttk.LabelFrame(parent, text="Row operation settings", padding=6)
        box.pack(fill="x", pady=3)
        self.kind_box = ttk.Combobox(box, state="readonly", width=24, values=[
            "Swap  (Ri <-> Rj)", "Scale  (Ri = k * Ri)", "Add  (Ri = Ri + k * Rj)"])
        self.kind_box.current(2)
        self.kind_box.grid(row=0, column=0, columnspan=4, pady=2)
        ttk.Label(box, text="i =").grid(row=1, column=0)
        self.i_box = ttk.Combobox(box, state="readonly", width=3, values=["1", "2", "3"])
        self.i_box.current(1)
        self.i_box.grid(row=1, column=1)
        ttk.Label(box, text="j =").grid(row=1, column=2)
        self.j_box = ttk.Combobox(box, state="readonly", width=3, values=["1", "2", "3"])
        self.j_box.current(0)
        self.j_box.grid(row=1, column=3)
        ttk.Label(box, text="k =").grid(row=2, column=0, pady=3)
        self.rc_k_entry = ttk.Entry(box, width=7, justify="center")
        self.rc_k_entry.insert(0, "-2")
        self.rc_k_entry.grid(row=2, column=1, columnspan=2)

        box = ttk.LabelFrame(parent, text="Tools", padding=6)
        box.pack(fill="x", pady=3)
        ttk.Button(box, text="Use last result as A", command=lambda: self.use_result(self.A_entries)).pack(fill="x", pady=1)
        ttk.Button(box, text="Use last result as B", command=lambda: self.use_result(self.B_entries)).pack(fill="x", pady=1)
        ttk.Button(box, text="Reset all inputs", command=self.reset_inputs).pack(fill="x", pady=1)

    def build_buttons(self, parent):
        area = ttk.Frame(parent)
        area.pack(fill="x")
        for g, (title, ncols, items) in enumerate(BUTTON_GROUPS):
            frame = ttk.LabelFrame(area, text=title, padding=5)
            frame.grid(row=g // 3, column=g % 3, sticky="nsew", padx=3, pady=3)
            for idx, (text, func) in enumerate(items):
                b = ttk.Button(frame, text=text, width=26 if ncols == 1 else 13,
                               command=lambda t=text, f=func: self.run(t, f))
                b.grid(row=idx // ncols, column=idx % ncols, padx=2, pady=2, sticky="ew")
            if title == "Row work":
                ttk.Checkbutton(frame, text="Use augmented [A | b]",
                                variable=self.aug_var).grid(
                    row=len(items), column=0, pady=2, sticky="w")

    def build_output(self, parent):
        frame = ttk.LabelFrame(parent, text="Output", padding=4)
        frame.pack(fill="both", expand=True, pady=6)
        self.output = tk.Text(frame, wrap="none", font=("Courier", 11))
        ys = ttk.Scrollbar(frame, orient="vertical", command=self.output.yview)
        xs = ttk.Scrollbar(frame, orient="horizontal", command=self.output.xview)
        self.output.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        ys.pack(side="right", fill="y")
        xs.pack(side="bottom", fill="x")
        self.output.pack(fill="both", expand=True)

    # ---------- reading the input boxes ----------
    def wrap(self, rows):
        return np.array(rows, dtype=float)      # NumPy array

    def read_grid(self, entries, name):
        try:
            return [[parse_number(e.get()) for e in row] for row in entries]
        except ValueError as err:
            raise ValueError(f"Matrix {name}: {err}")

    def get_A(self):
        return self.wrap(self.read_grid(self.A_entries, "A"))

    def get_B(self):
        return self.wrap(self.read_grid(self.B_entries, "B"))

    def get_b(self):
        try:
            return self.wrap([parse_number(e.get()) for e in self.b_entries])
        except ValueError as err:
            raise ValueError(f"Vector b: {err}")

    def get_k(self):
        return parse_number(self.k_entry.get())

    def get_n(self):
        x = parse_number(self.n_entry.get())
        if x != int(x):
            raise ValueError("The power n must be a whole number.")
        return int(x)

    def get_rowcol(self):
        kind = ["swap", "scale", "add"][self.kind_box.current()]
        k = parse_number(self.rc_k_entry.get())
        return kind, int(self.i_box.get()) - 1, int(self.j_box.get()) - 1, k

    def use_augmented(self):
        return self.aug_var.get()

    # ---------- running an operation ----------
    def run(self, title, func):
        try:
            text, matrix = func(self)
        except ValueError as err:
            messagebox.showerror("Problem", str(err))
            return
        except Exception as err:                     # anything unexpected
            messagebox.showerror("Unexpected error", repr(err))
            return
        self.last_matrix = matrix
        self.show(f"===== {title} =====\n\n{text}")

    def show(self, text):
        self.output.delete("1.0", "end")
        self.output.insert("1.0", text)

    def use_result(self, entries):
        M = self.last_matrix
        if M is None:
            messagebox.showinfo("No result", "The last operation did not give a 3x3 matrix.")
            return
        M = [list(row) for row in M]
        if len(M) != N or len(M[0]) != N:
            messagebox.showinfo("No result", "The last result is not a 3x3 matrix.")
            return
        if any(isinstance(v, complex) and abs(v.imag) > 1e-9 for r in M for v in r):
            messagebox.showinfo("No result", "The result contains complex numbers.")
            return
        for i in range(N):
            for j in range(N):
                entries[i][j].delete(0, "end")
                entries[i][j].insert(0, fmt(M[i][j].real if isinstance(M[i][j], complex) else M[i][j], 6))

    def reset_inputs(self):
        for entries, vals in ((self.A_entries, DEFAULT_A), (self.B_entries, DEFAULT_B)):
            for i in range(N):
                for j in range(N):
                    entries[i][j].delete(0, "end")
                    entries[i][j].insert(0, str(vals[i][j]))
        for e, v in zip(self.b_entries, DEFAULT_b):
            e.delete(0, "end")
            e.insert(0, str(v))
        self.show("Inputs reset.")


if __name__ == "__main__":
    MatrixApp().mainloop()
