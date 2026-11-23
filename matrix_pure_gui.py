"""
==============================================================================
 3x3 MATRIX CALCULATOR  -  PURE PYTHON VERSION  (NO NumPy anywhere)
==============================================================================
 Everything (determinant, inverse, eigenvalues, LU, ...) is written by hand
 using only plain Python lists, the math/cmath modules, and Tkinter for the GUI.

 HOW TO FIND AN OPERATION
 ------------------------
 Every operation sits between two comment lines like:
        # ===== START OF ADDITION OPERATION =====
        # ===== END OF ADDITION OPERATION =====
 Each block contains the whole operation (reading the inputs + the maths +
 the text that is shown in the output box).

 Run it:   python matrix_pure_gui.py
==============================================================================
"""

import math
import cmath
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


def mat_str(M, indent="  "):
    """Turn a matrix (list of rows) into aligned text."""
    cells = [[fmt(v) for v in row] for row in M]
    w = max(len(c) for row in cells for c in row)
    return "\n".join(indent + "[ " + "  ".join(c.rjust(w) for c in row) + " ]"
                     for row in cells)


def vec_str(v):
    return "[ " + ",  ".join(fmt(x) for x in v) + " ]"


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


def identity(n=N):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def copy_m(M):
    return [list(row) for row in M]


def clean(M, tol=1e-12):
    """Replace tiny rounding noise (like 3e-17) by exact 0."""
    return [[0.0 if abs(v) < tol else v for v in row] for row in M]


def matmul_m(X, Y):
    """Plain triple-loop matrix product."""
    return [[sum(X[i][k] * Y[k][j] for k in range(len(Y)))
             for j in range(len(Y[0]))] for i in range(len(X))]


def mat_vec(M, v):
    return [sum(M[i][k] * v[k] for k in range(N)) for i in range(N)]


def reduce_matrix(M, jordan, tol=EPS):
    """
    Core row-reduction engine used by Rank, Gauss-Jordan elimination and
    the eigenvector finder.
      jordan=False -> Gaussian elimination (row echelon form)
      jordan=True  -> Gauss-Jordan elimination (reduced row echelon form)
    Returns (result_matrix, list_of_steps, pivot_columns).
    """
    A = copy_m(M)
    rows, cols = len(A), len(A[0])
    steps = [("Start", clean(A))]
    pivots = []
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        # choose the row with the biggest number in this column (most accurate)
        p = max(range(r, rows), key=lambda i: abs(A[i][c]))
        if abs(A[p][c]) < tol:
            continue                      # no pivot in this column
        if p != r:
            A[r], A[p] = A[p], A[r]
            steps.append((f"Swap R{r+1} <-> R{p+1}", clean(A)))
        if jordan and abs(A[r][c] - 1) > tol:      # make the pivot equal to 1
            f = A[r][c]
            A[r] = [v / f for v in A[r]]
            steps.append((f"R{r+1} = R{r+1} / {fmt(f)}", clean(A)))
        targets = range(rows) if jordan else range(r + 1, rows)
        for i in targets:                          # make zeros in the column
            if i != r and abs(A[i][c]) > tol:
                f = A[i][c] / A[r][c]
                A[i] = [a - f * b for a, b in zip(A[i], A[r])]
                A[i][c] = 0.0
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


def null_space(M, tol=1e-6):
    """Find all vectors v with M v = 0 (used to get eigenvectors)."""
    R, _, piv = reduce_matrix(M, True, tol)
    n = len(M[0])
    vectors = []
    for f in [c for c in range(n) if c not in piv]:      # every free column
        v = [0.0] * n
        v[f] = 1.0
        for row, pc in enumerate(piv):
            v[pc] = -R[row][f]
        norm = math.sqrt(sum(abs(x) ** 2 for x in v))
        v = [x / norm for x in v]                         # unit length
        k = next(i for i, x in enumerate(v) if abs(x) > 1e-9)
        phase = v[k] / abs(v[k])                          # first entry positive
        vectors.append([x / phase for x in v])
    return vectors


# ==============================================================================
#  OPERATIONS   (each returns  (text_to_display, result_matrix_or_None) )
# ==============================================================================

# ===== START OF ADDITION OPERATION =====
def op_addition(app):
    A, B = app.get_A(), app.get_B()
    # add the numbers that sit in the same position
    R = [[A[i][j] + B[i][j] for j in range(N)] for i in range(N)]
    return "A + B =\n" + mat_str(R), R
# ===== END OF ADDITION OPERATION =====


# ===== START OF SUBTRACTION OPERATION =====
def op_subtraction(app):
    A, B = app.get_A(), app.get_B()
    # subtract the numbers that sit in the same position
    R = [[A[i][j] - B[i][j] for j in range(N)] for i in range(N)]
    return "A - B =\n" + mat_str(R), R
# ===== END OF SUBTRACTION OPERATION =====


# ===== START OF SCALAR MULTIPLICATION OPERATION =====
def op_scalar_multiplication(app):
    A, k = app.get_A(), app.get_k()
    # multiply every number by k
    R = [[k * A[i][j] for j in range(N)] for i in range(N)]
    return f"{fmt(k)} * A =\n" + mat_str(R), R
# ===== END OF SCALAR MULTIPLICATION OPERATION =====


# ===== START OF MATRIX MULTIPLICATION OPERATION =====
def op_matrix_multiplication(app):
    A, B = app.get_A(), app.get_B()
    # entry (i,j) = row i of A  dot  column j of B
    R = [[sum(A[i][k] * B[k][j] for k in range(N)) for j in range(N)]
         for i in range(N)]
    return "A x B =\n" + mat_str(R), R
# ===== END OF MATRIX MULTIPLICATION OPERATION =====


# ===== START OF TRANSPOSE OPERATION =====
def op_transpose(app):
    A = app.get_A()
    # rows become columns
    R = [[A[j][i] for j in range(N)] for i in range(N)]
    return "Transpose of A =\n" + mat_str(R), R
# ===== END OF TRANSPOSE OPERATION =====


# ===== START OF DETERMINANT OPERATION =====
def minor_submatrix(M, i, j):
    """Cross out row i and column j -> leftover 2x2 matrix."""
    return [[M[r][c] for c in range(N) if c != j] for r in range(N) if r != i]


def det2(m):
    return m[0][0] * m[1][1] - m[0][1] * m[1][0]


def det3(M):
    """3x3 determinant by expanding along the first row."""
    return sum(((-1) ** j) * M[0][j] * det2(minor_submatrix(M, 0, j))
               for j in range(N))


def op_determinant(app):
    A = app.get_A()
    m = [det2(minor_submatrix(A, 0, j)) for j in range(N)]
    d = det3(A)
    text = ("Expansion along the first row:\n"
            f"  det(A) = ({fmt(A[0][0])})({fmt(m[0])}) - ({fmt(A[0][1])})({fmt(m[1])})"
            f" + ({fmt(A[0][2])})({fmt(m[2])})\n"
            f"  det(A) = {fmt(d)}")
    return text, None
# ===== END OF DETERMINANT OPERATION =====


# ===== START OF INVERSE OPERATION =====
# --- helper steps for the formula  A^-1 = adj(A) / det(A) ---
def minors_matrix(M):
    """Minor M[i][j] = determinant of the 2x2 left after crossing out row i, col j."""
    return [[det2(minor_submatrix(M, i, j)) for j in range(N)] for i in range(N)]


def cofactor_matrix(M):
    """Cofactor = minor with a checkerboard sign  (+ - + / - + - / + - +)."""
    mn = minors_matrix(M)
    return [[((-1) ** (i + j)) * mn[i][j] for j in range(N)] for i in range(N)]


def adjugate(M):
    """Adjoint (adjugate) = transpose of the cofactor matrix."""
    C = cofactor_matrix(M)
    return [[C[j][i] for j in range(N)] for i in range(N)]


def inverse_matrix(M):
    """inverse = adjoint / determinant. Returns None if the matrix is singular."""
    d = det3(M)
    if abs(d) < EPS:
        return None
    adj = adjugate(M)
    return [[adj[i][j] / d for j in range(N)] for i in range(N)]


def op_inverse(app):
    A = app.get_A()
    d = det3(A)
    inv = inverse_matrix(A)
    if inv is None:
        raise ValueError("det(A) = 0, so A has no inverse (it is singular).")
    check = matmul_m(A, inv)
    text = (f"det(A) = {fmt(d)}\n\nA^-1 = adj(A) / det(A) =\n" + mat_str(inv) +
            "\n\nCheck: A x A^-1 =\n" + mat_str(clean(check, 1e-9)))
    return text, inv
# ===== END OF INVERSE OPERATION =====


# ===== START OF TRACE OPERATION =====
def op_trace(app):
    A = app.get_A()
    t = sum(A[i][i] for i in range(N))          # add the main diagonal
    return (f"Trace(A) = {fmt(A[0][0])} + {fmt(A[1][1])} + {fmt(A[2][2])}"
            f" = {fmt(t)}"), None
# ===== END OF TRACE OPERATION =====


# ===== START OF RANK OPERATION =====
def op_rank(app):
    A = app.get_A()
    # rank = number of pivots (non-zero rows) after row reduction
    R, _, pivots = reduce_matrix(A, False)
    text = ("Row echelon form of A:\n" + mat_str(R) +
            f"\n\nRank(A) = {len(pivots)}")
    return text, None
# ===== END OF RANK OPERATION =====


# ===== START OF POWER OPERATION =====
def op_power(app):
    A, n = app.get_A(), app.get_n()
    if abs(n) > 1000:
        raise ValueError("Please use a power between -1000 and 1000.")
    base = A
    if n < 0:                                    # negative power -> use inverse
        base = inverse_matrix(A)
        if base is None:
            raise ValueError("A is singular, so a negative power is impossible.")
    R = identity()                               # A^0 = identity
    for _ in range(abs(n)):                      # multiply repeatedly
        R = matmul_m(R, base)
    return f"A^{n} =\n" + mat_str(R), R
# ===== END OF POWER OPERATION =====


# ===== START OF ROW OPERATION =====
def op_row_operation(app):
    A = app.get_A()
    kind, i, j, k = app.get_rowcol()
    if kind == "swap":
        if i == j:
            raise ValueError("To swap rows, Ri and Rj must be different.")
        A[i], A[j] = A[j], A[i]
        what = f"R{i+1} <-> R{j+1}"
    elif kind == "scale":
        if abs(k) < EPS:
            raise ValueError("k must not be 0 when scaling a row.")
        A[i] = [k * v for v in A[i]]
        what = f"R{i+1} = {fmt(k)} * R{i+1}"
    else:  # add a multiple of another row
        if i == j:
            raise ValueError("To add a multiple of a row, Ri and Rj must be different.")
        A[i] = [A[i][c] + k * A[j][c] for c in range(N)]
        what = f"R{i+1} = R{i+1} + ({fmt(k)}) * R{j+1}"
    return f"Row operation  {what}\n\nResult:\n" + mat_str(A), A
# ===== END OF ROW OPERATION =====


# ===== START OF GAUSS-JORDAN ELIMINATION OPERATION =====
def op_gauss_jordan(app):
    A = app.get_A()
    aug = app.use_augmented()
    M = [A[i] + [app.get_b()[i]] for i in range(N)] if aug else A
    R, steps, pivots = reduce_matrix(M, True)        # forward + backward elimination
    text = ("GAUSS-JORDAN ELIMINATION (reduced row echelon form)"
            + ("  on [A | b]" if aug else "") + "\n\n" + steps_text(steps))
    text += "Final reduced row echelon form:\n" + mat_str(R)
    if aug:
        if N in pivots:
            text += "\n\nThe system has NO solution (inconsistent)."
        elif pivots == [0, 1, 2]:
            text += "\n\nSolution (last column):  x = " + vec_str([R[i][N] for i in range(N)])
        else:
            text += "\n\nThe system has infinitely many solutions."
    return text, (None if aug else R)
# ===== END OF GAUSS-JORDAN ELIMINATION OPERATION =====


# ===== START OF EIGENVALUES AND EIGENVECTORS OPERATION =====
def char_poly(A):
    """Characteristic polynomial  lambda^3 - trace*lambda^2 + c2*lambda - det."""
    tr = A[0][0] + A[1][1] + A[2][2]
    c2 = ((A[0][0] * A[1][1] - A[0][1] * A[1][0]) +
          (A[0][0] * A[2][2] - A[0][2] * A[2][0]) +
          (A[1][1] * A[2][2] - A[1][2] * A[2][1]))
    return [1.0, -tr, c2, -det3(A)]


def poly_str(c):
    out = ""
    for coef, name in zip(c, ["λ³", "λ²", "λ", ""]):
        if abs(coef) < 1e-12:
            continue
        mag = fmt(abs(coef)) if (name == "" or abs(abs(coef) - 1) > 1e-12) else ""
        term = mag + name
        out += ("-" if coef < 0 else "") + term if not out else \
               (" - " if coef < 0 else " + ") + term
    return out + " = 0"


def cubic_roots(b2, b1, b0):
    """Solve  x^3 + b2 x^2 + b1 x + b0 = 0  (Cardano's formula, complex maths)."""
    p = b1 - b2 * b2 / 3
    q = 2 * b2 ** 3 / 27 - b2 * b1 / 3 + b0
    root_d = cmath.sqrt((q / 2) ** 2 + (p / 3) ** 3)
    u3 = max(-q / 2 + root_d, -q / 2 - root_d, key=abs)   # avoid cancellation
    if abs(u3) < 1e-12:
        ts = [0j, 0j, 0j]
    else:
        u = u3 ** (1 / 3)
        v = -p / (3 * u)
        w = cmath.exp(2j * math.pi / 3)
        ts = [u * w ** k + v * w ** (-k) for k in range(3)]
    roots = []
    for t in ts:
        r = t - b2 / 3
        for _ in range(30):                               # Newton polishing
            f = ((r + b2) * r + b1) * r + b0
            df = (3 * r + 2 * b2) * r + b1
            if abs(df) < 1e-12:
                break
            r -= f / df
        roots.append(r)
    return roots


def eigen_pairs(A):
    """Return a list of (eigenvalue, multiplicity, [eigenvectors])."""
    c = char_poly(A)
    roots = []
    for r in cubic_roots(c[1], c[2], c[3]):
        roots.append(r.real if abs(r.imag) < 1e-6 * (1 + abs(r)) else r)
    groups = []                                  # group equal (repeated) roots
    for r in roots:
        for g in groups:
            if abs(g[0] - r) < 1e-4 * (1 + abs(r)):
                g.append(r)
                break
        else:
            groups.append([r])
    pairs = []
    for g in groups:
        val = sum(g) / len(g)
        if len(g) > 1:                           # clean repeated roots
            val = (complex(round(val.real, 6), round(val.imag, 6))
                   if isinstance(val, complex) else round(val, 6))
        shifted = [[A[i][j] - (val if i == j else 0) for j in range(N)]
                   for i in range(N)]
        pairs.append((val, len(g), null_space(shifted)))
    pairs.sort(key=lambda t: (-t[0].real, -(t[0].imag if isinstance(t[0], complex) else 0)))
    return pairs


def op_eigen(app):
    A = app.get_A()
    pairs = eigen_pairs(A)
    lines = ["Characteristic polynomial  det(A - λI) = 0 :",
             "  " + poly_str(char_poly(A)), ""]
    for idx, (val, mult, vecs) in enumerate(pairs, 1):
        lines.append(f"λ{idx} = {fmt(val)}    (algebraic multiplicity {mult})")
        if not vecs:
            lines.append("    (no eigenvector found numerically)")
        for v in vecs:
            lines.append("    eigenvector: " + vec_str(v))
        lines.append("")
    if any(isinstance(p[0], complex) for p in pairs):
        lines.append("Note: complex eigenvalues come in conjugate pairs for a real matrix.")
    return "\n".join(lines), None
# ===== END OF EIGENVALUES AND EIGENVECTORS OPERATION =====


# ===== START OF LU DECOMPOSITION OPERATION =====
def op_lu(app):
    A = app.get_A()
    U = copy_m(A)
    L = [[0.0] * N for _ in range(N)]
    P = identity()
    for k in range(N):
        p = max(range(k, N), key=lambda i: abs(U[i][k]))   # partial pivoting
        if p != k:
            U[k], U[p] = U[p], U[k]
            P[k], P[p] = P[p], P[k]
            L[k], L[p] = L[p], L[k]
        if abs(U[k][k]) < EPS:
            continue
        for i in range(k + 1, N):
            L[i][k] = U[i][k] / U[k][k]                    # multiplier
            U[i] = [U[i][j] - L[i][k] * U[k][j] for j in range(N)]
            U[i][k] = 0.0
    for i in range(N):
        L[i][i] = 1.0
    PA = matmul_m(P, A)
    LU = clean(matmul_m(L, U), 1e-9)
    text = ("P A = L U   (P = row-swap matrix, L = lower, U = upper triangular)\n\n"
            "P =\n" + mat_str(P) + "\n\nL =\n" + mat_str(L) +
            "\n\nU =\n" + mat_str(clean(U)) +
            "\n\nCheck: P A =\n" + mat_str(PA) + "\nCheck: L U =\n" + mat_str(LU))
    return text, None
# ===== END OF LU DECOMPOSITION OPERATION =====


# ===== START OF INVERSE METHOD (SOLVE Ax = b) OPERATION =====
def op_inverse_method(app):
    A, b = app.get_A(), app.get_b()
    inv = inverse_matrix(A)
    if inv is None:
        raise ValueError("det(A) = 0, so the inverse method cannot be used.")
    x = mat_vec(inv, b)                          # x = A^-1 b
    text = ("x = A^-1 b\n\nA^-1 =\n" + mat_str(inv) +
            "\n\nSolution:  x = " + vec_str(x) +
            "\nCheck: A x = " + vec_str(clean([mat_vec(A, x)], 1e-9)[0]))
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
        self.title("3x3 Matrix Calculator - Pure Python (no NumPy)")
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
        return rows            # plain Python lists

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
