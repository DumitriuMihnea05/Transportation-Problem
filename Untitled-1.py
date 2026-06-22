import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk

class TransportSolver:
    def __init__(self, costuri, disponibil, necesar, init_method="NV"):
        self.C_orig = np.array(costuri, dtype=float)
        self.A_orig = np.array(disponibil, dtype=float)
        self.B_orig = np.array(necesar, dtype=float)
        self.init_method = init_method
        
        self.m_orig, self.n_orig = self.C_orig.shape
        
        # Check system balance
        sum_A = np.sum(self.A_orig)
        sum_B = np.sum(self.B_orig)
        self.balanced = True
        
        if abs(sum_A - sum_B) > 1e-7:
            self.balanced = False
            if sum_A > sum_B:
                # Supply surplus: add dummy customer (dummy column) with costs = 0
                self.C = np.zeros((self.m_orig, self.n_orig + 1))
                self.C[:, :self.n_orig] = self.C_orig
                self.A = self.A_orig.copy()
                self.B = np.zeros(self.n_orig + 1)
                self.B[:self.n_orig] = self.B_orig
                self.B[-1] = sum_A - sum_B
                self.m, self.n = self.C.shape
            else:
                # Supply deficit: add dummy supplier (dummy row) with costs = 0
                self.C = np.zeros((self.m_orig + 1, self.n_orig))
                self.C[:self.m_orig, :] = self.C_orig
                self.A = np.zeros(self.m_orig + 1)
                self.A[:self.m_orig] = self.A_orig
                self.A[-1] = sum_B - sum_A
                self.B = self.B_orig.copy()
                self.m, self.n = self.C.shape
        else:
            self.C = self.C_orig.copy()
            self.A = self.A_orig.copy()
            self.B = self.B_orig.copy()
            self.m, self.n = self.C.shape

        self.X = np.zeros((self.m, self.n))
        self.basis = []  # List of tuples (i, j) containing exactly m + n - 1 basic cells
        
        self.iteration = 0
        self.history = []
        self.status = "running"  # Solver status: running, optimal, error, max_iterations
        
        # Run initial BFS (Basic Feasible Solution)
        self.run_bfs()
        self.record_state()

    def run_bfs(self):
        if self.init_method == "NV":
            self.X, self.basis = self.get_bfs_nw()
        elif self.init_method == "MinCost":
            self.X, self.basis = self.get_bfs_min_cost()
        elif self.init_method == "Vogel":
            self.X, self.basis = self.get_bfs_vogel()
        else:
            self.X, self.basis = self.get_bfs_nw()

    def get_bfs_nw(self):
        """North-West Corner Method"""
        m, n = self.m, self.n
        X = np.zeros((m, n))
        basis = []
        a = self.A.copy()
        b = self.B.copy()
        i, j = 0, 0
        while i < m and j < n:
            basis.append((i, j))
            if a[i] < b[j]:
                X[i, j] = a[i]
                b[j] -= a[i]
                a[i] = 0
                i += 1
            elif a[i] > b[j]:
                X[i, j] = b[j]
                a[i] -= b[j]
                b[j] = 0
                j += 1
            else:
                # Equality (degeneracy managed by keeping cell in basis with 0 allocation)
                X[i, j] = a[i]
                a[i] = 0
                b[j] = 0
                if i < m - 1:
                    i += 1
                elif j < n - 1:
                    j += 1
                else:
                    i += 1
                    j += 1
        return X, basis

    def get_bfs_min_cost(self):
        """Minimum Cost Method"""
        m, n = self.m, self.n
        X = np.zeros((m, n))
        basis = []
        a = self.A.copy()
        b = self.B.copy()
        
        rows_free = list(range(m))
        cols_free = list(range(n))
        
        while len(rows_free) + len(cols_free) > 1:
            # Find cell with minimum cost in active rows and columns
            min_c = float('inf')
            best_i, best_j = -1, -1
            for i in rows_free:
                for j in cols_free:
                    if self.C[i, j] < min_c:
                        min_c = self.C[i, j]
                        best_i, best_j = i, j
            
            i, j = best_i, best_j
            basis.append((i, j))
            
            if a[i] < b[j]:
                X[i, j] = a[i]
                b[j] -= a[i]
                a[i] = 0
                rows_free.remove(i)
            elif a[i] > b[j]:
                X[i, j] = b[j]
                a[i] -= b[j]
                b[j] = 0
                cols_free.remove(j)
            else:
                # Equality: remove only one to preserve exact number of basic cells
                X[i, j] = a[i]
                a[i] = 0
                b[j] = 0
                if len(rows_free) > 1:
                    rows_free.remove(i)
                else:
                    cols_free.remove(j)
        return X, basis

    def get_bfs_vogel(self):
        """Vogel's Approximation Method (VAM)"""
        m, n = self.m, self.n
        X = np.zeros((m, n))
        basis = []
        a = self.A.copy()
        b = self.B.copy()
        
        rows_free = list(range(m))
        cols_free = list(range(n))
        
        while len(rows_free) + len(cols_free) > 1:
            # Calculate penalties for active rows
            row_penalties = {}
            for i in rows_free:
                costs = sorted([self.C[i, j] for j in cols_free])
                if len(costs) >= 2:
                    row_penalties[i] = costs[1] - costs[0]
                elif len(costs) == 1:
                    row_penalties[i] = costs[0]
                else:
                    row_penalties[i] = 0
            
            # Calculate penalties for active columns
            col_penalties = {}
            for j in cols_free:
                costs = sorted([self.C[i, j] for i in rows_free])
                if len(costs) >= 2:
                    col_penalties[j] = costs[1] - costs[0]
                elif len(costs) == 1:
                    col_penalties[j] = costs[0]
                else:
                    col_penalties[j] = 0
            
            # Find maximum penalty
            max_penalty = -1
            is_row = True
            idx = -1
            
            for i, p in row_penalties.items():
                if p > max_penalty:
                    max_penalty = p
                    is_row = True
                    idx = i
            for j, p in col_penalties.items():
                if p > max_penalty:
                    max_penalty = p
                    is_row = False
                    idx = j
            
            # In the row/column with maximum penalty, choose the cell with minimum cost
            if is_row:
                i = idx
                j = min(cols_free, key=lambda c: self.C[i, c])
            else:
                j = idx
                i = min(rows_free, key=lambda r: self.C[r, j])
                
            basis.append((i, j))
            
            if a[i] < b[j]:
                X[i, j] = a[i]
                b[j] -= a[i]
                a[i] = 0
                rows_free.remove(i)
            elif a[i] > b[j]:
                X[i, j] = b[j]
                a[i] -= b[j]
                b[j] = 0
                cols_free.remove(j)
            else:
                X[i, j] = a[i]
                a[i] = 0
                b[j] = 0
                if len(rows_free) > 1:
                    rows_free.remove(i)
                else:
                    cols_free.remove(j)
        return X, basis

    def calculeaza_potentiale(self):
        """Determine potentials u and v by propagation on the basis tree"""
        u = [None] * self.m
        v = [None] * self.n
        u[0] = 0.0
        
        # Propagate values until all potentials are determined
        changed = True
        while changed:
            changed = False
            for i, j in self.basis:
                if u[i] is not None and v[j] is None:
                    v[j] = float(self.C[i, j] - u[i])
                    changed = True
                elif v[j] is not None and u[i] is None:
                    u[i] = float(self.C[i, j] - v[j])
                    changed = True
        
        # Safety fallback
        for i in range(self.m):
            if u[i] is None: u[i] = 0.0
        for j in range(self.n):
            if v[j] is None: v[j] = 0.0
            
        return np.array(u), np.array(v)

    def record_state(self, entering=None, leaving=None, circuit=None, theta=None):
        u, v = self.calculeaza_potentiale()
        C_tilde = u[:, np.newaxis] + v
        delta = self.C - C_tilde
        cost_total = np.sum(self.X * self.C)
        
        self.history.append({
            'iteration': self.iteration,
            'X': self.X.copy(),
            'basis': list(self.basis),
            'u': u,
            'v': v,
            'C_tilde': C_tilde,
            'delta': delta.copy(),
            'cost': cost_total,
            'entering': entering,
            'leaving': leaving,
            'circuit': circuit,
            'theta': theta,
            'status': self.status
        })

    def gaseste_circuit(self, start_node):
        """Identify the unique alternating cycle formed by the entering cell"""
        path = [start_node]
        
        def dfs(curr, horizontal):
            if len(path) > 3:
                # Check if loop can close at start_node
                if horizontal and curr[0] == start_node[0]:
                    return True
                if not horizontal and curr[1] == start_node[1]:
                    return True
            
            for r, c in self.basis:
                if (r, c) in path:
                    continue
                if horizontal and r == curr[0]:
                    # Move horizontally (same row, other column)
                    path.append((r, c))
                    if dfs((r, c), False):
                        return True
                    path.pop()
                elif not horizontal and c == curr[1]:
                    # Move vertically (same column, other row)
                    path.append((r, c))
                    if dfs((r, c), True):
                        return True
                    path.pop()
            return False
            
        # Try both starting directions
        if dfs(start_node, True):
            return path
        if dfs(start_node, False):
            return path
        return None

    def step(self):
        """Execute a single step of the MODI algorithm"""
        if self.status != "running":
            return False
            
        if self.iteration >= 50:
            self.status = "max_iterations"
            self.history[-1]['status'] = "max_iterations"
            return False
            
        u, v = self.calculeaza_potentiale()
        C_tilde = u[:, np.newaxis] + v
        delta = self.C - C_tilde
        
        # Optimality test: All delta_ij >= 0
        min_val = np.min(delta)
        if min_val >= -1e-9:
            self.status = "optimal"
            self.history[-1]['status'] = "optimal"
            return False
            
        # Select entering cell (most negative difference)
        p, q = np.unravel_index(np.argmin(delta), delta.shape)
        entering = (int(p), int(q))
        
        # Find cycle formed by entering cell
        circuit = self.gaseste_circuit(entering)
        if not circuit:
            self.status = "error"
            self.history[-1]['status'] = "error"
            return False
            
        # Negative cells (odd indices of cycle)
        minus_cells = circuit[1::2]
        theta = min(self.X[r, c] for r, c in minus_cells)
        
        # Select leaving cell (first that hits theta)
        leaving = None
        for r, c in minus_cells:
            if abs(self.X[r, c] - theta) < 1e-9:
                leaving = (int(r), int(c))
                break
                
        # Update allocations
        for idx, (r, c) in enumerate(circuit):
            if idx % 2 == 0:
                self.X[r, c] += theta
            else:
                self.X[r, c] -= theta
                
        # Update basis (remove leaving, add entering)
        self.basis.remove(leaving)
        self.basis.append(entering)
        
        self.iteration += 1
        
        # Check if new state is optimal
        new_u, new_v = self.calculeaza_potentiale()
        new_C_tilde = new_u[:, np.newaxis] + new_v
        new_delta = self.C - new_C_tilde
        if np.min(new_delta) >= -1e-9:
            self.status = "optimal"
            
        self.record_state(entering=entering, leaving=leaving, circuit=circuit, theta=theta)
        return True


def format_val(val):
    if val is None:
        return "—"
    if abs(val - round(val)) < 1e-7:
        return f"{int(round(val))}"
    return f"{val:.2f}"


class AppTransportPro:
    def __init__(self, root):
        self.root = root
        self.root.title("Expert System - Transportation Optimization (MODI)")
        self.root.geometry("1200x820")
        self.root.minsize(1050, 700)
        
        # Zinc/Slate style colors (Dark Theme)
        self.bg_dark = "#09090b"
        self.bg_card = "#18181b"
        self.border_color = "#27272a"
        self.accent_blue = "#3b82f6"
        self.accent_purple = "#a855f7"
        self.text_light = "#f4f4f5"
        self.text_dark = "#71717a"
        self.highlight_green = "#10b981"
        self.highlight_gold = "#f59e0b"
        self.highlight_red = "#ef4444"
        
        self.root.configure(bg=self.bg_dark)
        
        # Setup ttk styles
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('.', background=self.bg_dark, foreground=self.text_light)
        self.style.configure('TLabel', background=self.bg_dark, foreground=self.text_light, font=("Segoe UI", 10))
        self.style.configure('TFrame', background=self.bg_dark)
        self.style.configure('Card.TFrame', background=self.bg_card)
        self.style.configure('Vertical.TScrollbar', gripcount=0, background=self.bg_card, troughcolor=self.bg_dark)
        
        self.m, self.n = 3, 4
        self.solver = None
        
        self._build_layout()
        self.load_default_values()

    def _build_layout(self):
        # Horizontal PanedWindow
        self.main_pane = tk.PanedWindow(self.root, orient="horizontal", bg=self.bg_dark, bd=0, sashwidth=5)
        self.main_pane.pack(fill="both", expand=True, padx=15, pady=15)
        
        # --- LEFT PANEL: CONFIG & DATA ---
        self.left_frame = tk.Frame(self.main_pane, bg=self.bg_dark, width=480)
        self.left_frame.pack_propagate(False)
        self.main_pane.add(self.left_frame, minsize=420)
        
        # Header title
        tk.Label(self.left_frame, text="TRANSPORTATION PROBLEM", font=("Segoe UI Semibold", 13), bg=self.bg_dark, fg=self.accent_blue).pack(anchor="w", pady=(0, 10))
        
        # Config card: Dimensions & BFS Method
        config_card = tk.Frame(self.left_frame, bg=self.bg_card, bd=1, relief="solid", highlightbackground=self.border_color, highlightcolor=self.border_color, highlightthickness=1)
        config_card.pack(fill="x", pady=(0, 10))
        
        cfg_inner = tk.Frame(config_card, bg=self.bg_card, padx=12, pady=12)
        cfg_inner.pack(fill="x")
        
        # Dimensions selectors M and N
        dims_frame = tk.Frame(cfg_inner, bg=self.bg_card)
        dims_frame.pack(fill="x", pady=(0, 8))
        
        tk.Label(dims_frame, text="M (Suppliers):", font=("Segoe UI", 9, "bold"), bg=self.bg_card, fg=self.text_dark).pack(side="left")
        self.spin_m = tk.Spinbox(dims_frame, from_=2, to=6, width=4, buttonbackground=self.bg_card, bg=self.bg_dark, fg=self.text_light, insertbackground=self.text_light, bd=0, highlightthickness=1, highlightbackground=self.border_color, justify='center')
        self.spin_m.pack(side="left", padx=5)
        self.spin_m.delete(0, "end"); self.spin_m.insert(0, str(self.m))
        
        tk.Label(dims_frame, text="N (Customers):", font=("Segoe UI", 9, "bold"), bg=self.bg_card, fg=self.text_dark).pack(side="left", padx=(15, 0))
        self.spin_n = tk.Spinbox(dims_frame, from_=2, to=6, width=4, buttonbackground=self.bg_card, bg=self.bg_dark, fg=self.text_light, insertbackground=self.text_light, bd=0, highlightthickness=1, highlightbackground=self.border_color, justify='center')
        self.spin_n.pack(side="left", padx=5)
        self.spin_n.delete(0, "end"); self.spin_n.insert(0, str(self.n))
        
        btn_resize = tk.Button(dims_frame, text="Resize Grid", command=self.resize_grid, bg=self.bg_card, fg=self.text_light, activebackground=self.bg_dark, activeforeground=self.text_light, bd=1, relief="solid", font=("Segoe UI Semibold", 8), cursor="hand2", padx=8)
        btn_resize.pack(side="right")
        self.bind_hover(btn_resize, self.bg_dark, self.bg_card)
        
        # BFS Method select
        method_frame = tk.Frame(cfg_inner, bg=self.bg_card)
        method_frame.pack(fill="x")
        
        tk.Label(method_frame, text="Initial BFS Method:", font=("Segoe UI", 9, "bold"), bg=self.bg_card, fg=self.text_dark).pack(side="left")
        self.method_var = tk.StringVar(value="North-West Corner")
        self.method_combo = ttk.Combobox(method_frame, textvariable=self.method_var, values=["North-West Corner", "Minimum Cost", "Vogel's Approximation (VAM)"], state="readonly", width=26)
        self.method_combo.pack(side="left", padx=10)
        
        # Card for input grid
        self.grid_card = tk.Frame(self.left_frame, bg=self.bg_card, bd=1, relief="solid", highlightbackground=self.border_color, highlightcolor=self.border_color, highlightthickness=1)
        self.grid_card.pack(fill="both", expand=True, pady=(0, 10))
        
        self.grid_inner = tk.Frame(self.grid_card, bg=self.bg_card, padx=12, pady=12)
        self.grid_inner.pack(fill="both", expand=True)
        
        # Draw grid
        self.draw_grid()
        
        # Sub-buttons in left panel
        aux_frame = tk.Frame(self.left_frame, bg=self.bg_dark)
        aux_frame.pack(fill="x", pady=(0, 10))
        
        btn_default = tk.Button(aux_frame, text="Load Example", command=self.load_default_values, bg=self.bg_card, fg=self.accent_blue, activebackground=self.bg_dark, activeforeground=self.accent_blue, bd=1, relief="solid", font=("Segoe UI Semibold", 9), cursor="hand2", pady=4)
        btn_default.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.bind_hover(btn_default, self.bg_dark, self.bg_card)
        
        btn_clear = tk.Button(aux_frame, text="Clear Fields", command=self.clear_all, bg=self.bg_card, fg=self.highlight_red, activebackground=self.bg_dark, activeforeground=self.highlight_red, bd=1, relief="solid", font=("Segoe UI Semibold", 9), cursor="hand2", pady=4)
        btn_clear.pack(side="left", fill="x", expand=True, padx=(5, 0))
        self.bind_hover(btn_clear, self.bg_dark, self.bg_card)
        
        # Solver control panel
        ctrl_card = tk.Frame(self.left_frame, bg=self.bg_card, bd=1, relief="solid", highlightbackground=self.border_color, highlightcolor=self.border_color, highlightthickness=1)
        ctrl_card.pack(fill="x")
        
        ctrl_inner = tk.Frame(ctrl_card, bg=self.bg_card, padx=12, pady=12)
        ctrl_inner.pack(fill="x")
        
        self.btn_init = tk.Button(ctrl_inner, text="Initialize Solver", command=self.initialize_solver, bg=self.accent_blue, fg=self.text_light, activebackground="#2563eb", activeforeground=self.text_light, bd=0, font=("Segoe UI Semibold", 10), cursor="hand2", pady=8)
        self.btn_init.pack(fill="x", pady=(0, 6))
        
        step_frame = tk.Frame(ctrl_inner, bg=self.bg_card)
        step_frame.pack(fill="x")
        
        self.btn_step = tk.Button(step_frame, text="Next Step", command=self.solver_step, state="disabled", bg=self.bg_card, fg=self.text_light, activebackground=self.bg_dark, activeforeground=self.accent_blue, bd=1, relief="solid", font=("Segoe UI Semibold", 9), cursor="hand2", pady=6)
        self.btn_step.pack(side="left", fill="x", expand=True, padx=(0, 3))
        self.bind_hover(self.btn_step, self.bg_dark, self.bg_card)
        
        self.btn_solve = tk.Button(step_frame, text="Solve All", command=self.solver_solve_all, state="disabled", bg=self.bg_card, fg=self.text_light, activebackground=self.bg_dark, activeforeground=self.highlight_green, bd=1, relief="solid", font=("Segoe UI Semibold", 9), cursor="hand2", pady=6)
        self.btn_solve.pack(side="left", fill="x", expand=True, padx=(3, 0))
        self.bind_hover(self.btn_solve, self.bg_dark, self.bg_card)
        
        self.btn_reset = tk.Button(ctrl_inner, text="Reset Solver", command=self.reset_solver, state="disabled", bg=self.bg_card, fg=self.highlight_gold, activebackground=self.bg_dark, activeforeground=self.highlight_gold, bd=1, relief="solid", font=("Segoe UI Semibold", 9), cursor="hand2", pady=6)
        self.btn_reset.pack(fill="x", pady=(6, 0))
        self.bind_hover(self.btn_reset, self.bg_dark, self.bg_card)
        
        # --- RIGHT PANEL: DETAILED LOGS & RESULTS ---
        self.right_frame = tk.Frame(self.main_pane, bg=self.bg_dark)
        self.main_pane.add(self.right_frame, minsize=500)
        
        tk.Label(self.right_frame, text="DETAILED LOGS / ITERATIONS", font=("Segoe UI Semibold", 13), bg=self.bg_dark, fg=self.accent_purple).pack(anchor="w", pady=(0, 10))
        
        text_card = tk.Frame(self.right_frame, bg=self.bg_card, bd=1, relief="solid", highlightbackground=self.border_color, highlightcolor=self.border_color, highlightthickness=1)
        text_card.pack(fill="both", expand=True)
        
        # Scrollable Text area
        self.txt = tk.Text(text_card, wrap="none", font=("Consolas", 10), bg=self.bg_card, fg=self.text_light, insertbackground=self.text_light, padx=15, pady=15, relief="flat", highlightthickness=0)
        sy = ttk.Scrollbar(text_card, orient="vertical", command=self.txt.yview)
        sx = ttk.Scrollbar(text_card, orient="horizontal", command=self.txt.xview)
        self.txt.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
        
        self.txt.pack(side="top", fill="both", expand=True)
        sy.pack(side="right", fill="y")
        sx.pack(side="bottom", fill="x")
        
        # Setup tags
        self.txt.tag_configure("section", foreground=self.accent_blue, font=("Consolas", 11, "bold"))
        self.txt.tag_configure("label", foreground=self.accent_purple, font=("Consolas", 10, "bold"))
        self.txt.tag_configure("success", foreground=self.highlight_green, font=("Consolas", 11, "bold"))
        self.txt.tag_configure("warning", foreground=self.highlight_gold, font=("Consolas", 10, "bold"))
        self.txt.tag_configure("error", foreground=self.highlight_red, font=("Consolas", 10, "bold"))
        self.txt.tag_configure("info", foreground=self.text_light, font=("Consolas", 10))
        self.txt.tag_configure("matrix", foreground=self.text_light, font=("Consolas", 10))

    def bind_hover(self, widget, hover_bg, normal_bg):
        widget.bind("<Enter>", lambda e: widget.configure(bg=hover_bg) if widget["state"] != "disabled" else None)
        widget.bind("<Leave>", lambda e: widget.configure(bg=normal_bg) if widget["state"] != "disabled" else None)

    def draw_grid(self):
        for widget in self.grid_inner.winfo_children():
            widget.destroy()
            
        tk.Label(self.grid_inner, text="Costs Table", font=("Segoe UI Semibold", 10), bg=self.bg_card, fg=self.text_light).grid(row=0, column=0, columnspan=self.n+2, pady=(0, 10), sticky="w")
        
        # Columns B1, B2...
        for j in range(self.n):
            tk.Label(self.grid_inner, text=f"B{j+1}", font=("Segoe UI", 9, "bold"), bg=self.bg_card, fg=self.accent_purple).grid(row=1, column=j+1, padx=4, pady=5)
        # Supply header
        tk.Label(self.grid_inner, text="Supply (A)", font=("Segoe UI", 9, "bold"), bg=self.bg_card, fg=self.accent_blue).grid(row=1, column=self.n+1, padx=8, pady=5)
        
        self.cost_entries = []
        self.disp_entries = []
        for i in range(self.m):
            tk.Label(self.grid_inner, text=f"A{i+1}", font=("Segoe UI", 9, "bold"), bg=self.bg_card, fg=self.accent_blue).grid(row=i+2, column=0, padx=5, pady=3)
            row_entries = []
            for j in range(self.n):
                e = tk.Entry(self.grid_inner, width=7, justify='center', font=("Consolas", 10), bg=self.bg_dark, fg=self.text_light, insertbackground=self.text_light, bd=0, highlightthickness=1, highlightbackground=self.border_color, highlightcolor=self.accent_purple)
                e.grid(row=i+2, column=j+1, padx=3, pady=3)
                row_entries.append(e)
            self.cost_entries.append(row_entries)
            
            # Supply entry
            e_disp = tk.Entry(self.grid_inner, width=9, justify='center', font=("Consolas", 10, "bold"), bg=self.bg_dark, fg=self.accent_blue, insertbackground=self.text_light, bd=0, highlightthickness=1, highlightbackground=self.border_color, highlightcolor=self.accent_blue)
            e_disp.grid(row=i+2, column=self.n+1, padx=8, pady=3)
            self.disp_entries.append(e_disp)
            
        # Demand row labels
        tk.Label(self.grid_inner, text="Demand (B)", font=("Segoe UI", 9, "bold"), bg=self.bg_card, fg=self.accent_purple).grid(row=self.m+2, column=0, padx=5, pady=8)
        self.nec_entries = []
        for j in range(self.n):
            e_nec = tk.Entry(self.grid_inner, width=7, justify='center', font=("Consolas", 10, "bold"), bg=self.bg_dark, fg=self.accent_purple, insertbackground=self.text_light, bd=0, highlightthickness=1, highlightbackground=self.border_color, highlightcolor=self.accent_purple)
            e_nec.grid(row=self.m+2, column=j+1, padx=3, pady=8)
            self.nec_entries.append(e_nec)

    def resize_grid(self):
        try:
            m = int(self.spin_m.get())
            n = int(self.spin_n.get())
            if not (2 <= m <= 6) or not (2 <= n <= 6):
                raise ValueError("Dimensions must be between 2 and 6.")
            self.m = m
            self.n = n
            self.draw_grid()
            self.load_default_values()
        except ValueError:
            messagebox.showerror("Invalid Dimensions", "Please input valid integers between 2 and 6.")

    def read_grid_data(self):
        try:
            C = []
            for i in range(self.m):
                row = []
                for j in range(self.n):
                    val_str = self.cost_entries[i][j].get().strip()
                    if not val_str:
                        raise ValueError(f"Cost for A{i+1} -> B{j+1} is empty.")
                    val = float(val_str)
                    if val < 0:
                        raise ValueError(f"Cost for A{i+1} -> B{j+1} cannot be negative.")
                    row.append(val)
                C.append(row)
                
            A = []
            for i in range(self.m):
                val_str = self.disp_entries[i].get().strip()
                if not val_str:
                    raise ValueError(f"Supply A{i+1} is empty.")
                val = float(val_str)
                if val < 0:
                    raise ValueError(f"Supply A{i+1} cannot be negative.")
                A.append(val)
                
            B = []
            for j in range(self.n):
                val_str = self.nec_entries[j].get().strip()
                if not val_str:
                    raise ValueError(f"Demand B{j+1} is empty.")
                val = float(val_str)
                if val < 0:
                    raise ValueError(f"Demand B{j+1} cannot be negative.")
                B.append(val)
                
            return C, A, B
        except ValueError as e:
            messagebox.showerror("Data Error", str(e))
            return None, None, None

    def load_default_values(self):
        m, n = self.m, self.n
        self.clear_all()
        
        default_C = [
            [2, 3, 11, 7],
            [1, 0, 6, 1],
            [5, 8, 15, 9],
            [6, 4, 3, 8],
            [3, 5, 2, 7],
            [8, 2, 9, 4]
        ]
        default_A = [6, 1, 10, 8, 15, 12]
        default_B = [7, 5, 3, 2, 10, 8]
        
        for i in range(m):
            for j in range(n):
                if i < len(default_C) and j < len(default_C[0]):
                    val = default_C[i][j]
                else:
                    val = (i + 1) * (j + 1)
                self.cost_entries[i][j].insert(0, str(val))
                
        for i in range(m):
            if i < len(default_A):
                val = default_A[i]
            else:
                val = (i + 1) * 10
            self.disp_entries[i].insert(0, str(val))
            
        for j in range(n):
            if j < len(default_B):
                val = default_B[j]
            else:
                val = (j + 1) * 5
            self.nec_entries[j].insert(0, str(val))

    def clear_all(self):
        for i in range(self.m):
            for j in range(self.n):
                self.cost_entries[i][j].delete(0, tk.END)
        for i in range(self.m):
            self.disp_entries[i].delete(0, tk.END)
        for j in range(self.n):
            self.nec_entries[j].delete(0, tk.END)

    def format_allocation_matrix(self, X, basis):
        m, n = X.shape
        res = "          " + "".join([f"  B{j+1:<7}" if (self.solver.balanced or j != self.solver.n - 1 or self.solver.m_orig == self.solver.m) else f"  Dummy" for j in range(n)]) + "\n"
        res += "        " + "—" * (9 * n + 5) + "\n"
        for i in range(m):
            lbl = f"A{i+1:<3}" if (self.solver.balanced or i != self.solver.m - 1 or self.solver.n_orig == self.solver.n) else "Dummy"
            row_str = f"{lbl:<6} | "
            for j in range(n):
                val = X[i, j]
                formatted = format_val(val)
                if (i, j) in basis:
                    row_str += f"[{formatted:>5}] "
                else:
                    row_str += f" {formatted:>5}  "
            res += row_str + "\n"
        return res

    def format_potentials_and_costs(self, u, v, C):
        m, n = C.shape
        res = "          " + "".join([f"  B{j+1:<7}" if (self.solver.balanced or j != self.solver.n - 1 or self.solver.m_orig == self.solver.m) else f"  Dummy" for j in range(n)]) + " |   u_i\n"
        res += "        " + "—" * (9 * n + 16) + "\n"
        for i in range(m):
            lbl = f"A{i+1:<3}" if (self.solver.balanced or i != self.solver.m - 1 or self.solver.n_orig == self.solver.n) else "Dummy"
            row_str = f"{lbl:<6} | "
            for j in range(n):
                val = C[i, j]
                row_str += f"  {format_val(val):>5}  "
            row_str += f" |  {format_val(u[i]):>5}"
            res += row_str + "\n"
        res += "        " + "—" * (9 * n + 16) + "\n"
        res += "  v_j    | " + "".join([f"  {format_val(v[j]):>5}  " for j in range(n)]) + "\n"
        return res

    def format_delta_matrix(self, delta, basis):
        m, n = delta.shape
        res = "          " + "".join([f"  B{j+1:<7}" if (self.solver.balanced or j != self.solver.n - 1 or self.solver.m_orig == self.solver.m) else f"  Dummy" for j in range(n)]) + "\n"
        res += "        " + "—" * (9 * n + 5) + "\n"
        for i in range(m):
            lbl = f"A{i+1:<3}" if (self.solver.balanced or i != self.solver.m - 1 or self.solver.n_orig == self.solver.n) else "Dummy"
            row_str = f"{lbl:<6} | "
            for j in range(n):
                val = delta[i, j]
                formatted = format_val(val)
                if (i, j) in basis:
                    row_str += " [    0] "
                else:
                    row_str += f"  {formatted:>5}  "
            res += row_str + "\n"
        return res

    def initialize_solver(self):
        C, A, B = self.read_grid_data()
        if C is None:
            return
            
        method_str = self.method_combo.get()
        method_map = {
            "North-West Corner": "NV",
            "Minimum Cost": "MinCost",
            "Vogel's Approximation (VAM)": "Vogel"
        }
        init_method = method_map.get(method_str, "NV")
        
        self.txt.delete(1.0, tk.END)
        self.txt.insert(tk.END, "=== TRANSPORT SOLVER INITIALIZATION ===\n\n", "section")
        
        # Instantiate solver
        self.solver = TransportSolver(C, A, B, init_method)
        
        sA, sB = sum(A), sum(B)
        self.txt.insert(tk.END, f"Total Supply ΣA = {format_val(sA)} | Total Demand ΣB = {format_val(sB)}\n")
        if self.solver.balanced:
            self.txt.insert(tk.END, "System is balanced.\n\n", "success")
        else:
            diff = abs(sA - sB)
            self.txt.insert(tk.END, f"System is UNBALANCED (difference: {format_val(diff)}).\n", "warning")
            if sA > sB:
                self.txt.insert(tk.END, f" -> Added a dummy customer (B{self.solver.n}) with demand = {format_val(diff)} and zero costs.\n\n", "info")
            else:
                self.txt.insert(tk.END, f" -> Added a dummy supplier (A{self.solver.m}) with supply = {format_val(diff)} and zero costs.\n\n", "info")
                
        # Disable changes during solving
        self.set_grid_state("disabled")
        
        self.btn_init.configure(state="disabled", bg=self.bg_card)
        self.btn_step.configure(state="normal")
        self.btn_solve.configure(state="normal")
        self.btn_reset.configure(state="normal")
        
        # Display first step (BFS)
        self.display_step(0)

    def set_grid_state(self, state):
        self.spin_m.configure(state=state)
        self.spin_n.configure(state=state)
        for i in range(self.m):
            self.disp_entries[i].configure(state=state)
            for j in range(self.n):
                self.cost_entries[i][j].configure(state=state)
        for j in range(self.n):
            self.nec_entries[j].configure(state=state)

    def display_step(self, step_idx):
        if not self.solver or step_idx >= len(self.solver.history):
            return
            
        r = self.solver.history[step_idx]
        k = r['iteration']
        
        if k == 0:
            self.txt.insert(tk.END, f">>> INITIAL BASIC FEASIBLE SOLUTION (I{k}) — Method: {self.method_combo.get()}\n", "section")
        else:
            self.txt.insert(tk.END, f">>> ITERATION I{k} — MODI Optimization\n", "section")
            
        # 1. Allocation Matrix
        self.txt.insert(tk.END, "1. Current Allocation Matrix (basic cells are in square brackets):\n", "label")
        self.txt.insert(tk.END, self.format_allocation_matrix(r['X'], r['basis']), "matrix")
        
        # Positive components
        m, n = r['X'].shape
        components = []
        for i in range(m):
            for j in range(n):
                if (i, j) in r['basis'] and r['X'][i, j] > 1e-9:
                    lbl_i = f"A{i+1}" if (self.solver.balanced or i != self.solver.m - 1 or self.solver.m_orig == self.solver.m) else "Dummy"
                    lbl_j = f"B{j+1}" if (self.solver.balanced or j != self.solver.n - 1 or self.solver.n_orig == self.solver.n) else "Dummy"
                    components.append(f"x({lbl_i},{lbl_j}) = {format_val(r['X'][i, j])}")
        self.txt.insert(tk.END, "   Positive allocations: " + ", ".join(components) + "\n")
        self.txt.insert(tk.END, f"   Current transportation cost: f{k} = {format_val(r['cost'])}\n\n", "info")
        
        # 2. Potentials u and v
        self.txt.insert(tk.END, "2. Original Costs Matrix (C) and Potentials u and v:\n", "label")
        self.txt.insert(tk.END, self.format_potentials_and_costs(r['u'], r['v'], self.solver.C), "matrix")
        self.txt.insert(tk.END, "\n")
        
        # 3. Delta
        self.txt.insert(tk.END, "3. Differences Matrix Delta (Δ_ij = C_ij - u_i - v_j):\n", "label")
        self.txt.insert(tk.END, self.format_delta_matrix(r['delta'], r['basis']), "matrix")
        
        # Status
        if r['status'] == "optimal":
            self.txt.insert(tk.END, "Optimality Test: All Δ_ij ≥ 0 for non-basic cells. OPTIMAL SOLUTION FOUND!\n\n", "success")
        elif r['status'] == "max_iterations":
            self.txt.insert(tk.END, "Optimality Test: Maximum iteration limit reached.\n\n", "warning")
        elif r['status'] == "error":
            self.txt.insert(tk.END, "Error: Solving failed due to cycle calculation loop.\n\n", "error")
        else:
            self.txt.insert(tk.END, "Optimality Test: There are negative Δ_ij values. The solution is not optimal.\n", "info")
            if r['entering'] is not None:
                p, q = r['entering']
                lbl_p = f"A{p+1}" if (self.solver.balanced or p != self.solver.m - 1 or self.solver.m_orig == self.solver.m) else "Dummy"
                lbl_q = f"B{q+1}" if (self.solver.balanced or q != self.solver.n - 1 or self.solver.n_orig == self.solver.n) else "Dummy"
                self.txt.insert(tk.END, f"   -> Entering cell: ({lbl_p}, {lbl_q}) with Δ = {format_val(r['delta'][p, q])}\n", "info")
                
            if r['circuit'] is not None:
                circuit_str = " -> ".join([
                    f"({f'A{r+1}' if (self.solver.balanced or r != self.solver.m - 1 or self.solver.m_orig == self.solver.m) else 'Dummy'}, {f'B{c+1}' if (self.solver.balanced or c != self.solver.n - 1 or self.solver.n_orig == self.solver.n) else 'Dummy'})[{'+' if idx%2==0 else '-'}]"
                    for idx, (r, c) in enumerate(r['circuit'])
                ])
                start_r, start_c = r['circuit'][0]
                lbl_sr = f"A{start_r+1}" if (self.solver.balanced or start_r != self.solver.m - 1 or self.solver.m_orig == self.solver.m) else 'Dummy'
                lbl_sc = f"B{start_c+1}" if (self.solver.balanced or start_c != self.solver.n - 1 or self.solver.n_orig == self.solver.n) else 'Dummy'
                circuit_str += f" -> ({lbl_sr}, {lbl_sc})"
                self.txt.insert(tk.END, f"   -> Alternating loop cycle: {circuit_str}\n", "info")
                
            if r['theta'] is not None:
                self.txt.insert(tk.END, f"   -> Transferred amount (θ): {format_val(r['theta'])}\n", "info")
                
            if r['leaving'] is not None:
                lp, lq = r['leaving']
                lbl_lp = f"A{lp+1}" if (self.solver.balanced or lp != self.solver.m - 1 or self.solver.m_orig == self.solver.m) else "Dummy"
                lbl_lq = f"B{lq+1}" if (self.solver.balanced or lq != self.solver.n - 1 or self.solver.n_orig == self.solver.n) else "Dummy"
                self.txt.insert(tk.END, f"   -> Leaving cell: ({lbl_lp}, {lbl_lq})\n", "info")
            self.txt.insert(tk.END, "\n")
            
        self.txt.insert(tk.END, "═" * 80 + "\n\n", "matrix")
        self.txt.see(tk.END)

    def solver_step(self):
        if not self.solver:
            return
            
        res = self.solver.step()
        self.display_step(len(self.solver.history) - 1)
        
        if not res or self.solver.status != "running":
            self.btn_step.configure(state="disabled")
            self.btn_solve.configure(state="disabled")
            self.display_final_result()

    def solver_solve_all(self):
        if not self.solver:
            return
            
        while self.solver.step():
            self.display_step(len(self.solver.history) - 1)
            
        self.btn_step.configure(state="disabled")
        self.btn_solve.configure(state="disabled")
        self.display_final_result()

    def display_final_result(self):
        if not self.solver or len(self.solver.history) == 0:
            return
            
        r = self.solver.history[-1]
        
        self.txt.insert(tk.END, "================== FINAL RESULT ==================\n", "success")
        self.txt.insert(tk.END, f"Optimal Transportation Cost: f* = {format_val(r['cost'])}\n", "success")
        self.txt.insert(tk.END, "Final Allocation Matrix:\n", "label")
        self.txt.insert(tk.END, self.format_allocation_matrix(r['X'], r['basis']), "matrix")
        
        self.txt.insert(tk.END, "Detailed optimal transportation plan:\n", "label")
        m, n = r['X'].shape
        components = []
        for i in range(m):
            for j in range(n):
                if r['X'][i, j] > 1e-9:
                    is_dummy = False
                    if not self.solver.balanced:
                        if self.solver.m_orig != self.solver.m and i == self.solver.m - 1:
                            is_dummy = True
                        if self.solver.n_orig != self.solver.n and j == self.solver.n - 1:
                            is_dummy = True
                            
                    lbl_i = f"A{i+1}" if (self.solver.balanced or i != self.solver.m - 1 or self.solver.m_orig == self.solver.m) else "Dummy"
                    lbl_j = f"B{j+1}" if (self.solver.balanced or j != self.solver.n - 1 or self.solver.n_orig == self.solver.n) else "Dummy"
                    
                    line = f"   - From Supplier {lbl_i} to Customer {lbl_j}: quantity = {format_val(r['X'][i, j])}"
                    if is_dummy:
                        line += " (Dummy - unallocated surplus/demand)"
                    components.append(line)
                    
        self.txt.insert(tk.END, "\n".join(components) + "\n\n")
        self.txt.insert(tk.END, "═" * 80 + "\n\n", "matrix")
        self.txt.see(tk.END)

    def reset_solver(self):
        self.solver = None
        self.txt.delete(1.0, tk.END)
        self.set_grid_state("normal")
        
        self.btn_init.configure(state="normal", bg=self.accent_blue)
        self.btn_step.configure(state="disabled")
        self.btn_solve.configure(state="disabled")
        self.btn_reset.configure(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    app = AppTransportPro(root)
    root.mainloop()