import math

import tensorplay as tp

from ._utils import max_abs, scalar_value
from .optimizer import Optimizer


def _as_loss(value):
    return float(value.item()) if isinstance(value, tp.Tensor) else float(value)


def _dot(lhs, rhs):
    return float((lhs * rhs).sum().item())


def _cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=None):
    """Torch's two-point cubic interpolation used by strong Wolfe search."""

    if bounds is not None:
        xmin_bound, xmax_bound = bounds
    else:
        xmin_bound, xmax_bound = (x1, x2) if x1 <= x2 else (x2, x1)

    d1 = g1 + g2 - 3.0 * (f1 - f2) / (x1 - x2)
    d2_square = d1 * d1 - g1 * g2
    if d2_square >= 0.0:
        d2 = math.sqrt(d2_square)
        if x1 <= x2:
            min_pos = x2 - (x2 - x1) * (
                (g2 + d2 - d1) / (g2 - g1 + 2.0 * d2)
            )
        else:
            min_pos = x1 - (x1 - x2) * (
                (g1 + d2 - d1) / (g1 - g2 + 2.0 * d2)
            )
        return min(max(min_pos, xmin_bound), xmax_bound)
    return (xmin_bound + xmax_bound) / 2.0


def _strong_wolfe(
    obj_func,
    x,
    t,
    d,
    f,
    g,
    gtd,
    c1=1e-4,
    c2=0.9,
    tolerance_change=1e-9,
    max_ls=25,
):
    """Direct port of ``torch.optim.lbfgs._strong_wolfe``."""

    d_norm = max_abs(d)
    g = g.clone()
    f_new, g_new = obj_func(x, t, d)
    ls_func_evals = 1
    gtd_new = _dot(g_new, d)

    t_prev, f_prev, g_prev, gtd_prev = 0.0, f, g, gtd
    done = False
    ls_iter = 0

    while ls_iter < max_ls:
        if f_new > f + c1 * t * gtd or (
            ls_iter > 1 and f_new >= f_prev
        ):
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone()]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        if abs(gtd_new) <= -c2 * gtd:
            bracket = [t]
            bracket_f = [f_new]
            bracket_g = [g_new]
            done = True
            break

        if gtd_new >= 0.0:
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone()]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        min_step = t + 0.01 * (t - t_prev)
        max_step = t * 10.0
        tmp = t
        t = _cubic_interpolate(
            t_prev,
            f_prev,
            gtd_prev,
            t,
            f_new,
            gtd_new,
            bounds=(min_step, max_step),
        )

        t_prev = tmp
        f_prev = f_new
        g_prev = g_new.clone()
        gtd_prev = gtd_new
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = _dot(g_new, d)
        ls_iter += 1

    if ls_iter == max_ls:
        bracket = [0.0, t]
        bracket_f = [f, f_new]
        bracket_g = [g, g_new]
        bracket_gtd = [gtd, gtd_new]

    insuf_progress = False
    low_pos, high_pos = (
        (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)
    )
    while not done and ls_iter < max_ls:
        if abs(bracket[1] - bracket[0]) * d_norm < tolerance_change:
            break

        t = _cubic_interpolate(
            bracket[0],
            bracket_f[0],
            bracket_gtd[0],
            bracket[1],
            bracket_f[1],
            bracket_gtd[1],
        )

        eps = 0.1 * (max(bracket) - min(bracket))
        if min(max(bracket) - t, t - min(bracket)) < eps:
            if insuf_progress or t >= max(bracket) or t <= min(bracket):
                if abs(t - max(bracket)) < abs(t - min(bracket)):
                    t = max(bracket) - eps
                else:
                    t = min(bracket) + eps
                insuf_progress = False
            else:
                insuf_progress = True
        else:
            insuf_progress = False

        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = _dot(g_new, d)
        ls_iter += 1

        if f_new > f + c1 * t * gtd or f_new >= bracket_f[low_pos]:
            bracket[high_pos] = t
            bracket_f[high_pos] = f_new
            bracket_g[high_pos] = g_new.clone()
            bracket_gtd[high_pos] = gtd_new
            low_pos, high_pos = (
                (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
            )
        else:
            if abs(gtd_new) <= -c2 * gtd:
                done = True
            elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0.0:
                bracket[high_pos] = bracket[low_pos]
                bracket_f[high_pos] = bracket_f[low_pos]
                bracket_g[high_pos] = bracket_g[low_pos]
                bracket_gtd[high_pos] = bracket_gtd[low_pos]

            bracket[low_pos] = t
            bracket_f[low_pos] = f_new
            bracket_g[low_pos] = g_new.clone()
            bracket_gtd[low_pos] = gtd_new

    t = bracket[low_pos]
    return bracket_f[low_pos], bracket_g[low_pos], t, ls_func_evals


class LBFGS(Optimizer):
    """Limited-memory BFGS optimizer, aligned with ``torch.optim.LBFGS``."""

    def __init__(
        self,
        params,
        lr=1,
        max_iter=20,
        max_eval=None,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=100,
        line_search_fn=None,
    ):
        if isinstance(lr, tp.Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if scalar_value(lr, "learning rate") < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        defaults = dict(
            lr=lr,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
            line_search_fn=line_search_fn,
        )
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError(
                "LBFGS doesn't support per-parameter options (parameter groups)"
            )

        self._params = self.param_groups[0]["params"]
        self._numel_cache = None

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = sum(
                2 * p.numel() if p.is_complex() else p.numel()
                for p in self._params
            )
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = tp.zeros(p.numel(), dtype=p.dtype, device=p.device)
            elif p.grad.is_sparse:
                view = p.grad.to_dense().reshape((-1,))
            else:
                view = p.grad.reshape((-1,))
            if p.is_complex():
                view = tp.view_as_real(view).reshape((-1,))
            views.append(view)
        if not views:
            return tp.zeros((0,), dtype=tp.float32)
        return tp.cat(views, dim=0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            target = tp.view_as_real(p) if p.is_complex() else p
            end = offset + target.numel()
            value = update.slice(0, offset, end, 1).reshape(target.shape)
            target.add_(value, alpha=step_size)
            offset = end
        if offset != self._numel():
            raise AssertionError(f"Expected offset {offset} to equal {self._numel()}")

    def _clone_param(self):
        return [p.clone() for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            p.copy_(pdata)

    def _directional_evaluate(self, closure, x, t, d):
        self._add_grad(t, d)
        loss = _as_loss(closure())
        flat_grad = self._gather_flat_grad()
        self._set_param(x)
        return loss, flat_grad

    def step(self, closure):
        if closure is None:
            raise RuntimeError("LBFGS requires a closure")
        if len(self.param_groups) != 1:
            raise AssertionError(
                f"Expected exactly one param_group, but got {len(self.param_groups)}"
            )

        closure = tp.enable_grad()(closure)
        if len(self._params) == 0:
            return closure()

        with tp.no_grad():
            group = self.param_groups[0]
            lr = group["lr"]
            max_iter = group["max_iter"]
            max_eval = group["max_eval"]
            tolerance_grad = group["tolerance_grad"]
            tolerance_change = group["tolerance_change"]
            line_search_fn = group["line_search_fn"]
            history_size = group["history_size"]

            state = self.state[self._params[0]]
            state.setdefault("func_evals", 0)
            state.setdefault("n_iter", 0)

            orig_loss = closure()
            loss = _as_loss(orig_loss)
            current_evals = 1
            state["func_evals"] += 1

            flat_grad = self._gather_flat_grad()
            opt_cond = max_abs(flat_grad) <= tolerance_grad
            if opt_cond:
                return orig_loss

            d = state.get("d")
            t = state.get("t")
            old_dirs = state.get("old_dirs")
            old_stps = state.get("old_stps")
            ro = state.get("ro")
            h_diag = state.get("H_diag")
            prev_flat_grad = state.get("prev_flat_grad")
            prev_loss = state.get("prev_loss")

            n_iter = 0
            while n_iter < max_iter:
                n_iter += 1
                state["n_iter"] += 1

                if state["n_iter"] == 1:
                    d = flat_grad.neg()
                    old_dirs, old_stps, ro = [], [], []
                    h_diag = 1.0
                else:
                    y = flat_grad - prev_flat_grad
                    s = d * t
                    ys = _dot(y, s)
                    if ys > 1e-10:
                        if len(old_dirs) == history_size:
                            old_dirs.pop(0)
                            old_stps.pop(0)
                            ro.pop(0)
                        old_dirs.append(y)
                        old_stps.append(s)
                        ro.append(1.0 / ys)
                        h_diag = ys / _dot(y, y)

                    num_old = len(old_dirs)
                    al = state.setdefault("al", [None] * history_size)
                    q = flat_grad.neg()
                    for i in range(num_old - 1, -1, -1):
                        al[i] = _dot(old_stps[i], q) * ro[i]
                        q.add_(old_dirs[i], alpha=-al[i])
                    d = q * h_diag
                    r = d
                    for i in range(num_old):
                        be_i = _dot(old_dirs[i], r) * ro[i]
                        r.add_(old_stps[i], alpha=al[i] - be_i)

                if prev_flat_grad is None:
                    prev_flat_grad = flat_grad.clone()
                else:
                    prev_flat_grad.copy_(flat_grad)
                prev_loss = loss

                if state["n_iter"] == 1:
                    grad_sum = float(flat_grad.abs().sum().item())
                    t = min(1.0, 1.0 / grad_sum) * scalar_value(lr, "lr")
                else:
                    t = scalar_value(lr, "lr")

                gtd = _dot(flat_grad, d)
                if gtd > -tolerance_change:
                    break

                ls_func_evals = 0
                if line_search_fn is not None:
                    if line_search_fn != "strong_wolfe":
                        raise RuntimeError("only 'strong_wolfe' is supported")
                    x_init = self._clone_param()

                    def obj_func(x, step, direction):
                        return self._directional_evaluate(
                            closure, x, step, direction
                        )

                    loss, flat_grad, t, ls_func_evals = _strong_wolfe(
                        obj_func,
                        x_init,
                        t,
                        d,
                        loss,
                        flat_grad,
                        gtd,
                        max_ls=max_eval - current_evals,
                    )
                    self._add_grad(t, d)
                    opt_cond = max_abs(flat_grad) <= tolerance_grad
                else:
                    self._add_grad(t, d)
                    if n_iter != max_iter:
                        loss_tensor = closure()
                        loss = _as_loss(loss_tensor)
                        flat_grad = self._gather_flat_grad()
                        opt_cond = max_abs(flat_grad) <= tolerance_grad
                        ls_func_evals = 1

                current_evals += ls_func_evals
                state["func_evals"] += ls_func_evals

                if n_iter == max_iter or current_evals >= max_eval or opt_cond:
                    break
                if max_abs(d * t) <= tolerance_change:
                    break
                if abs(loss - prev_loss) < tolerance_change:
                    break

            state["d"] = d
            state["t"] = t
            state["old_dirs"] = old_dirs
            state["old_stps"] = old_stps
            state["ro"] = ro
            state["H_diag"] = h_diag
            state["prev_flat_grad"] = prev_flat_grad
            state["prev_loss"] = prev_loss

            return orig_loss
