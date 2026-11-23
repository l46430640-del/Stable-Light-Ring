"""Trace reversal of the scalar stress."""
import sympy as sp


def matter_trace_reversal():
    g = sp.diag(-1, 1, 1, 1)
    h = sp.Matrix(4, 4, lambda a, b: sp.Symbol(f'h{min(a,b)}{max(a,b)}'))
    v, vb, p, pb = [sp.Matrix(sp.symbols(f'{name}0:4')) for name in ('v', 'b', 'p', 'c')]
    V, Vs, ds = sp.symbols('V Vs ds')
    dinv = -g*h*g
    norm = (vb.T*g*v)[0]
    dnorm = (pb.T*g*v+vb.T*g*p+vb.T*dinv*v)[0]
    kinetic = (vb*v.T+v*vb.T)/2
    dkinetic = (pb*v.T+vb*p.T+p*vb.T+v*pb.T)/2
    T = kinetic-g*(norm+V)/2
    dT = dkinetic-h*(norm+V)/2-g*(dnorm+Vs*ds)/2
    trace = sp.trace(g*T)
    dtrace = sp.trace(dinv*T+g*dT)
    varied_reversal = dT-h*trace/2-g*dtrace/2
    expected = dkinetic+V*h/2+g*Vs*ds/2
    assert all(sp.expand(value) == 0 for value in varied_reversal-expected)
    return dict(trace_reversed_stress_variation=True,
                metric_source_contains_only_first_scalar_derivatives=True,
                metric_source_has_no_metric_derivatives=True)
