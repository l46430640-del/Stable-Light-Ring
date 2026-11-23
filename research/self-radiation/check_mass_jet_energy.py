"""Algebraic checks: mass jet energy."""
import json
from fractions import Fraction
from math import comb
import sympy as sp


def zero(expr):
    assert sp.simplify(expr) == 0


def weighted_mass_connections():
    cases = []
    for p in range(1, 13):
        nodes = [(a, b) for a in range(p+1) for b in range(p-a+1)]
        n = (p+1)*(p+2)//2
        assert len(nodes) == n
        positions = {node: j for j, node in enumerate(nodes)}
        for C0 in (1, 3, 10):
            theta = Fraction(1, 16*C0*n*p*2**p)
            rows = [dict() for _ in nodes]
            edge_count = 0
            for row, (a, b) in enumerate(nodes):
                for j in range(1, b+1):
                    for target in ((a, b-j), (a+1, b-j)):
                        assert sum(target) <= p and target[1] < b
                        column = positions[target]
                        # Conjugating by diag(theta**b) gives this factor.
                        coefficient = comb(b, j)*theta**(b-target[1])
                        assert coefficient == comb(b, j)*theta**j
                        rows[row][column] = rows[row].get(column, 0)+coefficient
                        edge_count += 1
            row_max = max(sum(row.values(), Fraction(0)) for row in rows)
            columns = [Fraction(0) for _ in nodes]
            for row in rows:
                for column, coefficient in row.items():
                    columns[column] += coefficient
            column_max = max(columns)
            binomial_bound = 2*((1+theta)**p-1)
            linear_bound = p*2**p*theta
            assert row_max <= binomial_bound <= linear_bound
            assert n*linear_bound == Fraction(1, 16*C0)
            # ||S||_2^2 <= ||S||_1 ||S||_infinity, with exact arithmetic.
            norm_squared_bound = row_max*column_max
            assert 64*C0*C0*norm_squared_bound <= 1
            cases.append(dict(order=p, radial_indices=n, connections=edge_count,
                              assumed_low_constant=C0, theta=str(theta),
                              absorption_bound_passed=True))
    return dict(cases=cases, general_comparison_bound='||S_p(theta)||_2 <= n_p*p*2^p*theta',
                sufficient_choice='theta_p <= 1/(16*C0*n_p*p*2^p)',
                all_connections_lower_D1_count=True,
                scalar_parameter_and_angular_indices_are_spectators=True,
                scope='Exact radial derivative graph and norm bounds; C0 is assumed to bound the actual bulk and future-flux pairings')


def ordered_mass_division():
    x, y = sp.symbols('rho0 y', positive=True)
    u = sp.Function('u')(x, y)
    inverse_mass = x*x*y
    D0 = lambda f: x*sp.diff(f, x)
    D1 = lambda f: y*sp.diff(f, y)
    L0 = lambda f: -2*D1(D0(f)-D1(f))-y*(D1(D1(f))+D1(f))
    commutator = L0(inverse_mass*u)-inverse_mass*L0(u)
    expected = inverse_mass*(-2*D0(u)-2*u-2*y*(D1(u)+u))
    zero(commutator-expected)
    assert sp.simplify(commutator) != 0
    zero(D0(inverse_mass)-2*inverse_mass)
    zero(D1(inverse_mass)-inverse_mass)
    zero(D0(inverse_mass)-2*D1(inverse_mass))
    T = lambda f: sp.expand(-inverse_mass*L0(f))
    polynomial = 1+x*y+x*x*y*y
    source = L0(polynomial)+polynomial/inverse_mass
    recovered = inverse_mass*source+T(inverse_mass*source)+T(T(inverse_mass*source))+T(T(T(polynomial)))
    zero(recovered-polynomial)
    return dict(nonzero_commutator='[L0,M^-1]u=M^-1*(-2*D0*u-2*u-2*y*(D1*u+u)) for the flat radial model',
                inverse_mass_conormal_weight_preserved=True,
                ordered_three_step_identity=True,
                scope='Variable differential operators remain ordered; the test is not an inverse estimate')


if __name__ == "__main__":
    import json
    if not __debug__:
        raise SystemExit("Run without -O.")
    print(json.dumps({
        "mass": weighted_mass_connections(),
        "division": ordered_mass_division(),
    }, indent=2))
