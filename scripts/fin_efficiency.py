"""
Fin Efficiency Calculator
-------------------------

This module provides thermal engineering calculations for simple straight fins.
It computes:

1. Fin efficiency (η)
2. Fin effectiveness (ε)
3. Heat transfer rate (Q)

The script supports:
- Robust input validation (separate geometry check)
- Use of Corrected Fin Length (Lc) for higher accuracy
- Error handling
- Three example materials (Aluminum, Copper, Steel)
- Clean function-based architecture

Author: Akhil Pillai
Repository: thermal-tools-python
"""

import math
from typing import Dict


def validate_geometry(length: float, thickness: float, k: float, h: float) -> None:
    """Validate fin geometry and material properties for physical consistency.

    Args:
        length: Fin length in meters (must be > 0).
        thickness: Fin thickness in meters (must be > 0).
        k: Thermal conductivity in W/m·K (must be > 0).
        h: Convection coefficient in W/m²·K (must be > 0).

    Raises:
        ValueError: If any physical constraint is violated.
        TypeError: If inputs are not numeric.
    """
    inputs = [length, thickness, k, h]
    param_names = ["length", "thickness", "k", "h"]
    for i, val in enumerate(inputs):
        if not isinstance(val, (int, float)):
            raise TypeError(f"{param_names[i]} must be numeric. Got {type(val).__name__}.")

    if length <= 0:
        raise ValueError(f"Fin length must be > 0. Got {length}.")
    if thickness <= 0:
        raise ValueError(f"Fin thickness must be > 0. Got {thickness}.")
    if k <= 0:
        raise ValueError(f"Thermal conductivity must be > 0. Got {k}.")
    if h <= 0:
        raise ValueError(f"Convection coefficient must be > 0. Got {h}.")


def validate_input(T_base: float, T_inf: float) -> None:
    """Validate temperature inputs."""
    if not isinstance(T_base, (int, float)) or not isinstance(T_inf, (int, float)):
        raise TypeError("T_base and T_inf must be numeric.")
    if T_base <= T_inf:
        raise ValueError(f"Base temperature ({T_base}°C) must exceed ambient ({T_inf}°C).")


def calculate_m_parameter(thickness: float, k: float, h: float, width: float = 1.0) -> float:
    """Calculate the fin parameter m using the general formula.

    For a straight rectangular fin:
        m = sqrt(h * P / (k * Ac))
    where P is the perimeter and Ac is the cross-sectional area.

    Args:
        thickness: Fin thickness in meters.
        k: Thermal conductivity in W/m·K.
        h: Convection coefficient in W/m²·K.
        width: Fin width in meters (default=1).

    Returns:
        Fin parameter m in 1/m.
    """
    perimeter = 2.0 * (thickness + width)
    cross_sectional_area = thickness * width
    return math.sqrt(h * perimeter / (k * cross_sectional_area))


def fin_efficiency(length: float, thickness: float, k: float, h: float) -> float:
    """
    Calculate fin efficiency for a straight rectangular fin using the
    corrected length (Lc) to account for tip heat loss.

    Uses the relation:
        η = tanh(m Lc) / (m Lc)
    where Lc = L + t/2.

    Args:
        length: Fin length (L) in meters.
        thickness: Fin thickness (t) in meters.
        k: Thermal conductivity in W/m·K.
        h: Convection coefficient in W/m²·K.

    Returns:
        Fin efficiency (0 < η ≤ 1).
    """
    validate_geometry(length, thickness, k, h)

    m = calculate_m_parameter(thickness, k, h)
    Lc = length + (thickness / 2.0)
    mLc = m * Lc

    if abs(mLc) < 1e-6:
        return 1.0

    eta = math.tanh(mLc) / mLc
    return max(0.0, min(1.0, eta))


def fin_effectiveness(length: float, thickness: float, k: float, h: float, width: float = 1.0) -> float:
    """
    Calculate fin effectiveness.

    Uses:
        ε = Q_fin / Q_base = η * (A_fin_corrected / A_base)

    Args:
        length: Fin length in meters.
        thickness: Fin thickness in meters.
        k: Thermal conductivity in W/m·K.
        h: Convection coefficient in W/m²·K.
        width: Fin width in meters (default=1).

    Returns:
        Fin effectiveness (ε > 0).
    """
    eta = fin_efficiency(length, thickness, k, h)
    perimeter = 2.0 * (thickness + width)
    Lc = length + (thickness / 2.0)
    area_fin_corrected = perimeter * Lc
    area_base = thickness * width
    area_ratio = area_fin_corrected / area_base
    effectiveness = eta * area_ratio
    return max(0.0, effectiveness)


def heat_transfer_rate(length: float, thickness: float, k: float, h: float, T_base: float, T_inf: float, width: float = 1.0) -> float:
    """
    Compute fin heat transfer rate.

    Uses:
        Q = η * h * A_fin_corrected * (T_base - T_inf)

    Args:
        length: Fin length in meters.
        thickness: Fin thickness in meters.
        k: Thermal conductivity in W/m·K.
        h: Convection coefficient in W/m²·K.
        T_base: Base temperature in °C.
        T_inf: Ambient temperature in °C.
        width: Fin width in meters (default=1 for per unit width).

    Returns:
        Heat transfer rate in Watts.
    """
    validate_input(T_base, T_inf)
    eta = fin_efficiency(length, thickness, k, h)
    delta_T = T_base - T_inf
    Lc = length + (thickness / 2.0)
    perimeter = 2.0 * (thickness + width)
    area_fin_corrected = perimeter * Lc
    Q = eta * h * area_fin_corrected * delta_T
    return max(0.0, Q)


def calculate_all_metrics(length: float, thickness: float, k: float, h: float, T_base: float, T_inf: float, width: float = 1.0) -> Dict[str, float]:
    """
    Calculate all fin performance metrics in one call.
    This function is the primary entry point and handles all validation.
    """
    try:
        validate_geometry(length, thickness, k, h)
        validate_input(T_base, T_inf)

        m = calculate_m_parameter(thickness, k, h, width)
        eta = fin_efficiency(length, thickness, k, h)
        epsilon = fin_effectiveness(length, thickness, k, h, width)
        Q = heat_transfer_rate(length, thickness, k, h, T_base, T_inf, width)

        Lc = length + (thickness / 2.0)
        mLc = m * Lc

        return {
            "efficiency": round(eta, 4),
            "effectiveness": round(epsilon, 3),
            "heat_transfer_W": round(Q, 3),
            "m_parameter_1/m": round(m, 3),
            "mLc": round(mLc, 3),
        }

    except (ValueError, TypeError) as e:
        raise RuntimeError(f"Calculation failed: {str(e)}")


def example_cases() -> Dict[str, Dict[str, float]]:
    """Run 3 standard example cases for common fin materials."""
    length = 0.05      # 5 cm
    thickness = 0.003  # 3 mm
    h = 30             # W/m²·K
    T_base = 100       # °C
    T_inf = 25         # °C
    width = 1.0

    materials = {
        "Aluminum (k=205 W/m·K)": {"k": 205, "description": "Excellent for heat dissipation"},
        "Copper (k=391 W/m·K)": {"k": 391, "description": "Best performance, more expensive"},
        "Steel (k=45 W/m·K)": {"k": 45, "description": "Good for structural fins"},
    }

    results = {}

    print("\n" + "=" * 60)
    print("EXAMPLE CASES - Fin Performance Comparison")
    print("=" * 60)
    print("Common parameters:")
    print(f"  Length (L): {length*1000:.1f} mm, Thickness (t): {thickness*1000:.1f} mm")
    print(f"  h: {h} W/m²·K, ΔT: {T_base-T_inf}°C, Width: {width} m")
    print("=" * 60)

    for name, props in materials.items():
        try:
            metrics = calculate_all_metrics(length, thickness, props["k"], h, T_base, T_inf, width)
            results[name] = metrics

            print(f"\n{name}:")
            print(f"  Description: {props['description']}")
            print(f"  Efficiency (η): {metrics['efficiency']:.3f} ({metrics['efficiency']*100:.1f}%)")
            print(f"  Effectiveness (ε): {metrics['effectiveness']:.2f}")
            print(f"  Heat Transfer: {metrics['heat_transfer_W']:.2f} W per unit width")
            print(f"  m parameter: {metrics['m_parameter_1/m']:.3f} 1/m")
            print(f"  Dimensionless mLc: {metrics['mLc']:.3f}")

        except RuntimeError as e:
            print(f"\nError calculating {name}: {e}")
            results[name] = {"error": str(e)}

    print("\n" + "=" * 60)
    print("SUMMARY: Higher conductivity → Higher efficiency and Q")
    print("=" * 60)

    return results


def run_interactive() -> None:
    """Run interactive fin calculator with user input."""
    print("\n" + "=" * 60)
    print("FIN EFFICIENCY CALCULATOR - Interactive Mode")
    print("=" * 60)

    try:
        print("\nEnter fin parameters (defaults to 1m width):")
        length = float(input("Fin length, L (m): ").strip())
        thickness = float(input("Fin thickness, t (m): ").strip())
        k = float(input("Thermal conductivity, k (W/m·K): ").strip())
        h = float(input("Convection coefficient, h (W/m²·K): ").strip())
        T_base = float(input("Base temperature, T_base (°C): ").strip())
        T_inf = float(input("Ambient temperature, T_inf (°C): ").strip())

        print("\n" + "-" * 40)
        print("Calculating fin performance...")
        print("-" * 40)

        metrics = calculate_all_metrics(length, thickness, k, h, T_base, T_inf)

        print(f"\nRESULTS:")
        print(f"  Fin efficiency (η): {metrics['efficiency']:.4f} ({metrics['efficiency']*100:.2f}%)")
        print(f"  Fin effectiveness (ε): {metrics['effectiveness']:.3f}")
        print(f"  Heat transfer rate: {metrics['heat_transfer_W']:.3f} W per unit width")
        print(f"  Fin parameter m: {metrics['m_parameter_1/m']:.3f} 1/m")
        print(f"  Dimensionless mLc: {metrics['mLc']:.3f}")

        print(f"\nINTERPRETATION:")
        if metrics["efficiency"] > 0.9:
            print("  ✓ Excellent efficiency - fin is well-sized for its material.")
        elif metrics["efficiency"] > 0.7:
            print("  ✓ Good efficiency - fin is effective.")
        elif metrics["efficiency"] > 0.5:
            print("  ⚠ Moderate efficiency - consider shorter/thicker fin or higher 'k'.")
        else:
            print("  ⚠ Low efficiency - fin may be too long/thin; heat transfer drops quickly along its length.")

        if metrics["effectiveness"] > 2:
            print("  ✓ High effectiveness - fin significantly enhances heat transfer (good design).")
        elif metrics["effectiveness"] > 1:
            print("  ✓ Effective - fin improves heat transfer.")
        else:
            print("  ⚠ Low effectiveness - fin may not be beneficial (ε < 2 is often a guideline to justify fins).")

    except ValueError as e:
        print(f"\n❌ Input Error: Please enter valid numbers. {e}")
    except RuntimeError as e:
        print(f"\n❌ Calculation Error: {e}")
    except KeyboardInterrupt:
        print("\n\nCalculation cancelled by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


def main() -> None:
    """Main function to run fin calculator with examples or interactive mode."""
    print(__doc__)

    while True:
        print("\n" + "=" * 60)
        print("MAIN MENU")
        print("=" * 60)
        print("1. Run example cases (Aluminum, Copper, Steel)")
        print("2. Run interactive calculator")
        print("3. Exit")

        choice = input("\nSelect option (1-3): ").strip()

        if choice == "1":
            example_cases()
        elif choice == "2":
            run_interactive()
        elif choice == "3":
            print("\nExiting Fin Efficiency Calculator. Goodbye!")
            break
        else:
            print("\nInvalid option. Please enter 1, 2, or 3.")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    example_cases()
    main()
