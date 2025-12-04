"""
Enhanced Heat Exchanger Analysis Toolkit
----------------------------------------

A comprehensive toolkit for heat exchanger analysis, design, and comparison.

Features:
- LMTD and effectiveness-NTU methods for multiple configurations
- Enhanced crossflow models with mixing conditions
- Temperature-dependent fluid properties
- Pressure drop calculations with multiple friction models
- Geometry-specific models (plate, shell-and-tube, circular tube)
- Fouling impact analysis
- Design optimization and parameter studies
- Advanced visualization and comparison tools
- Batch processing capability

Units: SI (kg, m, s, K, W, Pa)
Temperatures in °C for input/output, K for absolute calculations

Author: Akhil Pillai
Repo: thermal-tools-python
Version: 2.0
"""
import math
import warnings
from typing import Tuple, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

# Optional dependencies with graceful fallback
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    warnings.warn("NumPy not found. Some advanced features limited.")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    warnings.warn("Pandas not found. DataFrames unavailable.")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("Matplotlib not found. Plotting unavailable.")

try:
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("SciPy not found. Optimization features unavailable.")


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class FlowConfig(str, Enum):
    """Heat exchanger flow configuration types."""
    PARALLEL = "parallel"
    COUNTER = "counter"
    CROSSFLOW = "crossflow"
    SHELL_TUBE = "shell_tube"
    PLATE = "plate"


class MixingCondition(str, Enum):
    """Mixing conditions for crossflow heat exchangers."""
    BOTH_UNMIXED = "both_unmixed"
    BOTH_MIXED = "both_mixed"
    MIN_MIXED_MAX_UNMIXED = "min_mixed_max_unmixed"
    MIN_UNMIXED_MAX_MIXED = "min_unmixed_max_mixed"


class FluidType(str, Enum):
    """Predefined fluid types with property correlations."""
    WATER = "water"
    AIR = "air"
    ETHYLENE_GLYCOL_50 = "ethylene_glycol_50"
    ENGINE_OIL = "engine_oil"
    REFRIGERANT_R134A = "refrigerant_r134a"


class FrictionModel(str, Enum):
    """Friction factor models."""
    DARCY_WEISBACH = "darcy_weisbach"
    BLASIUS = "blasius"
    CHURCHILL = "churchill"
    HAALAND = "haaland"
    MOODY = "moody"


# ============================================================================
# FLUID PROPERTIES
# ============================================================================

@dataclass
class FluidProperties:
    """Container for fluid thermophysical properties."""
    name: str
    density: float  # kg/m³
    viscosity: float  # Pa·s
    specific_heat: float  # J/(kg·K)
    thermal_conductivity: float  # W/(m·K)
    prandtl: float  # Prandtl number
    
    def reynolds(self, velocity: float, diameter: float) -> float:
        """Calculate Reynolds number."""
        return self.density * velocity * diameter / self.viscosity


class FluidPropertyLibrary:
    """Library of temperature-dependent fluid properties."""
    
    @staticmethod
    @lru_cache(maxsize=128)
    def water_properties(T_celsius: float = 25.0, P_kpa: float = 101.325) -> FluidProperties:
        """
        Water properties using IAPWS correlations (simplified).
        
        Args:
            T_celsius: Temperature in °C
            P_kpa: Pressure in kPa
        
        Returns:
            FluidProperties object
        """
        T_k = T_celsius + 273.15
        
        # Density (kg/m³) - Simplified correlation for liquid water
        rho = (999.8396 + 16.945176*T_celsius - 7.9870401e-3*T_celsius**2 
               - 46.170461e-6*T_celsius**3 + 105.56302e-9*T_celsius**4 
               - 280.54253e-12*T_celsius**5) / (1 + 16.879850e-3*T_celsius)
        
        # Viscosity (Pa·s) - Simplified correlation
        mu = 2.414e-5 * 10**(247.8/(T_k - 140))
        
        # Specific heat (J/(kg·K)) - Polynomial correlation
        cp = (4217.4 - 3.720283*T_celsius + 0.1412855*T_celsius**2 
              - 2.654387e-3*T_celsius**3 + 2.093236e-5*T_celsius**4)
        
        # Thermal conductivity (W/(m·K))
        k = 0.6065 * (-1.48445 + 4.12292*(T_k/298.15) - 1.63866*(T_k/298.15)**2)
        
        # Prandtl number
        pr = cp * mu / k if k > 0 else 7.0
        
        return FluidProperties(
            name="water",
            density=rho,
            viscosity=mu,
            specific_heat=cp,
            thermal_conductivity=k,
            prandtl=pr
        )
    
    @staticmethod
    @lru_cache(maxsize=128)
    def air_properties(T_celsius: float = 25.0, P_kpa: float = 101.325) -> FluidProperties:
        """
        Air properties at given temperature and pressure.
        
        Args:
            T_celsius: Temperature in °C
            P_kpa: Pressure in kPa
        
        Returns:
            FluidProperties object
        """
        T_k = T_celsius + 273.15
        
        # Ideal gas law for density
        R = 287.058  # J/(kg·K)
        rho = (P_kpa * 1000) / (R * T_k)
        
        # Sutherland's law for viscosity
        mu0 = 1.716e-5  # Pa·s at 273.15K
        T0 = 273.15  # K
        S = 110.4  # Sutherland constant for air
        mu = mu0 * (T0 + S)/(T_k + S) * (T_k/T0)**1.5
        
        # Specific heat at constant pressure
        cp = 1005 + 0.05*T_celsius
        
        # Thermal conductivity
        k = 0.0241 * (T_k/273.15)**0.9
        
        # Prandtl number
        pr = cp * mu / k
        
        return FluidProperties(
            name="air",
            density=rho,
            viscosity=mu,
            specific_heat=cp,
            thermal_conductivity=k,
            prandtl=pr
        )
    
    @staticmethod
    def get_fluid(fluid_type: FluidType, T_celsius: float = 25.0, 
                  P_kpa: float = 101.325) -> FluidProperties:
        """
        Get properties for a specific fluid type.
        
        Args:
            fluid_type: FluidType enum
            T_celsius: Temperature in °C
            P_kpa: Pressure in kPa
        
        Returns:
            FluidProperties object
        """
        if fluid_type == FluidType.WATER:
            return FluidPropertyLibrary.water_properties(T_celsius, P_kpa)
        elif fluid_type == FluidType.AIR:
            return FluidPropertyLibrary.air_properties(T_celsius, P_kpa)
        elif fluid_type == FluidType.ETHYLENE_GLYCOL_50:
            # Simplified properties for 50% ethylene glycol/water mixture
            rho = 1080 - 0.6*T_celsius
            mu = 0.01 * math.exp(-0.02*T_celsius)  # Rough approximation
            cp = 3400
            k = 0.4
            pr = cp * mu / k
            return FluidProperties("ethylene_glycol_50", rho, mu, cp, k, pr)
        else:
            # Default to water if fluid type not implemented
            warnings.warn(f"Fluid type {fluid_type} not fully implemented. Using water properties.")
            return FluidPropertyLibrary.water_properties(T_celsius, P_kpa)


# ============================================================================
# CORE HEAT EXCHANGER FUNCTIONS
# ============================================================================

def lmtd(T_hot_in: float, T_hot_out: float, 
         T_cold_in: float, T_cold_out: float, 
         config: FlowConfig = FlowConfig.COUNTER) -> float:
    """
    Compute Log Mean Temperature Difference (LMTD) with correction factors.
    
    Args:
        T_hot_in, T_hot_out: Hot stream temperatures (°C)
        T_cold_in, T_cold_out: Cold stream temperatures (°C)
        config: Flow configuration
    
    Returns:
        Corrected LMTD (K)
    """
    # Temperature differences for counterflow arrangement
    dT1 = T_hot_in - T_cold_out
    dT2 = T_hot_out - T_cold_in
    
    # Handle parallel flow
    if config == FlowConfig.PARALLEL:
        dT1 = T_hot_in - T_cold_in
        dT2 = T_hot_out - T_cold_out
    
    # Basic LMTD calculation
    if dT1 <= 0 or dT2 <= 0:
        warnings.warn("Temperature crossing or approach. LMTD may not be valid.")
        return 0.0
    
    if abs(dT1 - dT2) < 1e-9:
        lmtd_value = dT1
    else:
        lmtd_value = (dT1 - dT2) / math.log(dT1 / dT2)
    
    # Apply correction factor for non-counterflow configurations
    if config in [FlowConfig.CROSSFLOW, FlowConfig.SHELL_TUBE, FlowConfig.PLATE]:
        F = lmtd_correction_factor(config, dT1, dT2, T_hot_in, T_hot_out, 
                                   T_cold_in, T_cold_out)
        lmtd_value *= F
    
    return max(0.0, lmtd_value)


def lmtd_correction_factor(config: FlowConfig, dT1: float, dT2: float,
                          T_hot_in: float, T_hot_out: float,
                          T_cold_in: float, T_cold_out: float,
                          geometry_params: Optional[Dict] = None) -> float:
    """
    Calculate LMTD correction factor for various heat exchanger geometries.
    
    Args:
        config: Flow configuration
        dT1, dT2: Temperature differences
        geometry_params: Geometry-specific parameters
    
    Returns:
        Correction factor F (0 < F ≤ 1)
    """
    if config == FlowConfig.PARALLEL or config == FlowConfig.COUNTER:
        return 1.0
    
    # Calculate P and R parameters
    P = (T_cold_out - T_cold_in) / (T_hot_in - T_cold_in)
    R = (T_hot_in - T_hot_out) / (T_cold_out - T_cold_in)
    
    if config == FlowConfig.CROSSFLOW:
        # Bowman's approximation for crossflow with both fluids unmixed
        if geometry_params and geometry_params.get('mixing') == MixingCondition.BOTH_MIXED:
            # Both fluids mixed
            if R != 1:
                F = (math.sqrt(R**2 + 1) / (R - 1) * 
                     math.log((1 - P) / (1 - P*R))) / math.log((2/P - 1 - R + math.sqrt(R**2 + 1)) /
                                                               (2/P - 1 - R - math.sqrt(R**2 + 1)))
            else:
                F = (math.sqrt(2) * P / (1 - P)) / math.log((2/P - 2 + math.sqrt(2)) /
                                                           (2/P - 2 - math.sqrt(2)))
        else:
            # One or both fluids unmixed (common approximation)
            F = 1.0  # Simplified - in practice, use charts or more detailed correlations
    
    elif config == FlowConfig.SHELL_TUBE:
        # For 1-shell pass, 2-tube passes
        if geometry_params and geometry_params.get('n_shell_passes') == 1:
            if R != 1:
                sqrt_term = math.sqrt(R**2 + 1)
                F = sqrt_term * math.log((1 - P) / (1 - P*R)) / (
                    (R - 1) * math.log((2/P - 1 - R + sqrt_term) / (2/P - 1 - R - sqrt_term)))
            else:
                F = P * math.sqrt(2) / ((1 - P) * math.log((2/P - 2 + math.sqrt(2)) / 
                                                         (2/P - 2 - math.sqrt(2))))
    
    elif config == FlowConfig.PLATE:
        # Plate heat exchangers typically have high F values
        F = 0.95  # Conservative estimate
    
    else:
        F = 1.0
    
    return min(1.0, max(0.7, F))  # Clamp to reasonable range


def effectiveness_ntu(NTU: float, Cr: float, 
                     config: FlowConfig = FlowConfig.COUNTER,
                     mixing: MixingCondition = MixingCondition.BOTH_UNMIXED) -> float:
    """
    Calculate effectiveness for given NTU and capacity ratio.
    
    Args:
        NTU: Number of Transfer Units
        Cr: Capacity ratio (C_min/C_max)
        config: Flow configuration
        mixing: Mixing condition for crossflow
    
    Returns:
        Effectiveness ε (0-1)
    """
    if NTU <= 0:
        return 0.0
    
    if config == FlowConfig.PARALLEL:
        # Parallel flow
        return (1.0 - math.exp(-NTU * (1.0 + Cr))) / (1.0 + Cr)
    
    elif config == FlowConfig.COUNTER:
        # Counterflow
        if abs(Cr - 1.0) < 1e-9:
            return NTU / (1.0 + NTU)
        num = 1.0 - math.exp(-NTU * (1.0 - Cr))
        den = 1.0 - Cr * math.exp(-NTU * (1.0 - Cr))
        return num / den
    
    elif config == FlowConfig.CROSSFLOW:
        # Crossflow with different mixing conditions
        return effectiveness_ntu_crossflow(NTU, Cr, mixing)
    
    elif config == FlowConfig.SHELL_TUBE:
        # 1-shell pass, 2-tube passes
        if abs(Cr - 1.0) < 1e-9:
            NTU1 = NTU / 2
            eps1 = 2 / (1 + Cr + math.sqrt(1 + Cr**2) * 
                       (1 + math.exp(-NTU1 * math.sqrt(1 + Cr**2))) / 
                       (1 - math.exp(-NTU1 * math.sqrt(1 + Cr**2))))
            return ((1 - eps1 * Cr) / (1 - eps1))**2 - 1 / (
                ((1 - eps1 * Cr) / (1 - eps1))**2 - Cr)
        else:
            sqrt_term = math.sqrt(1 + Cr**2)
            return 2 / (1 + Cr + sqrt_term * 
                       (1 + math.exp(-NTU * sqrt_term)) / 
                       (1 - math.exp(-NTU * sqrt_term)))
    
    else:
        raise ValueError(f"Unsupported configuration: {config}")


def effectiveness_ntu_crossflow(NTU: float, Cr: float, 
                               mixing: MixingCondition = MixingCondition.BOTH_UNMIXED) -> float:
    """
    Crossflow effectiveness for different mixing conditions.
    
    Args:
        NTU: Number of Transfer Units
        Cr: Capacity ratio (C_min/C_max)
        mixing: Mixing condition
    
    Returns:
        Effectiveness ε
    """
    if mixing == MixingCondition.BOTH_UNMIXED:
        # Common approximation for both fluids unmixed
        try:
            inner = 1.0 - math.exp(-Cr * (NTU ** 0.78))
            return 1.0 - math.exp(- (NTU ** 0.22) * (inner / Cr))
        except:
            return 0.0
    
    elif mixing == MixingCondition.BOTH_MIXED:
        # Both fluids mixed
        try:
            term1 = 1.0 / (1.0 - math.exp(-NTU))
            term2 = Cr / (1.0 - math.exp(-Cr * NTU))
            term3 = 1.0 / NTU
            return 1.0 / (term1 + term2 - term3)
        except:
            return 0.0
    
    elif mixing == MixingCondition.MIN_MIXED_MAX_UNMIXED:
        # C_min mixed, C_max unmixed
        try:
            return (1.0 / Cr) * (1.0 - math.exp(-Cr * (1.0 - math.exp(-NTU))))
        except:
            return 0.0
    
    elif mixing == MixingCondition.MIN_UNMIXED_MAX_MIXED:
        # C_min unmixed, C_max mixed
        try:
            return 1.0 - math.exp(-(1.0 / Cr) * (1.0 - math.exp(-Cr * NTU)))
        except:
            return 0.0
    
    else:
        warnings.warn(f"Unknown mixing condition: {mixing}. Using both unmixed.")
        return effectiveness_ntu_crossflow(NTU, Cr, MixingCondition.BOTH_UNMIXED)


# ============================================================================
# HEAT EXCHANGER SOLVER
# ============================================================================

def solve_heat_exchanger(
    m_dot_hot: float,
    m_dot_cold: float,
    T_hot_in: float,
    T_cold_in: float,
    cp_hot: Union[float, Callable[[float], float]],
    cp_cold: Union[float, Callable[[float], float]],
    U: float,
    A: float,
    config: FlowConfig = FlowConfig.COUNTER,
    mixing: MixingCondition = MixingCondition.BOTH_UNMIXED,
    fluid_hot: Optional[FluidProperties] = None,
    fluid_cold: Optional[FluidProperties] = None,
    geometry_params: Optional[Dict] = None
) -> Dict[str, float]:
    """
    Solve heat exchanger using effectiveness-NTU method.
    
    Args:
        m_dot_hot, m_dot_cold: Mass flow rates (kg/s)
        T_hot_in, T_cold_in: Inlet temperatures (°C)
        cp_hot, cp_cold: Specific heats (J/kgK) or callable functions
        U: Overall heat transfer coefficient (W/m²K)
        A: Heat transfer area (m²)
        config: Flow configuration
        mixing: Mixing condition for crossflow
        fluid_hot, fluid_cold: Fluid properties (optional)
        geometry_params: Geometry-specific parameters
    
    Returns:
        Dictionary with all calculated parameters
    """
    # Handle temperature-dependent specific heats
    if callable(cp_hot):
        cp_hot_val = cp_hot(T_hot_in)
    else:
        cp_hot_val = float(cp_hot)
    
    if callable(cp_cold):
        cp_cold_val = cp_cold(T_cold_in)
    else:
        cp_cold_val = float(cp_cold)
    
    # Heat capacity rates
    C_hot = m_dot_hot * cp_hot_val
    C_cold = m_dot_cold * cp_cold_val
    C_min = min(C_hot, C_cold)
    C_max = max(C_hot, C_cold)
    Cr = C_min / C_max
    
    # NTU
    NTU = U * A / C_min if C_min > 0 else 0.0
    
    # Effectiveness
    eff = effectiveness_ntu(NTU, Cr, config, mixing)
    
    # Heat transfer rate
    Q_max = C_min * (T_hot_in - T_cold_in)
    Q = eff * Q_max
    
    # Outlet temperatures
    T_hot_out = T_hot_in - Q / C_hot if C_hot > 0 else T_hot_in
    T_cold_out = T_cold_in + Q / C_cold if C_cold > 0 else T_cold_in
    
    # LMTD
    lmtd_val = lmtd(T_hot_in, T_hot_out, T_cold_in, T_cold_out, config)
    
    # Check U calculation consistency
    U_calc = Q / (A * lmtd_val) if A * lmtd_val > 0 else U
    
    return {
        "T_hot_out": T_hot_out,
        "T_cold_out": T_cold_out,
        "Q_W": Q,
        "NTU": NTU,
        "Cr": Cr,
        "effectiveness": eff,
        "C_min": C_min,
        "C_max": C_max,
        "lmtd_K": lmtd_val,
        "U_input": U,
        "U_calculated": U_calc,
        "config": config.value,
        "mixing": mixing.value if config == FlowConfig.CROSSFLOW else "N/A"
    }


# ============================================================================
# PRESSURE DROP CALCULATIONS
# ============================================================================

def pressure_drop(
    m_dot: float,
    fluid: FluidProperties,
    geometry: Dict[str, float],
    model: FrictionModel = FrictionModel.DARCY_WEISBACH,
    roughness: float = 0.0,
    n_parallel: int = 1
) -> Dict[str, float]:
    """
    Calculate pressure drop for various geometries and friction models.
    
    Args:
        m_dot: Total mass flow rate (kg/s)
        fluid: Fluid properties
        geometry: Geometry parameters (depends on geometry type)
        model: Friction model to use
        roughness: Surface roughness (m)
        n_parallel: Number of parallel channels
    
    Returns:
        Dictionary with pressure drop and related parameters
    """
    # Flow per channel
    m_dot_ch = m_dot / n_parallel
    
    # Geometry-specific calculations
    geometry_type = geometry.get('type', 'circular')
    
    if geometry_type == 'circular':
        D = geometry['diameter']
        L = geometry['length']
        A_flow = math.pi * D**2 / 4
        hydraulic_diameter = D
    elif geometry_type == 'rectangular':
        W = geometry['width']
        H = geometry['height']
        L = geometry['length']
        A_flow = W * H
        hydraulic_diameter = 2 * W * H / (W + H)
    elif geometry_type == 'annular':
        D_outer = geometry['diameter_outer']
        D_inner = geometry['diameter_inner']
        L = geometry['length']
        A_flow = math.pi * (D_outer**2 - D_inner**2) / 4
        hydraulic_diameter = D_outer - D_inner
    else:
        raise ValueError(f"Unsupported geometry type: {geometry_type}")
    
    # Velocity
    V = m_dot_ch / (fluid.density * A_flow)
    
    # Reynolds number
    Re = fluid.density * V * hydraulic_diameter / fluid.viscosity
    
    # Friction factor
    f = friction_factor(Re, roughness/hydraulic_diameter, model)
    
    # Minor losses (if specified)
    K_minor = geometry.get('K_minor', 0.0)
    
    # Pressure drop
    deltaP_major = f * (L / hydraulic_diameter) * 0.5 * fluid.density * V**2
    deltaP_minor = K_minor * 0.5 * fluid.density * V**2
    deltaP_total = deltaP_major + deltaP_minor
    
    return {
        "Re": Re,
        "velocity_m_s": V,
        "friction_factor": f,
        "deltaP_major_Pa": deltaP_major,
        "deltaP_minor_Pa": deltaP_minor,
        "deltaP_total_Pa": deltaP_total,
        "hydraulic_diameter": hydraulic_diameter
    }


def friction_factor(Re: float, relative_roughness: float = 0.0,
                   model: FrictionModel = FrictionModel.DARCY_WEISBACH) -> float:
    """
    Calculate friction factor using specified model.
    
    Args:
        Re: Reynolds number
        relative_roughness: ε/D
        model: Friction model
    
    Returns:
        Darcy friction factor f
    """
    if Re <= 0:
        return 0.0
    
    if Re < 2300:
        # Laminar flow
        return 64.0 / Re
    
    elif Re < 4000:
        # Transitional flow - interpolate
        f_lam = 64.0 / 2300.0
        f_turb = friction_factor(4000, relative_roughness, model)
        frac = (Re - 2300.0) / (4000.0 - 2300.0)
        return f_lam + frac * (f_turb - f_lam)
    
    else:
        # Turbulent flow
        if model == FrictionModel.BLASIUS:
            return 0.3164 / (Re ** 0.25)
        
        elif model == FrictionModel.HAALAND:
            term = (relative_roughness/3.7)**1.11 + 6.9/Re
            return (-1.8 * math.log10(term))**(-2)
        
        elif model == FrictionModel.CHURCHILL:
            # Churchill's all-regime correlation
            A = (2.457 * math.log(1/((7/Re)**0.9 + 0.27*relative_roughness)))**16
            B = (37530/Re)**16
            return 8 * ((8/Re)**12 + 1/(A + B)**1.5)**(1/12)
        
        elif model == FrictionModel.MOODY:
            # Moody chart approximation
            f_guess = 0.02
            for _ in range(10):
                f_guess = 0.0055 * (1 + (2e4*relative_roughness + 1e6/Re)**(1/3))
            return f_guess
        
        else:  # DARCY_WEISBACH
            # Colebrook-White equation (simplified)
            if relative_roughness < 1e-6:
                return 0.3164 / (Re ** 0.25)
            else:
                # Iterative solution
                f = 0.02
                for _ in range(20):
                    rhs = -2.0 * math.log10(relative_roughness/3.7 + 2.51/(Re*math.sqrt(f)))
                    f_new = 1.0 / (rhs**2)
                    if abs(f_new - f) < 1e-8:
                        break
                    f = f_new
                return f


# ============================================================================
# GEOMETRY-SPECIFIC MODELS
# ============================================================================

class HeatExchangerGeometry:
    """Base class for heat exchanger geometries."""
    
    def __init__(self, geometry_type: str, parameters: Dict):
        self.geometry_type = geometry_type
        self.parameters = parameters
    
    def heat_transfer_area(self) -> float:
        """Calculate heat transfer area."""
        raise NotImplementedError
    
    def flow_area(self) -> float:
        """Calculate flow area."""
        raise NotImplementedError
    
    def hydraulic_diameter(self) -> float:
        """Calculate hydraulic diameter."""
        raise NotImplementedError
    
    def heat_transfer_coefficient(self, Re: float, Pr: float,
                                  fluid: FluidProperties) -> float:
        """Calculate heat transfer coefficient."""
        raise NotImplementedError


class PlateHeatExchanger(HeatExchangerGeometry):
    """Plate heat exchanger geometry."""
    
    def __init__(self, plate_width: float, plate_height: float, 
                 plate_spacing: float, n_plates: int,
                 chevron_angle: float = 60.0, plate_thickness: float = 0.001):
        params = {
            'type': 'plate',
            'plate_width': plate_width,
            'plate_height': plate_height,
            'plate_spacing': plate_spacing,
            'n_plates': n_plates,
            'chevron_angle': chevron_angle,
            'plate_thickness': plate_thickness
        }
        super().__init__('plate', params)
    
    def heat_transfer_area(self) -> float:
        """Total heat transfer area."""
        w = self.parameters['plate_width']
        h = self.parameters['plate_height']
        n = self.parameters['n_plates']
        # Two sides per plate, minus two end plates
        return 2 * w * h * (n - 2)
    
    def flow_area(self) -> float:
        """Flow area per channel."""
        w = self.parameters['plate_width']
        s = self.parameters['plate_spacing']
        return w * s
    
    def hydraulic_diameter(self) -> float:
        """Hydraulic diameter for plate heat exchanger."""
        s = self.parameters['plate_spacing']
        return 2 * s
    
    def heat_transfer_coefficient(self, Re: float, Pr: float,
                                  fluid: FluidProperties) -> float:
        """Calculate Nusselt number using Martin correlation."""
        beta = math.radians(self.parameters['chevron_angle'])
        
        # Martin's correlation for plate heat exchangers
        if Re < 10:
            Nu = 0.205 * Re**0.667 * Pr**0.4
        elif Re < 100:
            Nu = 0.383 * Re**0.667 * Pr**0.4
        else:
            phi = (beta / math.pi) * 180
            f = (0.2668 - 0.006967*phi + 0.00001685*phi**2) * Re**(
                0.728 + 0.0543*math.sin(math.radians(phi+3.6)))
            Nu = f * Pr**0.4
        
        return Nu * fluid.thermal_conductivity / self.hydraulic_diameter()


class ShellAndTubeHeatExchanger(HeatExchangerGeometry):
    """Shell-and-tube heat exchanger geometry."""
    
    def __init__(self, shell_diameter: float, tube_diameter: float,
                 tube_length: float, n_tubes: int,
                 n_baffles: int = 10, tube_pitch: float = 0.0,
                 layout_angle: float = 30.0):
        params = {
            'type': 'shell_tube',
            'shell_diameter': shell_diameter,
            'tube_diameter': tube_diameter,
            'tube_length': tube_length,
            'n_tubes': n_tubes,
            'n_baffles': n_baffles,
            'tube_pitch': tube_pitch,
            'layout_angle': layout_angle
        }
        super().__init__('shell_tube', params)
    
    def heat_transfer_area(self) -> float:
        """Total heat transfer area (tube side)."""
        D = self.parameters['tube_diameter']
        L = self.parameters['tube_length']
        n = self.parameters['n_tubes']
        return math.pi * D * L * n
    
    def flow_area_shell(self) -> float:
        """Shell-side flow area."""
        D_shell = self.parameters['shell_diameter']
        D_tube = self.parameters['tube_diameter']
        n_tubes = self.parameters['n_tubes']
        pitch = self.parameters['tube_pitch'] or 1.25 * D_tube
        
        # Approximate shell flow area
        return math.pi * D_shell**2 / 4 - n_tubes * math.pi * D_tube**2 / 4
    
    def heat_transfer_coefficient_shell(self, Re: float, Pr: float,
                                        fluid: FluidProperties) -> float:
        """Shell-side heat transfer coefficient (Kern method)."""
        # Kern correlation for shell-side
        Nu = 0.36 * Re**0.55 * Pr**(1/3)
        D_e = self.parameters['shell_diameter'] - self.parameters['tube_diameter']
        return Nu * fluid.thermal_conductivity / D_e


# ============================================================================
# FOULING AND PERFORMANCE DEGRADATION
# ============================================================================

def fouling_impact(
    U_clean: float,
    fouling_factors: Dict[str, float],
    time_years: float = 1.0,
    fouling_rate: float = 0.0001
) -> Dict[str, float]:
    """
    Calculate impact of fouling on heat exchanger performance.
    
    Args:
        U_clean: Clean overall heat transfer coefficient (W/m²K)
        fouling_factors: {'hot_side': Rf_hot, 'cold_side': Rf_cold} in m²K/W
        time_years: Operating time in years
        fouling_rate: Annual fouling factor increase (m²K/W/year)
    
    Returns:
        Dictionary with fouling impact parameters
    """
    # Time-dependent fouling
    Rf_hot = fouling_factors.get('hot_side', 0.0) + fouling_rate * time_years
    Rf_cold = fouling_factors.get('cold_side', 0.0) + fouling_rate * time_years
    Rf_total = Rf_hot + Rf_cold
    
    # Dirty overall coefficient
    U_dirty = 1 / (1/U_clean + Rf_total)
    
    # Effectiveness reduction (approximate)
    eff_reduction = 1 - U_dirty / U_clean
    
    # Additional pressure drop (approximate)
    dP_increase = 1 + 0.1 * time_years  # 10% increase per year
    
    return {
        'U_clean_W_m2K': U_clean,
        'U_dirty_W_m2K': U_dirty,
        'fouling_resistance_m2K_W': Rf_total,
        'effectiveness_reduction': eff_reduction,
        'performance_factor': U_dirty / U_clean,
        'pressure_drop_factor': dP_increase,
        'cleaning_recommended': Rf_total > 0.00035  # Typical threshold
    }


# ============================================================================
# OPTIMIZATION AND DESIGN
# ============================================================================

def optimize_hx_design(
    m_dot_hot: float,
    m_dot_cold: float,
    T_hot_in: float,
    T_cold_in: float,
    cp_hot: float,
    cp_cold: float,
    constraints: Dict[str, float],
    objective: str = "min_area",
    config: FlowConfig = FlowConfig.COUNTER,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, float]:
    """
    Optimize heat exchanger design parameters.
    
    Args:
        m_dot_hot, m_dot_cold: Mass flow rates (kg/s)
        T_hot_in, T_cold_in: Inlet temperatures (°C)
        cp_hot, cp_cold: Specific heats (J/kgK)
        constraints: {'max_dP': 10000, 'min_effectiveness': 0.7, 'max_cost': 5000}
        objective: 'min_area', 'min_cost', 'max_effectiveness', 'min_pressure_drop'
        config: Flow configuration
        bounds: Bounds for optimization variables {'U': (100, 5000), 'A': (0.1, 100)}
    
    Returns:
        Optimal design parameters
    """
    if not HAS_SCIPY:
        warnings.warn("SciPy not available. Using simplified optimization.")
        return _simplified_optimization(
            m_dot_hot, m_dot_cold, T_hot_in, T_cold_in,
            cp_hot, cp_cold, constraints, objective, config
        )
    
    # Default bounds
    if bounds is None:
        bounds = {'U': (100, 5000), 'A': (0.1, 100)}
    
    # Initial guess
    x0 = [(bounds['U'][0] + bounds['U'][1]) / 2,
          (bounds['A'][0] + bounds['A'][1]) / 2]
    
    # Objective function
    def objective_func(x):
        U, A = x
        
        # Solve heat exchanger
        try:
            result = solve_heat_exchanger(
                m_dot_hot, m_dot_cold, T_hot_in, T_cold_in,
                cp_hot, cp_cold, U, A, config
            )
        except:
            return 1e9
        
        # Calculate objective
        if objective == "min_area":
            return A
        elif objective == "min_cost":
            # Simple cost model: cost ~ A^0.6
            return A**0.6 * 500  # $500 per m^0.6
        elif objective == "max_effectiveness":
            return -result['effectiveness']  # Negative for minimization
        elif objective == "min_pressure_drop":
            # Simplified pressure drop estimate
            return A * U  # Proxy for pressure drop
        else:
            return A
    
    # Constraints
    constraints_list = []
    
    if 'min_effectiveness' in constraints:
        def effectiveness_constraint(x):
            U, A = x
            result = solve_heat_exchanger(
                m_dot_hot, m_dot_cold, T_hot_in, T_cold_in,
                cp_hot, cp_cold, U, A, config
            )
            return result['effectiveness'] - constraints['min_effectiveness']
        constraints_list.append({'type': 'ineq', 'fun': effectiveness_constraint})
    
    if 'max_dP' in constraints:
        def pressure_constraint(x):
            U, A = x
            # Simplified pressure drop calculation
            return constraints['max_dP'] - A * U / 1000  # Proxy
        constraints_list.append({'type': 'ineq', 'fun': pressure_constraint})
    
    # Bounds for variables
    bounds_list = [bounds['U'], bounds['A']]
    
    # Run optimization
    result = minimize(
        objective_func,
        x0,
        bounds=bounds_list,
        constraints=constraints_list,
        method='SLSQP',
        options={'maxiter': 100, 'ftol': 1e-6}
    )
    
    if result.success:
        U_opt, A_opt = result.x
        sol = solve_heat_exchanger(
            m_dot_hot, m_dot_cold, T_hot_in, T_cold_in,
            cp_hot, cp_cold, U_opt, A_opt, config
        )
        
        return {
            'U_optimal': U_opt,
            'A_optimal': A_opt,
            'effectiveness': sol['effectiveness'],
            'Q_W': sol['Q_W'],
            'objective_value': result.fun,
            'success': True,
            'message': result.message
        }
    else:
        return {
            'U_optimal': None,
            'A_optimal': None,
            'success': False,
            'message': result.message
        }


def _simplified_optimization(
    m_dot_hot: float,
    m_dot_cold: float,
    T_hot_in: float,
    T_cold_in: float,
    cp_hot: float,
    cp_cold: float,
    constraints: Dict[str, float],
    objective: str,
    config: FlowConfig
) -> Dict[str, float]:
    """Simplified optimization without SciPy."""
    # Simple grid search
    U_vals = [100, 500, 1000, 2000, 5000]
    A_vals = [0.1, 0.5, 1, 5, 10, 50]
    
    best_value = float('inf')
    best_params = {}
    
    for U in U_vals:
        for A in A_vals:
            try:
                sol = solve_heat_exchanger(
                    m_dot_hot, m_dot_cold, T_hot_in, T_cold_in,
                    cp_hot, cp_cold, U, A, config
                )
                
                # Check constraints
                if 'min_effectiveness' in constraints:
                    if sol['effectiveness'] < constraints['min_effectiveness']:
                        continue
                
                # Calculate objective
                if objective == "min_area":
                    value = A
                elif objective == "max_effectiveness":
                    value = -sol['effectiveness']
                else:
                    value = A
                
                if value < best_value:
                    best_value = value
                    best_params = {
                        'U_optimal': U,
                        'A_optimal': A,
                        'effectiveness': sol['effectiveness'],
                        'Q_W': sol['Q_W'],
                        'objective_value': value,
                        'success': True,
                        'message': 'Grid search optimization'
                    }
                    
            except:
                continue
    
    return best_params if best_params else {
        'U_optimal': None,
        'A_optimal': None,
        'success': False,
        'message': 'No feasible solution found'
    }


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def batch_analysis(
    cases: List[Dict[str, float]],
    configs: Optional[List[FlowConfig]] = None,
    fluid_hot: Optional[FluidType] = None,
    fluid_cold: Optional[FluidType] = None
) -> Union[List[Dict], 'pd.DataFrame']:
    """
    Perform heat exchanger analysis on multiple cases.
    
    Args:
        cases: List of dictionaries with input parameters
        configs: List of configurations to analyze (default: all)
        fluid_hot, fluid_cold: Fluid types for property lookup
    
    Returns:
        Results as DataFrame (if pandas available) or list of dicts
    """
    if configs is None:
        configs = [FlowConfig.PARALLEL, FlowConfig.COUNTER, FlowConfig.CROSSFLOW]
    
    results = []
    
    for case_idx, case in enumerate(cases):
        # Extract case parameters
        m_dot_h = case.get('m_dot_hot', 0.1)
        m_dot_c = case.get('m_dot_cold', 0.1)
        T_h_in = case.get('T_hot_in', 80.0)
        T_c_in = case.get('T_cold_in', 20.0)
        
        # Handle fluid properties
        if fluid_hot:
            fluid_h = FluidPropertyLibrary.get_fluid(fluid_hot, T_h_in)
            cp_h = fluid_h.specific_heat
        else:
            cp_h = case.get('cp_hot', 4180.0)
        
        if fluid_cold:
            fluid_c = FluidPropertyLibrary.get_fluid(fluid_cold, T_c_in)
            cp_c = fluid_c.specific_heat
        else:
            cp_c = case.get('cp_cold', 4180.0)
        
        U = case.get('U', 200.0)
        A = case.get('A', 1.0)
        
        for config in configs:
            try:
                sol = solve_heat_exchanger(
                    m_dot_h, m_dot_c, T_h_in, T_c_in,
                    cp_h, cp_c, U, A, config
                )
                
                # Add case identifier
                sol['case_id'] = case_idx
                sol['config'] = config.value
                sol['m_dot_hot'] = m_dot_h
                sol['m_dot_cold'] = m_dot_c
                sol['T_hot_in'] = T_h_in
                sol['T_cold_in'] = T_c_in
                
                results.append(sol)
                
            except Exception as e:
                warnings.warn(f"Failed case {case_idx}, config {config}: {e}")
                continue
    
    # Convert to DataFrame if pandas available
    if HAS_PANDAS and results:
        return pd.DataFrame(results)
    else:
        return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def compare_configurations_plot(
    m_dot_hot: float,
    m_dot_cold: float,
    T_hot_in: float,
    T_cold_in: float,
    cp_hot: Union[float, Callable],
    cp_cold: Union[float, Callable],
    U_values: List[float],
    A: float,
    tube_params: Optional[Dict] = None,
    mixing_conditions: Optional[List[MixingCondition]] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Create comprehensive comparison plots for different configurations.
    
    Args:
        m_dot_hot, m_dot_cold: Mass flow rates (kg/s)
        T_hot_in, T_cold_in: Inlet temperatures (°C)
        cp_hot, cp_cold: Specific heats (J/kgK) or callables
        U_values: List of U values to evaluate
        A: Heat transfer area (m²)
        tube_params: Parameters for pressure drop calculation
        mixing_conditions: List of mixing conditions for crossflow
        figsize: Figure size
    """
    if not HAS_MATPLOTLIB:
        warnings.warn("Matplotlib not available. Cannot create plots.")
        return
    
    # Default mixing conditions
    if mixing_conditions is None:
        mixing_conditions = [MixingCondition.BOTH_UNMIXED]
    
    # Prepare configurations
    configs = [
        (FlowConfig.PARALLEL, None),
        (FlowConfig.COUNTER, None),
    ]
    
    # Add crossflow with different mixing conditions
    for mixing in mixing_conditions:
        configs.append((FlowConfig.CROSSFLOW, mixing))
    
    # Store results
    results = {}
    for config, mixing in configs:
        label = config.value
        if mixing:
            label += f" ({mixing.value})"
        results[label] = {
            'U': [], 'eff': [], 'Q': [], 'Th_out': [], 'Tc_out': [], 'dP': []
        }
    
    # Calculate results
    for U in U_values:
        for config, mixing in configs:
            try:
                sol = solve_heat_exchanger(
                    m_dot_hot, m_dot_cold, T_hot_in, T_cold_in,
                    cp_hot, cp_cold, U, A, config, mixing
                )
                
                label = config.value
                if mixing:
                    label += f" ({mixing.value})"
                
                results[label]['U'].append(U)
                results[label]['eff'].append(sol['effectiveness'])
                results[label]['Q'].append(sol['Q_W'])
                results[label]['Th_out'].append(sol['T_hot_out'])
                results[label]['Tc_out'].append(sol['T_cold_out'])
                
                # Pressure drop (if tube_params provided)
                if tube_params:
                    fluid = FluidPropertyLibrary.water_properties(25.0)
                    pd = pressure_drop(
                        m_dot_cold, fluid, tube_params,
                        model=FrictionModel.DARCY_WEISBACH
                    )
                    results[label]['dP'].append(pd['deltaP_total_Pa'])
                else:
                    results[label]['dP'].append(0.0)
                    
            except Exception as e:
                warnings.warn(f"Failed U={U}, config={config}: {e}")
                continue
    
    # Create plots
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    
    # Effectiveness plot
    ax1 = axs[0, 0]
    for label, data in results.items():
        if data['U']:  # Check if there's data
            ax1.plot(data['U'], data['eff'], label=label, marker='o')
    ax1.set_xlabel('U (W/m²K)')
    ax1.set_ylabel('Effectiveness ε')
    ax1.set_title('Effectiveness vs Heat Transfer Coefficient')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Heat transfer rate plot
    ax2 = axs[0, 1]
    for label, data in results.items():
        if data['U']:
            ax2.plot(data['U'], [q/1000 for q in data['Q']], label=label, marker='s')
    ax2.set_xlabel('U (W/m²K)')
    ax2.set_ylabel('Heat Transfer Rate (kW)')
    ax2.set_title('Heat Transfer Rate vs Heat Transfer Coefficient')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Temperature plot
    ax3 = axs[1, 0]
    for label, data in results.items():
        if data['U']:
            ax3.plot(data['U'], data['Th_out'], label=f'{label} (Hot)', linestyle='-')
            ax3.plot(data['U'], data['Tc_out'], label=f'{label} (Cold)', linestyle='--')
    ax3.set_xlabel('U (W/m²K)')
    ax3.set_ylabel('Outlet Temperature (°C)')
    ax3.set_title('Outlet Temperatures vs Heat Transfer Coefficient')
    ax3.legend(ncol=2, fontsize='small')
    ax3.grid(True, alpha=0.3)
    
    # Pressure drop plot
    ax4 = axs[1, 1]
    for label, data in results.items():
        if data['U'] and any(d > 0 for d in data['dP']):
            ax4.plot(data['U'], data['dP'], label=label, marker='^')
    ax4.set_xlabel('U (W/m²K)')
    ax4.set_ylabel('Pressure Drop (Pa)')
    ax4.set_title('Pressure Drop vs Heat Transfer Coefficient')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_ntu_effectiveness(
    Cr_values: List[float] = None,
    config: FlowConfig = FlowConfig.COUNTER,
    mixing: MixingCondition = MixingCondition.BOTH_UNMIXED,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Plot NTU-effectiveness curves for different capacity ratios.
    
    Args:
        Cr_values: List of capacity ratios to plot
        config: Flow configuration
        mixing: Mixing condition for crossflow
        figsize: Figure size
    """
    if not HAS_MATPLOTLIB:
        warnings.warn("Matplotlib not available. Cannot create plots.")
        return
    
    if Cr_values is None:
        Cr_values = [0.1, 0.25, 0.5, 0.75, 1.0]
    
    NTU_range = np.linspace(0.1, 10, 100) if HAS_NUMPY else [0.1 + i*0.1 for i in range(100)]
    
    plt.figure(figsize=figsize)
    
    for Cr in Cr_values:
        eps_values = []
        for NTU in NTU_range:
            eps = effectiveness_ntu(NTU, Cr, config, mixing)
            eps_values.append(eps)
        
        plt.plot(NTU_range, eps_values, label=f'Cr = {Cr}', linewidth=2)
    
    plt.xlabel('Number of Transfer Units (NTU)')
    plt.ylabel('Effectiveness (ε)')
    plt.title(f'NTU-Effectiveness ({config.value}, {mixing.value})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 10])
    plt.ylim([0, 1])
    plt.show()


# ============================================================================
# EXAMPLE AND DEMONSTRATION
# ============================================================================

def comprehensive_example() -> None:
    """Demonstrate all features of the toolkit."""
    print("=" * 60)
    print("HEAT EXCHANGER ANALYSIS TOOLKIT - DEMONSTRATION")
    print("=" * 60)
    
    # 1. Fluid properties demonstration
    print("\n1. Fluid Properties:")
    water_25 = FluidPropertyLibrary.water_properties(25.0)
    water_80 = FluidPropertyLibrary.water_properties(80.0)
    air_25 = FluidPropertyLibrary.air_properties(25.0)
    
    print(f"Water at 25°C: ρ={water_25.density:.1f} kg/m³, cp={water_25.specific_heat:.0f} J/kgK")
    print(f"Water at 80°C: ρ={water_80.density:.1f} kg/m³, cp={water_80.specific_heat:.0f} J/kgK")
    print(f"Air at 25°C: ρ={air_25.density:.3f} kg/m³, cp={air_25.specific_heat:.0f} J/kgK")
    
    # 2. Basic heat exchanger calculation
    print("\n2. Basic Heat Exchanger Calculation:")
    
    m_dot_h = 0.5  # kg/s
    m_dot_c = 0.8  # kg/s
    Th_in = 120.0  # °C
    Tc_in = 25.0   # °C
    U = 300.0      # W/m²K
    A = 2.0        # m²
    
    # Using temperature-dependent properties
    cp_h_func = lambda T: FluidPropertyLibrary.water_properties(T).specific_heat
    cp_c_func = lambda T: FluidPropertyLibrary.water_properties(T).specific_heat
    
    for config in [FlowConfig.PARALLEL, FlowConfig.COUNTER, FlowConfig.CROSSFLOW]:
        sol = solve_heat_exchanger(
            m_dot_h, m_dot_c, Th_in, Tc_in,
            cp_h_func, cp_c_func, U, A, config
        )
        print(f"\n{config.value.upper()}:")
        print(f"  Q = {sol['Q_W']/1000:.2f} kW, ε = {sol['effectiveness']:.3f}")
        print(f"  Th_out = {sol['T_hot_out']:.1f}°C, Tc_out = {sol['T_cold_out']:.1f}°C")
    
    # 3. Pressure drop calculation
    print("\n3. Pressure Drop Calculation:")
    
    tube_geometry = {
        'type': 'circular',
        'diameter': 0.02,  # 20 mm
        'length': 5.0,     # 5 m
        'K_minor': 2.5     # Minor loss coefficient
    }
    
    pd_result = pressure_drop(
        m_dot=m_dot_c,
        fluid=water_25,
        geometry=tube_geometry,
        model=FrictionModel.CHURCHILL,
        roughness=0.000045,  # Commercial steel
        n_parallel=4
    )
    
    print(f"Re = {pd_result['Re']:.0f}, V = {pd_result['velocity_m_s']:.2f} m/s")
    print(f"f = {pd_result['friction_factor']:.4f}")
    print(f"ΔP_total = {pd_result['deltaP_total_Pa']:.0f} Pa ({pd_result['deltaP_total_Pa']/1000:.2f} kPa)")
    
    # 4. Fouling impact
    print("\n4. Fouling Impact Analysis:")
    
    fouling_result = fouling_impact(
        U_clean=U,
        fouling_factors={'hot_side': 0.0001, 'cold_side': 0.0001},
        time_years=2.0,
        fouling_rate=0.00005
    )
    
    print(f"U_clean = {fouling_result['U_clean_W_m2K']:.1f} W/m²K")
    print(f"U_dirty = {fouling_result['U_dirty_W_m2K']:.1f} W/m²K")
    print(f"Performance reduction = {fouling_result['effectiveness_reduction']:.1%}")
    print(f"Cleaning recommended: {fouling_result['cleaning_recommended']}")
    
    # 5. Design optimization
    print("\n5. Design Optimization:")
    
    if HAS_SCIPY:
        constraints = {
            'min_effectiveness': 0.7,
            'max_dP': 20000  # Pa
        }
        
        opt_result = optimize_hx_design(
            m_dot_hot=m_dot_h,
            m_dot_cold=m_dot_c,
            T_hot_in=Th_in,
            T_cold_in=Tc_in,
            cp_hot=cp_h_func(Th_in),
            cp_cold=cp_c_func(Tc_in),
            constraints=constraints,
            objective="min_area",
            config=FlowConfig.COUNTER
        )
        
        if opt_result['success']:
            print(f"Optimal U = {opt_result['U_optimal']:.1f} W/m²K")
            print(f"Optimal A = {opt_result['A_optimal']:.2f} m²")
            print(f"Effectiveness = {opt_result['effectiveness']:.3f}")
            print(f"Q = {opt_result['Q_W']/1000:.2f} kW")
        else:
            print(f"Optimization failed: {opt_result['message']}")
    else:
        print("SciPy not available. Skipping optimization example.")
    
    # 6. Batch processing
    print("\n6. Batch Processing Example:")
    
    cases = [
        {'m_dot_hot': 0.3, 'm_dot_cold': 0.5, 'T_hot_in': 100, 'T_cold_in': 20, 'U': 250, 'A': 1.5},
        {'m_dot_hot': 0.5, 'm_dot_cold': 0.8, 'T_hot_in': 120, 'T_cold_in': 25, 'U': 300, 'A': 2.0},
        {'m_dot_hot': 0.8, 'm_dot_cold': 1.2, 'T_hot_in': 150, 'T_cold_in': 30, 'U': 400, 'A': 3.0},
    ]
    
    batch_results = batch_analysis(
        cases=cases,
        configs=[FlowConfig.PARALLEL, FlowConfig.COUNTER],
        fluid_hot=FluidType.WATER,
        fluid_cold=FluidType.WATER
    )
    
    if HAS_PANDAS:
        print("Batch results summary:")
        if isinstance(batch_results, pd.DataFrame):
            summary = batch_results.groupby('config').agg({
                'effectiveness': 'mean',
                'Q_W': 'mean'
            }).round(3)
            print(summary)
    else:
        print(f"Processed {len(batch_results)} cases")
    
    # 7. Visualization (if matplotlib available)
    if HAS_MATPLOTLIB:
        print("\n7. Generating Comparison Plots...")
        
        # Prepare parameters for pressure drop calculation
        tube_params = {
            'type': 'circular',
            'diameter': 0.015,
            'length': 3.0,
            'K_minor': 1.5
        }
        
        # Generate U values for comparison
        U_vals = np.linspace(50, 1000, 20) if HAS_NUMPY else [50, 100, 200, 400, 600, 800, 1000]
        
        compare_configurations_plot(
            m_dot_hot=m_dot_h,
            m_dot_cold=m_dot_c,
            T_hot_in=Th_in,
            T_cold_in=Tc_in,
            cp_hot=cp_h_func,
            cp_cold=cp_c_func,
            U_values=U_vals,
            A=2.0,
            tube_params=tube_params,
            mixing_conditions=[
                MixingCondition.BOTH_UNMIXED,
                MixingCondition.BOTH_MIXED
            ]
        )
        
        # NTU-effectiveness plot
        plot_ntu_effectiveness(
            Cr_values=[0.2, 0.5, 0.8, 1.0],
            config=FlowConfig.CROSSFLOW,
            mixing=MixingCondition.BOTH_UNMIXED
        )
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run the comprehensive example
    comprehensive_example()
    
    # Quick test for basic functionality
    print("\n\nQuick Test:")
    m_dot_h = 0.1
    m_dot_c = 0.15
    Th_in = 80.0
    Tc_in = 20.0
    U = 200.0
    A = 1.0
    
    # Use default water properties
    cp_water = FluidPropertyLibrary.water_properties(25.0).specific_heat
    
    for cfg in [FlowConfig.PARALLEL, FlowConfig.COUNTER, FlowConfig.CROSSFLOW]:
        sol = solve_heat_exchanger(
            m_dot_h, m_dot_c, Th_in, Tc_in,
            cp_water, cp_water, U, A, cfg
        )
        print(f"{cfg.value}: Q={sol['Q_W']:.0f} W, ε={sol['effectiveness']:.3f}")