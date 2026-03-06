"""
SafeSpan AI — Failure Impact Simulation Agent
Simulates potential impact of critical infrastructure failure.
Calculates impact radius, population affected, traffic disruption,
and economic loss estimates.
"""

import numpy as np
from datetime import datetime
import uuid


class FailureImpactAgent:
    """
    Simulates the cascading impact of infrastructure collapse.
    Automatically activates when Health Score < 50 or Risk = Critical.
    """

    # Structure-type modifiers
    STRUCTURE_PROFILES = {
        'Bridge': {
            'base_radius': 300,
            'pop_density_factor': 1.2,
            'traffic_weight': 1.5,
            'repair_base_cost': 45_000_000,
            'traffic_loss_per_day': 2_500_000,
            'avg_recovery_days': 180,
        },
        'Flyover': {
            'base_radius': 250,
            'pop_density_factor': 1.8,
            'traffic_weight': 1.8,
            'repair_base_cost': 35_000_000,
            'traffic_loss_per_day': 3_200_000,
            'avg_recovery_days': 150,
        },
        'Building': {
            'base_radius': 150,
            'pop_density_factor': 2.5,
            'traffic_weight': 0.8,
            'repair_base_cost': 25_000_000,
            'traffic_loss_per_day': 800_000,
            'avg_recovery_days': 240,
        },
        'Tunnel': {
            'base_radius': 200,
            'pop_density_factor': 1.0,
            'traffic_weight': 2.0,
            'repair_base_cost': 80_000_000,
            'traffic_loss_per_day': 4_000_000,
            'avg_recovery_days': 365,
        },
        'Dam': {
            'base_radius': 800,
            'pop_density_factor': 3.0,
            'traffic_weight': 0.5,
            'repair_base_cost': 200_000_000,
            'traffic_loss_per_day': 1_500_000,
            'avg_recovery_days': 730,
        },
        'Default': {
            'base_radius': 200,
            'pop_density_factor': 1.5,
            'traffic_weight': 1.0,
            'repair_base_cost': 30_000_000,
            'traffic_loss_per_day': 1_500_000,
            'avg_recovery_days': 200,
        }
    }

    def __init__(self):
        self.simulation_id = f"SIM-{uuid.uuid4().hex[:8].upper()}"

    def should_activate(self, health_score: float, risk_level: str) -> bool:
        """Check if simulation should auto-activate."""
        return health_score < 50 or risk_level.lower() == 'critical'

    def run_simulation(
        self,
        health_score: float,
        risk_level: str,
        structure_type: str = 'Bridge',
        crack_probability: float = 0.5,
    ) -> dict:
        """
        Run full failure impact simulation.

        Returns comprehensive impact analysis dict.
        """
        profile = self.STRUCTURE_PROFILES.get(
            structure_type, self.STRUCTURE_PROFILES['Default']
        )

        # Severity multiplier: lower score = higher impact
        severity = self._compute_severity(health_score, crack_probability)

        # 1. Impact Radius
        impact_radius = self._calc_impact_radius(
            health_score, severity, profile['base_radius']
        )

        # 2. Affected Population
        affected_pop = self._estimate_population(
            impact_radius, profile['pop_density_factor'], severity
        )

        # 3. Traffic Disruption
        traffic = self._assess_traffic_disruption(
            severity, profile['traffic_weight']
        )

        # 4. Economic Loss
        economic = self._estimate_economic_loss(
            severity, profile, traffic['recovery_days']
        )

        # 5. Collapse probability
        collapse_prob = self._collapse_probability(health_score, crack_probability)

        # Impact severity classification
        impact_class = self._classify_impact(severity)

        return {
            'simulation_id': self.simulation_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'structure_type': structure_type,
            'health_score': health_score,
            'risk_level': risk_level,
            'severity_index': round(severity, 2),
            'collapse_probability': round(collapse_prob * 100, 1),

            # Impact Radius
            'impact_radius_m': round(impact_radius),
            'inner_zone_m': round(impact_radius * 0.3),
            'outer_zone_m': round(impact_radius * 0.7),

            # Population
            'affected_population': affected_pop['total'],
            'immediate_zone_pop': affected_pop['immediate'],
            'evacuation_zone_pop': affected_pop['evacuation'],
            'advisory_zone_pop': affected_pop['advisory'],

            # Traffic
            'traffic_disruption': traffic['level'],
            'traffic_color': traffic['color'],
            'affected_routes': traffic['affected_routes'],
            'detour_distance_km': traffic['detour_km'],
            'recovery_days': traffic['recovery_days'],

            # Economic
            'total_economic_loss': economic['total'],
            'repair_cost': economic['repair'],
            'traffic_loss': economic['traffic'],
            'productivity_loss': economic['productivity'],
            'emergency_response_cost': economic['emergency'],

            # Classification
            'impact_class': impact_class['label'],
            'impact_color': impact_class['color'],
            'impact_icon': impact_class['icon'],
            'impact_description': impact_class['description'],

            # Recommendations
            'recommendations': self._generate_recommendations(
                severity, structure_type, collapse_prob
            ),
        }

    def _compute_severity(self, health_score: float, crack_prob: float) -> float:
        """Compute overall severity index (0-1)."""
        # Lower health = higher severity
        health_factor = max(0, (50 - health_score) / 50)
        crack_factor = crack_prob
        return np.clip(0.6 * health_factor + 0.4 * crack_factor, 0.1, 1.0)

    def _calc_impact_radius(
        self, health_score: float, severity: float, base_radius: float
    ) -> float:
        """Calculate failure impact radius in meters."""
        # Scale radius based on severity
        multiplier = 0.5 + (severity * 1.5)
        radius = base_radius * multiplier

        # Add randomness for realism
        np.random.seed(int(health_score * 100) % 2**31)
        noise = np.random.uniform(0.9, 1.1)

        return radius * noise

    def _estimate_population(
        self, radius_m: float, density_factor: float, severity: float
    ) -> dict:
        """Estimate affected population based on impact radius."""
        # Assume urban density ~5000 people/km²
        base_density = 5000 * density_factor
        area_km2 = np.pi * (radius_m / 1000) ** 2

        total = int(area_km2 * base_density * (0.5 + severity * 0.5))

        return {
            'total': total,
            'immediate': int(total * 0.15),   # Direct danger zone
            'evacuation': int(total * 0.35),   # Evacuation zone
            'advisory': int(total * 0.50),     # Advisory zone
        }

    def _assess_traffic_disruption(
        self, severity: float, traffic_weight: float
    ) -> dict:
        """Assess traffic disruption level."""
        score = severity * traffic_weight

        if score >= 1.2:
            level, color = 'Severe', '#ff1744'
            routes, detour_km, recovery = 8, 25.0, 180
        elif score >= 0.8:
            level, color = 'High', '#ff6d00'
            routes, detour_km, recovery = 5, 15.0, 90
        elif score >= 0.4:
            level, color = 'Moderate', '#ffab00'
            routes, detour_km, recovery = 3, 8.0, 45
        else:
            level, color = 'Low', '#00e676'
            routes, detour_km, recovery = 1, 3.0, 14

        return {
            'level': level,
            'color': color,
            'affected_routes': routes,
            'detour_km': detour_km,
            'recovery_days': recovery,
        }

    def _estimate_economic_loss(
        self, severity: float, profile: dict, recovery_days: int
    ) -> dict:
        """Estimate economic loss from failure."""
        repair = profile['repair_base_cost'] * (0.3 + severity * 0.7)
        traffic = profile['traffic_loss_per_day'] * recovery_days
        productivity = traffic * 0.4
        emergency = repair * 0.15

        return {
            'repair': round(repair),
            'traffic': round(traffic),
            'productivity': round(productivity),
            'emergency': round(emergency),
            'total': round(repair + traffic + productivity + emergency),
        }

    def _collapse_probability(
        self, health_score: float, crack_prob: float
    ) -> float:
        """Estimate probability of structural collapse."""
        if health_score >= 70:
            base = 0.01
        elif health_score >= 50:
            base = 0.05
        elif health_score >= 30:
            base = 0.15
        elif health_score >= 15:
            base = 0.35
        else:
            base = 0.60

        adjusted = base + (crack_prob * 0.2)
        return np.clip(adjusted, 0.01, 0.95)

    def _classify_impact(self, severity: float) -> dict:
        """Classify overall impact severity."""
        if severity >= 0.8:
            return {
                'label': 'CATASTROPHIC',
                'color': '#ff1744',
                'icon': '🔴',
                'description': 'Potential catastrophic failure with mass casualties and severe economic damage.'
            }
        elif severity >= 0.6:
            return {
                'label': 'SEVERE',
                'color': '#ff6d00',
                'icon': '🟠',
                'description': 'Major failure risk with significant population impact and infrastructure disruption.'
            }
        elif severity >= 0.4:
            return {
                'label': 'SIGNIFICANT',
                'color': '#ffab00',
                'icon': '🟡',
                'description': 'Moderate failure risk requiring urgent preventive intervention.'
            }
        else:
            return {
                'label': 'CONTAINED',
                'color': '#00d4ff',
                'icon': '🔵',
                'description': 'Limited failure scope. Preventive maintenance recommended.'
            }

    def _generate_recommendations(
        self, severity: float, structure_type: str, collapse_prob: float
    ) -> list:
        """Generate prioritized action recommendations."""
        recs = []

        if collapse_prob > 0.3:
            recs.append({
                'priority': 'IMMEDIATE',
                'color': '#ff1744',
                'action': f'Restrict all traffic and public access to {structure_type.lower()} immediately.'
            })
            recs.append({
                'priority': 'IMMEDIATE',
                'color': '#ff1744',
                'action': 'Deploy emergency structural assessment team within 24 hours.'
            })

        if severity >= 0.6:
            recs.append({
                'priority': 'URGENT',
                'color': '#ff6d00',
                'action': 'Notify district emergency management and initiate standby evacuation plan.'
            })
            recs.append({
                'priority': 'URGENT',
                'color': '#ff6d00',
                'action': 'Install real-time structural monitoring sensors immediately.'
            })

        if severity >= 0.4:
            recs.append({
                'priority': 'HIGH',
                'color': '#ffab00',
                'action': 'Commission a detailed structural engineering assessment within 7 days.'
            })
            recs.append({
                'priority': 'HIGH',
                'color': '#ffab00',
                'action': 'Prepare traffic rerouting and public notification plan.'
            })

        recs.append({
            'priority': 'STANDARD',
            'color': '#00d4ff',
            'action': 'Include findings in next government infrastructure assessment report.'
        })
        recs.append({
            'priority': 'STANDARD',
            'color': '#00d4ff',
            'action': f'Schedule follow-up scan of {structure_type.lower()} within 14 days.'
        })

        return recs


def format_currency(amount: float) -> str:
    """Format large numbers as currency string (₹)."""
    if amount >= 1_00_00_00_000:  # 100 Crore+
        return f"₹{amount / 1_00_00_00_000:.1f} Bn"
    elif amount >= 1_00_00_000:  # 1 Crore+
        return f"₹{amount / 1_00_00_000:.1f} Cr"
    elif amount >= 1_00_000:  # 1 Lakh+
        return f"₹{amount / 1_00_000:.1f} L"
    else:
        return f"₹{amount:,.0f}"


def format_population(pop: int) -> str:
    """Format population number."""
    if pop >= 100_000:
        return f"{pop / 1000:.0f}K+"
    elif pop >= 1_000:
        return f"{pop / 1000:.1f}K"
    else:
        return str(pop)
