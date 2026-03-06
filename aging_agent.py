"""
SafeSpan AI — Structural Aging Clock Agent
Estimates remaining structural life, calculates degradation rate,
and provides lifecycle intelligence for infrastructure assets.
"""

import numpy as np
from datetime import datetime, timedelta
import uuid


class AgingClockAgent:
    """
    Predicts remaining structural life by analyzing health trajectory,
    crack severity, and deterioration rates. Generates lifecycle
    intelligence suitable for government infrastructure reports.
    """

    # Material-specific lifespan baselines (years)
    LIFESPAN_BASELINES = {
        'Bridge': {'design_life': 75, 'avg_actual': 55, 'critical_threshold': 25},
        'Flyover': {'design_life': 60, 'avg_actual': 45, 'critical_threshold': 20},
        'Building': {'design_life': 80, 'avg_actual': 60, 'critical_threshold': 30},
        'Tunnel': {'design_life': 100, 'avg_actual': 70, 'critical_threshold': 35},
        'Dam': {'design_life': 100, 'avg_actual': 75, 'critical_threshold': 40},
        'Default': {'design_life': 70, 'avg_actual': 50, 'critical_threshold': 25},
    }

    def __init__(self):
        self.analysis_id = f"AGE-{uuid.uuid4().hex[:8].upper()}"

    def run_analysis(
        self,
        health_score: float,
        crack_probability: float,
        structure_type: str = 'Bridge',
        structure_age_years: int = 20,
        forecast_scores: list = None,
    ) -> dict:
        """
        Run complete structural aging analysis.

        Args:
            health_score: Current health score (0-100)
            crack_probability: Crack probability (0-1)
            structure_type: Type of infrastructure
            structure_age_years: Current age in years
            forecast_scores: List of projected health scores over time

        Returns:
            Comprehensive aging analysis dict
        """
        baseline = self.LIFESPAN_BASELINES.get(
            structure_type, self.LIFESPAN_BASELINES['Default']
        )

        # 1. Degradation rate
        degradation = self._calc_degradation_rate(
            health_score, crack_probability, structure_age_years
        )

        # 2. Remaining life estimation
        remaining = self._estimate_remaining_life(
            health_score, degradation['annual_rate'],
            baseline, structure_age_years
        )

        # 3. Aging status classification
        aging_status = self._classify_aging(
            remaining['years'], degradation['annual_rate'], baseline
        )

        # 4. Risk acceleration
        acceleration = self._detect_acceleration(
            degradation['annual_rate'], health_score,
            remaining['years'], forecast_scores
        )

        # 5. Year-by-year projection
        projection = self._project_lifecycle(
            health_score, degradation['annual_rate'],
            remaining['years'], baseline
        )

        # 6. Criticality score (0-100)
        criticality = self._compute_criticality(
            health_score, remaining['years'],
            degradation['annual_rate'], crack_probability
        )

        return {
            'analysis_id': self.analysis_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'structure_type': structure_type,
            'structure_age_years': structure_age_years,
            'design_life_years': baseline['design_life'],
            'health_score': health_score,

            # Remaining life
            'remaining_life_years': remaining['years'],
            'remaining_life_months': remaining['months'],
            'safe_operation_until': remaining['safe_until'],
            'critical_year': remaining['critical_year'],
            'life_percentage_used': remaining['pct_used'],

            # Degradation
            'annual_degradation_rate': degradation['annual_rate'],
            'monthly_degradation_rate': degradation['monthly_rate'],
            'degradation_trend': degradation['trend'],
            'degradation_trend_icon': degradation['trend_icon'],
            'degradation_trend_color': degradation['trend_color'],

            # Aging status
            'aging_status': aging_status['status'],
            'aging_color': aging_status['color'],
            'aging_icon': aging_status['icon'],
            'aging_description': aging_status['description'],

            # Acceleration warning
            'acceleration_detected': acceleration['detected'],
            'acceleration_severity': acceleration['severity'],
            'acceleration_message': acceleration['message'],
            'life_below_3_years': remaining['years'] < 3,

            # Projection data (for charts)
            'projection_years': projection['years'],
            'projection_scores': projection['scores'],
            'projection_labels': projection['labels'],
            'projected_critical_year_label': projection['critical_label'],

            # Criticality
            'criticality_score': criticality['score'],
            'criticality_level': criticality['level'],
            'criticality_color': criticality['color'],

            # Combined summary
            'summary_text': self._generate_summary(
                remaining, degradation, aging_status, structure_type
            ),
        }

    def _calc_degradation_rate(
        self, health_score: float, crack_prob: float, age: int
    ) -> dict:
        """Calculate annual degradation rate."""
        # Base rate from health score
        if health_score >= 80:
            base_rate = np.random.uniform(0.8, 1.5)
        elif health_score >= 60:
            base_rate = np.random.uniform(1.5, 3.5)
        elif health_score >= 40:
            base_rate = np.random.uniform(3.5, 6.0)
        elif health_score >= 20:
            base_rate = np.random.uniform(6.0, 10.0)
        else:
            base_rate = np.random.uniform(10.0, 15.0)

        # Crack acceleration
        crack_factor = 1.0 + (crack_prob * 0.8)

        # Age acceleration (older structures degrade faster)
        age_factor = 1.0 + max(0, (age - 30) * 0.02)

        annual_rate = round(base_rate * crack_factor * age_factor, 1)
        monthly_rate = round(annual_rate / 12, 2)

        # Trend classification
        if annual_rate > 8:
            trend = 'Rapidly Increasing'
            trend_icon = '⚡'
            trend_color = '#ff1744'
        elif annual_rate > 4:
            trend = 'Increasing'
            trend_icon = '📈'
            trend_color = '#ff6d00'
        elif annual_rate > 2:
            trend = 'Moderate'
            trend_icon = '➡️'
            trend_color = '#ffab00'
        else:
            trend = 'Stable'
            trend_icon = '✅'
            trend_color = '#00e676'

        return {
            'annual_rate': annual_rate,
            'monthly_rate': monthly_rate,
            'trend': trend,
            'trend_icon': trend_icon,
            'trend_color': trend_color,
        }

    def _estimate_remaining_life(
        self, health_score: float, annual_rate: float,
        baseline: dict, current_age: int
    ) -> dict:
        """Estimate remaining safe structural life."""
        # Score at which structure is unsafe
        critical_score = 20.0

        if annual_rate > 0:
            points_to_critical = max(0, health_score - critical_score)
            remaining_years = points_to_critical / annual_rate
        else:
            remaining_years = baseline['design_life'] - current_age

        remaining_years = max(0.5, min(remaining_years, 50))
        remaining_months = int(remaining_years * 12)

        safe_until = (datetime.now() + timedelta(days=remaining_years * 365)).strftime('%Y')
        critical_year = (datetime.now() + timedelta(days=remaining_years * 365)).strftime('%B %Y')

        total_expected = baseline['design_life']
        pct_used = round(min(100, (current_age / total_expected) * 100), 1)

        return {
            'years': round(remaining_years, 1),
            'months': remaining_months,
            'safe_until': safe_until,
            'critical_year': critical_year,
            'pct_used': pct_used,
        }

    def _classify_aging(
        self, remaining_years: float, annual_rate: float, baseline: dict
    ) -> dict:
        """Classify the aging status."""
        if remaining_years >= 15 and annual_rate < 3:
            return {
                'status': 'Stable',
                'color': '#00e676',
                'icon': '🟢',
                'description': 'Structure aging within normal parameters. No immediate concerns.'
            }
        elif remaining_years >= 8 and annual_rate < 6:
            return {
                'status': 'Aging',
                'color': '#ffab00',
                'icon': '🟡',
                'description': 'Moderate aging detected. Increased monitoring and maintenance planning recommended.'
            }
        elif remaining_years >= 3:
            return {
                'status': 'Advanced Aging',
                'color': '#ff6d00',
                'icon': '🟠',
                'description': 'Significant aging observed. Structural reinforcement or replacement planning required.'
            }
        else:
            return {
                'status': 'Critical Aging',
                'color': '#ff1744',
                'icon': '🔴',
                'description': 'Critical end-of-life approaching. Immediate engineering assessment and action plan required.'
            }

    def _detect_acceleration(
        self, annual_rate: float, health_score: float,
        remaining_years: float, forecast_scores: list = None
    ) -> dict:
        """Detect if degradation is accelerating beyond normal."""
        threshold_rate = 6.0

        if annual_rate > threshold_rate and remaining_years < 5:
            return {
                'detected': True,
                'severity': 'CRITICAL',
                'message': f'⚡ ACCELERATED STRUCTURAL AGING DETECTED — Degradation rate {annual_rate}%/yr exceeds safety threshold. Estimated remaining life: {remaining_years} years.'
            }
        elif annual_rate > threshold_rate:
            return {
                'detected': True,
                'severity': 'WARNING',
                'message': f'⚠️ Elevated degradation rate ({annual_rate}%/yr) detected. Rate exceeds normal threshold of {threshold_rate}%/yr.'
            }
        elif remaining_years < 3:
            return {
                'detected': True,
                'severity': 'CRITICAL',
                'message': f'🚨 REMAINING LIFE CRITICAL — Less than 3 years of safe operational life remaining ({remaining_years} years).'
            }
        else:
            return {
                'detected': False,
                'severity': 'NONE',
                'message': 'Degradation rate within expected parameters. No acceleration detected.'
            }

    def _project_lifecycle(
        self, health_score: float, annual_rate: float,
        remaining_years: float, baseline: dict
    ) -> dict:
        """Project health scores year-by-year into the future."""
        max_years = min(int(remaining_years) + 5, 25)
        years = list(range(0, max_years + 1))
        scores = []
        labels = []
        critical_label = None

        np.random.seed(int(health_score * 10) % 2**31)

        current = health_score
        for y in years:
            scores.append(round(max(0, current), 1))
            label = (datetime.now() + timedelta(days=y * 365)).strftime('%Y')
            labels.append(label)

            if current <= 20 and critical_label is None:
                critical_label = label

            # Degradation with slight acceleration over time
            acceleration = 1 + (y * 0.03)
            noise = np.random.normal(0, 0.5)
            current -= (annual_rate * acceleration) + noise

        if critical_label is None:
            critical_label = labels[-1] if scores[-1] <= 20 else "Beyond projection"

        return {
            'years': years,
            'scores': scores,
            'labels': labels,
            'critical_label': critical_label,
        }

    def _compute_criticality(
        self, health_score: float, remaining_years: float,
        annual_rate: float, crack_prob: float
    ) -> dict:
        """Compute a combined criticality score (0-100, higher = worse)."""
        # Health component (0-30)
        health_component = max(0, (100 - health_score) * 0.3)

        # Remaining life component (0-30)
        life_component = max(0, min(30, (10 - remaining_years) * 3))

        # Degradation component (0-20)
        degrad_component = min(20, annual_rate * 2)

        # Crack component (0-20)
        crack_component = crack_prob * 20

        score = round(min(100, health_component + life_component +
                         degrad_component + crack_component), 1)

        if score >= 75:
            level, color = 'Extreme', '#ff1744'
        elif score >= 50:
            level, color = 'High', '#ff6d00'
        elif score >= 30:
            level, color = 'Moderate', '#ffab00'
        else:
            level, color = 'Low', '#00e676'

        return {'score': score, 'level': level, 'color': color}

    def _generate_summary(
        self, remaining: dict, degradation: dict,
        aging_status: dict, structure_type: str
    ) -> str:
        """Generate a human-readable aging summary."""
        return (
            f"The {structure_type.lower()} has an estimated remaining safe life of "
            f"{remaining['years']} years ({remaining['months']} months), "
            f"with projected safe operation until {remaining['safe_until']}. "
            f"Current degradation rate is {degradation['annual_rate']}% per year "
            f"({degradation['trend'].lower()} trend). "
            f"Aging status: {aging_status['status']}. "
            f"Life expectancy used: {remaining['pct_used']}%."
        )
