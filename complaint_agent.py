"""
SafeSpan AI — Government Complaint & Emergency Alert Agent
Automatically generates government-level complaints when infrastructure
risk becomes critical. Includes CRS calculation, location intelligence,
complaint generation, and PDF report capabilities.
"""

import numpy as np
from datetime import datetime
import uuid
import json


class GovernmentComplaintAgent:
    """
    Generates official government inspection complaints and emergency alerts
    when Critical Risk Score triggers threshold conditions.
    """

    # Demo infrastructure location database
    LOCATION_DB = {
        'BRG-2024-A1': {
            'name': 'Highway Bridge #47',
            'city': 'Mumbai, Maharashtra',
            'lat': 19.0760, 'lon': 72.8777,
            'landmark': 'Near Western Express Highway, Andheri',
            'type': 'Bridge', 'ward': 'K-West Ward'
        },
        'BLD-2024-B3': {
            'name': 'Metro Tower East',
            'city': 'Delhi, NCR',
            'lat': 28.6139, 'lon': 77.2090,
            'landmark': 'Near Connaught Place Metro Station',
            'type': 'Building', 'ward': 'New Delhi District'
        },
        'BRG-2024-C7': {
            'name': 'River Crossing #12',
            'city': 'Kolkata, West Bengal',
            'lat': 22.5726, 'lon': 88.3639,
            'landmark': 'Near Howrah Bridge Approach Road',
            'type': 'Bridge', 'ward': 'Howrah Municipal'
        },
        'TNL-2024-D2': {
            'name': 'Underground Tunnel A',
            'city': 'Bengaluru, Karnataka',
            'lat': 12.9716, 'lon': 77.5946,
            'landmark': 'Near M.G. Road Metro Station',
            'type': 'Tunnel', 'ward': 'BBMP East Zone'
        },
        'DAM-2024-E5': {
            'name': 'Reservoir Dam #3',
            'city': 'Chennai, Tamil Nadu',
            'lat': 13.0827, 'lon': 80.2707,
            'landmark': 'Near Chembarambakkam Lake',
            'type': 'Dam', 'ward': 'Tiruvallur District'
        },
        'BRG-2024-F8': {
            'name': 'Overpass Section 9',
            'city': 'Hyderabad, Telangana',
            'lat': 17.3850, 'lon': 78.4867,
            'landmark': 'Near HITEC City Flyover',
            'type': 'Bridge', 'ward': 'Serilingampally Zone'
        },
    }

    STATUS_OPTIONS = ['🔴 Reported', '🟡 Pending Inspection', '🟢 Resolved',
                      '🔵 Under Review', '⚪ Scheduled']

    def __init__(self):
        self.ticket_id = f"GOV-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"

    def compute_crs(
        self,
        health_score: float,
        predicted_risk_pct: float,
        aging_risk_factor: float,
        impact_radius_severity: float,
    ) -> dict:
        """
        Compute Critical Risk Score (CRS).

        CRS = (100 - health) × 0.4
            + predicted_risk × 0.3
            + aging_risk × 0.2
            + impact_severity × 0.1
        """
        component_health = (100 - health_score) * 0.4
        component_predicted = predicted_risk_pct * 0.3
        component_aging = aging_risk_factor * 0.2
        component_impact = impact_radius_severity * 0.1

        crs = round(component_health + component_predicted +
                    component_aging + component_impact, 1)
        crs = min(100, max(0, crs))

        if crs >= 85:
            level, color = 'EXTREME', '#ff1744'
        elif crs >= 70:
            level, color = 'CRITICAL', '#ff6d00'
        elif crs >= 50:
            level, color = 'HIGH', '#ffab00'
        elif crs >= 30:
            level, color = 'MODERATE', '#00d4ff'
        else:
            level, color = 'LOW', '#00e676'

        return {
            'score': crs,
            'level': level,
            'color': color,
            'components': {
                'health': round(component_health, 1),
                'predicted_risk': round(component_predicted, 1),
                'aging_risk': round(component_aging, 1),
                'impact_severity': round(component_impact, 1),
            }
        }

    def should_trigger(
        self,
        crs_score: float,
        health_score: float,
        remaining_life_years: float,
    ) -> dict:
        """Check if government complaint should be triggered."""
        reasons = []

        if crs_score >= 70:
            reasons.append(f"Critical Risk Score ({crs_score}) ≥ 70")
        if health_score < 50:
            reasons.append(f"Structural Health Index ({health_score}%) < 50%")
        if remaining_life_years < 3:
            reasons.append(f"Remaining Structural Life ({remaining_life_years}yr) < 3 years")

        triggered = len(reasons) > 0

        return {
            'triggered': triggered,
            'reasons': reasons,
            'reason_count': len(reasons),
            'severity': 'CRITICAL' if len(reasons) >= 2 else 'HIGH' if triggered else 'NORMAL',
        }

    def generate_complaint(
        self,
        structure_id: str,
        health_score: float,
        remaining_life: float,
        impact_radius: float,
        affected_population: int,
        crs: dict,
        trigger_info: dict,
        custom_location: dict = None,
    ) -> dict:
        """Generate a full government complaint record."""

        location = custom_location or self.LOCATION_DB.get(
            structure_id,
            {
                'name': f'Structure {structure_id}',
                'city': 'Unknown',
                'lat': 20.5937, 'lon': 78.9629,
                'landmark': 'N/A',
                'type': 'Unknown',
                'ward': 'N/A',
            }
        )

        # Determine recommended action based on severity
        if crs['score'] >= 85:
            action = 'IMMEDIATE closure and emergency structural assessment. Evacuate surrounding area. Deploy emergency response team within 12 hours.'
            priority = 'P0 — EMERGENCY'
            priority_color = '#ff1744'
        elif crs['score'] >= 70:
            action = 'Restrict traffic and public access immediately. Commission urgent structural engineering assessment within 48 hours.'
            priority = 'P1 — CRITICAL'
            priority_color = '#ff6d00'
        elif crs['score'] >= 50:
            action = 'Schedule priority inspection within 7 days. Implement load restrictions. Install monitoring sensors.'
            priority = 'P2 — HIGH'
            priority_color = '#ffab00'
        else:
            action = 'Schedule routine inspection within 30 days. Continue periodic monitoring.'
            priority = 'P3 — STANDARD'
            priority_color = '#00d4ff'

        return {
            'ticket_id': self.ticket_id,
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S IST'),
            'generated_date': datetime.now().strftime('%d %B %Y'),

            # Structure info
            'structure_id': structure_id,
            'structure_name': location['name'],
            'structure_type': location['type'],

            # Location
            'city': location['city'],
            'latitude': location['lat'],
            'longitude': location['lon'],
            'landmark': location['landmark'],
            'ward': location.get('ward', 'N/A'),
            'gps_string': f"{location['lat']:.4f}°N, {location['lon']:.4f}°E",

            # Scores
            'health_score': health_score,
            'crs_score': crs['score'],
            'crs_level': crs['level'],
            'crs_color': crs['color'],
            'crs_components': crs['components'],

            # Impact
            'remaining_life_years': remaining_life,
            'impact_radius_m': impact_radius,
            'affected_population': affected_population,

            # Priority
            'priority': priority,
            'priority_color': priority_color,
            'recommended_action': action,

            # Trigger
            'trigger_reasons': trigger_info['reasons'],
            'trigger_severity': trigger_info['severity'],

            # Status
            'status': '🔴 Reported',
            'assigned_officer': 'Pending Assignment',

            # Location data for map
            'location': location,
        }

    def get_all_structures_for_map(self) -> list:
        """Return all structures with location + demo health data for map."""
        np.random.seed(42)
        structures = []
        demo_scores = {
            'BRG-2024-A1': 82.4, 'BLD-2024-B3': 56.1,
            'BRG-2024-C7': 28.3, 'TNL-2024-D2': 91.7,
            'DAM-2024-E5': 44.9, 'BRG-2024-F8': 71.2,
        }

        for sid, loc in self.LOCATION_DB.items():
            score = demo_scores.get(sid, 50.0)
            if score >= 70:
                color, status = '#00e676', 'Safe'
            elif score >= 40:
                color, status = '#ffab00', 'Moderate'
            else:
                color, status = '#ff1744', 'Critical'

            structures.append({
                **loc,
                'id': sid,
                'score': score,
                'color': color,
                'status': status,
            })

        return structures

    def generate_pdf_text(self, complaint: dict) -> str:
        """Generate formatted text for government PDF report."""
        border = "=" * 60
        return f"""
{border}
      GOVERNMENT OF INDIA
      INFRASTRUCTURE SAFETY DIVISION
      EMERGENCY INSPECTION ORDER
{border}

TICKET ID:      {complaint['ticket_id']}
DATE:           {complaint['generated_date']}
PRIORITY:       {complaint['priority']}
STATUS:         {complaint['status']}

{"-" * 60}
SECTION 1: INFRASTRUCTURE IDENTIFICATION
{"-" * 60}

Structure Name:     {complaint['structure_name']}
Structure ID:       {complaint['structure_id']}
Type:               {complaint['structure_type']}
City/District:      {complaint['city']}
Ward/Zone:          {complaint['ward']}
GPS Coordinates:    {complaint['gps_string']}
Nearest Landmark:   {complaint['landmark']}

{"-" * 60}
SECTION 2: RISK ASSESSMENT
{"-" * 60}

Structural Health Score:    {complaint['health_score']}%
Critical Risk Score (CRS):  {complaint['crs_score']} / 100
Risk Classification:        {complaint['crs_level']}
Remaining Structural Life:  {complaint['remaining_life_years']} years
Impact Radius:              {complaint['impact_radius_m']}m
Est. Affected Population:   {complaint['affected_population']:,}

CRS Breakdown:
  - Health Component:       {complaint['crs_components']['health']}
  - Predicted Risk:         {complaint['crs_components']['predicted_risk']}
  - Aging Risk Factor:      {complaint['crs_components']['aging_risk']}
  - Impact Severity:        {complaint['crs_components']['impact_severity']}

{"-" * 60}
SECTION 3: TRIGGER CONDITIONS
{"-" * 60}

Alert Trigger Severity: {complaint['trigger_severity']}
Conditions Met:
{chr(10).join(f'  ✓ {r}' for r in complaint['trigger_reasons'])}

{"-" * 60}
SECTION 4: RECOMMENDED ACTION
{"-" * 60}

{complaint['recommended_action']}

{"-" * 60}
SECTION 5: ASSIGNMENT
{"-" * 60}

Assigned To:        {complaint['assigned_officer']}
Target Response:    Within 48 hours

{border}
  Generated by SafeSpan AI — Infrastructure Intelligence Platform
  This is an auto-generated government report.
{border}
"""
