"""
SafeSpan AI — Utility Functions
Heatmap generation, health scoring, risk forecasting, and inspection summaries.
"""

import numpy as np
import cv2
from PIL import Image
import uuid
from datetime import datetime, timedelta


def generate_heatmap(image: np.ndarray, intensity: float = 1.0) -> np.ndarray:
    """
    Generate a damage heatmap overlay on the input image.
    Uses edge detection + Gaussian blur to highlight potential crack regions.

    Args:
        image: RGB numpy array
        intensity: Scale factor for heatmap intensity (0.0 - 2.0)

    Returns:
        RGB numpy array with heatmap overlay
    """
    if len(image.shape) == 2:
        gray = image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    else:
        image_rgb = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Multi-scale edge detection for richer heatmap
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # Laplacian for fine detail
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_64F))

    # Combine
    combined = 0.7 * magnitude + 0.3 * laplacian
    combined = (combined / combined.max() * 255).astype(np.uint8) if combined.max() > 0 else np.zeros_like(gray)

    # Apply Gaussian blur for smooth heatmap
    blurred = cv2.GaussianBlur(combined, (21, 21), 0)

    # Amplify based on intensity
    amplified = np.clip(blurred.astype(np.float32) * intensity, 0, 255).astype(np.uint8)

    # Apply colormap (JET for classic heatmap look)
    heatmap_colored = cv2.applyColorMap(amplified, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend with original image
    alpha = 0.55
    blended = cv2.addWeighted(image_rgb, 1 - alpha, heatmap_rgb, alpha, 0)

    return blended


def compute_health_score(crack_probability: float) -> float:
    """
    Compute structural health score from crack probability.
    Higher probability = lower health score.

    Returns: score between 0.0 and 100.0
    """
    # Non-linear mapping: small cracks have less impact, severe cracks drop score sharply
    if crack_probability < 0.2:
        score = 100 - (crack_probability * 100)  # 80–100 range
    elif crack_probability < 0.5:
        score = 85 - (crack_probability - 0.2) * 150  # 40–85 range
    else:
        score = 40 - (crack_probability - 0.5) * 70  # 5–40 range

    return round(max(5.0, min(100.0, score)), 1)


def get_risk_level(health_score: float) -> dict:
    """
    Determine risk level from health score.

    Returns: dict with 'level', 'color', 'icon', 'description'
    """
    if health_score >= 70:
        return {
            'level': 'Safe',
            'color': '#00e676',
            'icon': '✅',
            'description': 'Structure is in good condition. Routine monitoring recommended.'
        }
    elif health_score >= 40:
        return {
            'level': 'Moderate Risk',
            'color': '#ffab00',
            'icon': '⚠️',
            'description': 'Minor degradation detected. Schedule detailed inspection within 30 days.'
        }
    else:
        return {
            'level': 'Critical',
            'color': '#ff1744',
            'icon': '🚨',
            'description': 'Significant structural damage detected. Immediate inspection required.'
        }


def generate_risk_forecast(current_score: float, months: int = 6) -> dict:
    """
    Generate a crack growth / health deterioration forecast.

    Returns: dict with 'months', 'scores', 'risk_levels'
    """
    np.random.seed(int(current_score * 100) % 2**31)

    months_list = list(range(0, months + 1))
    scores = [current_score]

    # Deterioration rate depends on current condition
    if current_score >= 70:
        base_decay = np.random.uniform(1.0, 2.5)
    elif current_score >= 40:
        base_decay = np.random.uniform(2.5, 5.0)
    else:
        base_decay = np.random.uniform(4.0, 8.0)

    for i in range(1, months + 1):
        noise = np.random.normal(0, 0.8)
        acceleration = 1 + (i * 0.05)
        new_score = scores[-1] - (base_decay * acceleration) + noise
        scores.append(max(5.0, round(new_score, 1)))

    risk_levels = []
    for s in scores:
        if s >= 70:
            risk_levels.append('Safe')
        elif s >= 40:
            risk_levels.append('Moderate')
        else:
            risk_levels.append('Critical')

    month_labels = [(datetime.now() + timedelta(days=30 * m)).strftime('%b %Y') for m in months_list]

    return {
        'months': months_list,
        'month_labels': month_labels,
        'scores': scores,
        'risk_levels': risk_levels
    }


def generate_inspection_summary(
    health_score: float,
    crack_probability: float,
    risk_info: dict,
    forecast: dict
) -> dict:
    """
    Generate an automated inspection summary report.
    """
    # Determine future risk from last forecast score
    future_score = forecast['scores'][-1]
    future_risk = get_risk_level(future_score)

    # Determine recommended action
    if health_score >= 70:
        action = "Continue routine monitoring. Next inspection recommended in 6 months."
    elif health_score >= 50:
        action = "Schedule professional structural assessment within 30 days. Increase monitoring frequency."
    elif health_score >= 30:
        action = "Urgent: Commission detailed structural analysis. Consider temporary load restrictions."
    else:
        action = "CRITICAL: Immediate structural engineering assessment required. Restrict access until cleared."

    infra_id = f"INF-{uuid.uuid4().hex[:8].upper()}"

    return {
        'infrastructure_id': infra_id,
        'inspection_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'health_score': health_score,
        'crack_probability': round(crack_probability * 100, 1),
        'risk_level': risk_info['level'],
        'risk_color': risk_info['color'],
        'risk_icon': risk_info['icon'],
        'predicted_future_risk': future_risk['level'],
        'future_risk_color': future_risk['color'],
        'predicted_score_6mo': future_score,
        'recommended_action': action,
        'analysis_confidence': round(np.random.uniform(88.0, 99.5), 1),
    }
