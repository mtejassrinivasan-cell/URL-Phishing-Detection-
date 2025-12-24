import re
import numpy as np
from typing import Dict, List
from collections import Counter
import pandas as pd  # For entropy

def entropy(url: str) -> float:
    """Shannon entropy of URL chars (high = obfuscated/phishing)"""
    char_counts = Counter(url.lower())
    length = len(url)
    if length == 0:
        return 0.0
    entropy_val = -sum((count / length) * np.log2(count / length) for count in char_counts.values() if count > 0)
    return entropy_val

def extract_features(url: str) -> Dict[str, float]:
    """
    Extracts lexical and structural features from a URL for phishing detection.
    Features inspired by prior research (e.g., Garera et al., 2007; Ma et al., 2009).
    Uses robust string parsing to handle malformed/IPv6 URLs (no urlparse crashes).
    Returns a dictionary of features.
    """
    features = {}
    
    # Basic length features (robust string-based parsing)
    features['url_length'] = len(url)
    
    # Extract hostname and path via string split (handles IPv6/malformed)
    if '//' in url:
        after_protocol = url.split('//', 1)[1]
    else:
        after_protocol = url
    
    host_end = after_protocol.find('/')
    if host_end != -1:
        hostname = after_protocol[:host_end]
        path = after_protocol[host_end + 1:]
    else:
        hostname = after_protocol
        path = ''
    
    features['hostname_length'] = len(hostname)
    features['path_length'] = len(path)
    
    # Count separators and special chars
    features['num_dots'] = url.count('.')
    features['num_slashes'] = url.count('/')
    features['num_hyphens'] = url.count('-')
    features['num_underscores'] = url.count('_')
    features['num_percent'] = url.count('%')
    features['num_qm'] = url.count('?')
    features['num_eq'] = url.count('=')
    features['num_amp'] = url.count('&')
    features['num_at'] = url.count('@')
    
    # Digit and letter ratios
    digits = sum(c.isdigit() for c in url)
    letters = sum(c.isalpha() for c in url)
    features['digit_ratio'] = digits / max(len(url), 1)
    features['letter_ratio'] = letters / max(len(url), 1)
    
    # IP address presence (phishing indicator) - simple regex (avoids IPv6 issues)
    ip_pattern = r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'  # IPv4 only; IPv6 rare in phishing
    features['has_ip'] = 1.0 if re.search(ip_pattern, url) else 0.0
    
    # Sensitive keywords (common in phishing)
    sensitive_keywords = ['login', 'secure', 'account', 'bank', 'paypal', 'update', 'verify', 'password', 'sign', 'free']
    features['num_sensitive'] = sum(1 for kw in sensitive_keywords if kw in url.lower())
    
    # Subdomain count (high count often suspicious)
    domain_parts = hostname.split('.')
    features['num_subdomains'] = max(0, len(domain_parts) - 2)  # Exclude TLD and main domain
    
    # HTTPS presence (legit sites often use it)
    features['has_https'] = 1.0 if url.lower().startswith('https') else 0.0
    
    # URL entropy
    features['url_entropy'] = entropy(url)
    
    return features

def prepare_features(df: 'pd.DataFrame') -> 'pd.DataFrame':
    """
    Applies feature extraction to entire dataset.
    """
    import pandas as pd
    feature_list = []
    for _, row in df.iterrows():
        feats = extract_features(row['url'])
        feats['status'] = row['status']  # Keep label
        feature_list.append(feats)
    return pd.DataFrame(feature_list)