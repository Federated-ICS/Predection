#!/usr/bin/env python3
"""
Generate realistic post-detection network features for attack sequences.

This script creates training data by:
1. Loading real attack sequences (70 sequences from MITRE, APT groups, malware)
2. Generating realistic network features for each technique transition
3. Creating multiple variants to simulate different attacker behaviors
4. Saving augmented dataset for hybrid GNN training

The network features represent aggregated traffic in the 3 seconds AFTER
a technique is detected, capturing attacker behavior that reveals intent.
"""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np


# ============================================================================
# TECHNIQUE CHARACTERISTICS
# ============================================================================

# Technique categories for pattern generation
RECONNAISSANCE_TECHNIQUES = ['T0846', 'T0888', 'T0840', 'T0842']
LATERAL_MOVEMENT_TECHNIQUES = ['T0800', 'T0812', 'T0859']
EXECUTION_TECHNIQUES = ['T0843', 'T0845', 'T0858', 'T0836', 'T0874']
IMPACT_TECHNIQUES = ['T0814', 'T0809', 'T0831', 'T0828', 'T0881']
EXFILTRATION_TECHNIQUES = ['T0802', 'T0811', 'T0882']
PERSISTENCE_TECHNIQUES = ['T0873', 'T0889', 'T0891']

# ICS ports
ICS_PORTS = [502, 2404, 20000, 44818, 102, 1911, 4840]

# Common ports
COMMON_PORTS = [80, 443, 22, 21, 23, 25, 53, 3389]


# ============================================================================
# POST-DETECTION FEATURE GENERATION
# ============================================================================

def generate_post_detection_features(current_technique, next_technique, variant_type='normal'):
    """
    Generate network features representing traffic AFTER current technique detection.
    
    These features capture what the attacker does in the 3 seconds after detection,
    which reveals their intent and helps predict the next move.
    
    Args:
        current_technique: The technique that was just detected
        next_technique: The technique that comes next (for training)
        variant_type: 'aggressive', 'stealthy', or 'normal'
    
    Returns:
        18 aggregated network features
    """
    
    # ========== SPECIFIC TRANSITION PATTERNS ==========
    
    # Pattern 1: Reconnaissance → Lateral Movement (Targeted)
    if current_technique in RECONNAISSANCE_TECHNIQUES and next_technique in LATERAL_MOVEMENT_TECHNIQUES:
        if variant_type == 'stealthy':
            return [
                random.randint(2, 5),          # packet_count - VERY LOW (3s window)
                random.randint(500, 1500),     # byte_count
                random.uniform(0.5, 1.5),      # packets_per_sec - VERY LOW
                random.uniform(0.95, 1.0),     # tcp_ratio
                random.uniform(0.0, 0.05),     # udp_ratio
                0.0,                           # icmp_ratio
                random.randint(1, 2),          # unique_dest_ports - VERY FEW
                1,                             # is_ics_port - YES
                0,                             # common_ports - NO
                1,                             # unique_sources
                random.randint(1, 2),          # unique_destinations - VERY FEW
                random.uniform(0.2, 0.8),      # connection_rate - VERY LOW
                random.randint(0, 1),          # failed_connections - VERY FEW
                random.randint(1, 3),          # syn_count - VERY LOW
                random.uniform(0.35, 0.55),    # payload_entropy
                random.uniform(0.0, 1.0),      # time_of_day
                random.choice([0, 1]),         # is_weekend
                0                              # time_since_last
            ]
        else:  # normal
            return [
                random.randint(3, 8),          # packet_count - LOW (3s window)
                random.randint(1000, 3000),    # byte_count
                random.uniform(1, 3),          # packets_per_sec - LOW
                random.uniform(0.9, 1.0),      # tcp_ratio
                random.uniform(0.0, 0.1),      # udp_ratio
                0.0,                           # icmp_ratio
                random.randint(1, 3),          # unique_dest_ports - FEW
                1,                             # is_ics_port - YES
                0,                             # common_ports - NO
                1,                             # unique_sources
                random.randint(1, 4),          # unique_destinations - FEW
                random.uniform(0.5, 2),        # connection_rate - LOW
                random.randint(0, 2),          # failed_connections - FEW
                random.randint(1, 5),          # syn_count - LOW
                random.uniform(0.3, 0.5),      # payload_entropy
                random.uniform(0.0, 1.0),      # time_of_day
                random.choice([0, 1]),         # is_weekend
                0                              # time_since_last
            ]
    
    # Pattern 2: Reconnaissance → Impact (Aggressive)
    elif current_technique in RECONNAISSANCE_TECHNIQUES and next_technique in IMPACT_TECHNIQUES:
        if variant_type == 'aggressive':
            return [
                random.randint(750, 1250),     # packet_count - VERY HIGH (3s window)
                random.randint(100000, 200000),  # byte_count - VERY HIGH
                random.uniform(200, 400),      # packets_per_sec - VERY HIGH
                random.uniform(0.6, 0.85),     # tcp_ratio
                random.uniform(0.15, 0.35),    # udp_ratio
                random.uniform(0.0, 0.15),     # icmp_ratio
                random.randint(5, 15),         # unique_dest_ports
                random.choice([0, 1]),         # is_ics_port
                1,                             # common_ports
                random.randint(1, 8),          # unique_sources
                random.randint(10, 30),        # unique_destinations - MANY
                random.uniform(80, 200),       # connection_rate - VERY HIGH
                random.randint(15, 40),        # failed_connections - MANY
                random.randint(400, 750),      # syn_count - VERY HIGH
                random.uniform(0.05, 0.25),    # payload_entropy - low
                random.uniform(0.0, 1.0),      # time_of_day
                random.choice([0, 1]),         # is_weekend
                0                              # time_since_last
            ]
        else:  # normal
            return [
                random.randint(500, 1000),     # packet_count - HIGH (3s window)
                random.randint(50000, 150000), # byte_count - HIGH
                random.uniform(150, 300),      # packets_per_sec - HIGH
                random.uniform(0.7, 0.9),      # tcp_ratio
                random.uniform(0.1, 0.25),     # udp_ratio
                random.uniform(0.0, 0.1),      # icmp_ratio
                random.randint(3, 10),         # unique_dest_ports
                random.choice([0, 1]),         # is_ics_port
                1,                             # common_ports
                random.randint(1, 5),          # unique_sources
                random.randint(8, 20),         # unique_destinations - MANY
                random.uniform(50, 150),       # connection_rate - HIGH
                random.randint(10, 30),        # failed_connections - MANY
                random.randint(250, 500),      # syn_count - HIGH
                random.uniform(0.1, 0.3),      # payload_entropy - low
                random.uniform(0.0, 1.0),      # time_of_day
                random.choice([0, 1]),         # is_weekend
                0                              # time_since_last
            ]
    
    # Pattern 3: Reconnaissance → Reconnaissance (Continued exploration)
    elif current_technique in RECONNAISSANCE_TECHNIQUES and next_technique in RECONNAISSANCE_TECHNIQUES:
        return [
            random.randint(8, 15),         # packet_count - MODERATE (3s window)
            random.randint(2500, 5000),    # byte_count
            random.uniform(2.5, 5),        # packets_per_sec - MODERATE
            random.uniform(0.85, 0.95),    # tcp_ratio
            random.uniform(0.05, 0.15),    # udp_ratio
            random.uniform(0.0, 0.05),     # icmp_ratio
            random.randint(3, 8),          # unique_dest_ports - MANY
            1,                             # is_ics_port
            1,                             # common_ports
            1,                             # unique_sources
            random.randint(2, 8),          # unique_destinations - MODERATE
            random.uniform(2, 8),          # connection_rate - MODERATE
            random.randint(1, 4),          # failed_connections - MODERATE
            random.randint(3, 8),          # syn_count - MODERATE
            random.uniform(0.25, 0.45),    # payload_entropy
            random.uniform(0.0, 1.0),      # time_of_day
            random.choice([0, 1]),         # is_weekend
            0                              # time_since_last
        ]
    
    # Pattern 4: Lateral Movement → Execution (Program Download/Upload)
    elif current_technique in LATERAL_MOVEMENT_TECHNIQUES and next_technique in EXECUTION_TECHNIQUES:
        return [
            random.randint(2, 6),          # packet_count - LOW (3s window)
            random.randint(5000, 50000),   # byte_count - HIGH (downloading)
            random.uniform(0.5, 2),        # packets_per_sec - VERY LOW
            random.uniform(0.95, 1.0),     # tcp_ratio - all TCP
            random.uniform(0.0, 0.05),     # udp_ratio
            0.0,                           # icmp_ratio
            1,                             # unique_dest_ports - SINGLE
            1,                             # is_ics_port - YES
            0,                             # common_ports - NO
            1,                             # unique_sources
            1,                             # unique_destinations - SINGLE
            random.uniform(0.2, 1),        # connection_rate - VERY LOW
            0,                             # failed_connections - NO failures
            random.randint(1, 3),          # syn_count - VERY LOW
            random.uniform(0.6, 0.9),      # payload_entropy - HIGH (encrypted)
            random.uniform(0.0, 1.0),      # time_of_day
            random.choice([0, 1]),         # is_weekend
            0                              # time_since_last
        ]
    
    # Pattern 5: Execution → Execution (Modifying control logic)
    elif current_technique in EXECUTION_TECHNIQUES and next_technique in EXECUTION_TECHNIQUES:
        return [
            random.randint(1, 3),          # packet_count - VERY LOW (3s window)
            random.randint(50, 500),       # byte_count - LOW
            random.uniform(0.2, 1),        # packets_per_sec - VERY LOW
            random.uniform(0.95, 1.0),     # tcp_ratio
            random.uniform(0.0, 0.05),     # udp_ratio
            0.0,                           # icmp_ratio
            1,                             # unique_dest_ports - SINGLE
            1,                             # is_ics_port - YES
            0,                             # common_ports - NO
            1,                             # unique_sources
            1,                             # unique_destinations
            random.uniform(0.1, 0.5),      # connection_rate - VERY LOW
            0,                             # failed_connections
            random.randint(1, 2),          # syn_count - VERY LOW
            random.uniform(0.4, 0.7),      # payload_entropy
            random.uniform(0.0, 1.0),      # time_of_day
            random.choice([0, 1]),         # is_weekend
            0                              # time_since_last
        ]
    
    # Pattern 6: Execution → Impact (Causing damage)
    elif current_technique in EXECUTION_TECHNIQUES and next_technique in IMPACT_TECHNIQUES:
        return [
            random.randint(1, 5),          # packet_count - LOW (3s window)
            random.randint(250, 1500),     # byte_count - LOW
            random.uniform(0.3, 1.5),      # packets_per_sec - LOW
            random.uniform(0.9, 1.0),      # tcp_ratio
            random.uniform(0.0, 0.1),      # udp_ratio
            0.0,                           # icmp_ratio
            random.randint(1, 2),          # unique_dest_ports - FEW
            1,                             # is_ics_port - YES
            0,                             # common_ports - NO
            1,                             # unique_sources
            random.randint(1, 3),          # unique_destinations - FEW
            random.uniform(0.2, 1),        # connection_rate - LOW
            random.randint(0, 1),          # failed_connections
            random.randint(1, 3),          # syn_count - LOW
            random.uniform(0.3, 0.6),      # payload_entropy
            random.uniform(0.0, 1.0),      # time_of_day
            random.choice([0, 1]),         # is_weekend
            0                              # time_since_last
        ]
    
    # Pattern 7: Lateral Movement → Exfiltration
    elif current_technique in LATERAL_MOVEMENT_TECHNIQUES and next_technique in EXFILTRATION_TECHNIQUES:
        return [
            random.randint(5, 15),         # packet_count (3s window)
            random.randint(25000, 100000), # byte_count - HIGH (exfiltrating)
            random.uniform(2, 8),          # packets_per_sec
            random.uniform(0.9, 1.0),      # tcp_ratio
            random.uniform(0.0, 0.1),      # udp_ratio
            0.0,                           # icmp_ratio
            random.randint(1, 2),          # unique_dest_ports
            0,                             # is_ics_port - NO (external)
            1,                             # common_ports - YES (HTTPS)
            1,                             # unique_sources
            random.randint(1, 2),          # unique_destinations - external
            random.uniform(2, 8),          # connection_rate
            random.randint(0, 2),          # failed_connections
            random.randint(1, 5),          # syn_count
            random.uniform(0.7, 0.95),     # payload_entropy - HIGH (encrypted)
            random.uniform(0.0, 1.0),      # time_of_day
            random.choice([0, 1]),         # is_weekend
            0                              # time_since_last
        ]
    
    # Pattern 8: Any → Persistence (Maintaining access)
    elif next_technique in PERSISTENCE_TECHNIQUES:
        return [
            random.randint(1, 4),          # packet_count - LOW (3s window)
            random.randint(250, 1000),     # byte_count - LOW
            random.uniform(0.3, 1),        # packets_per_sec - LOW
            random.uniform(0.9, 1.0),      # tcp_ratio
            random.uniform(0.0, 0.1),      # udp_ratio
            0.0,                           # icmp_ratio
            1,                             # unique_dest_ports
            random.choice([0, 1]),         # is_ics_port
            1,                             # common_ports
            1,                             # unique_sources
            random.randint(1, 2),          # unique_destinations
            random.uniform(0.2, 0.8),      # connection_rate - LOW
            random.randint(0, 1),          # failed_connections
            random.randint(1, 3),          # syn_count
            random.uniform(0.5, 0.8),      # payload_entropy
            random.uniform(0.0, 1.0),      # time_of_day
            random.choice([0, 1]),         # is_weekend
            0                              # time_since_last
        ]
    
    # ========== DEFAULT PATTERN (based on next technique category) ==========
    
    # Default for Reconnaissance
    elif next_technique in RECONNAISSANCE_TECHNIQUES:
        return [
            random.randint(5, 13),         # packet_count (3s window)
            random.randint(1500, 4000),
            random.uniform(1.5, 4),
            random.uniform(0.85, 0.95),
            random.uniform(0.05, 0.15),
            random.uniform(0.0, 0.05),
            random.randint(2, 6),
            1,
            1,
            1,
            random.randint(1, 5),
            random.uniform(1, 5),
            random.randint(1, 3),
            random.randint(2, 6),
            random.uniform(0.25, 0.45),
            random.uniform(0.0, 1.0),
            random.choice([0, 1]),
            0
        ]
    
    # Default for Lateral Movement
    elif next_technique in LATERAL_MOVEMENT_TECHNIQUES:
        return [
            random.randint(2, 6),
            random.randint(750, 2500),
            random.uniform(0.7, 2),
            random.uniform(0.9, 1.0),
            random.uniform(0.0, 0.1),
            0.0,
            random.randint(1, 3),
            1,
            0,
            1,
            random.randint(1, 4),
            random.uniform(0.3, 1.5),
            random.randint(0, 2),
            random.randint(1, 4),
            random.uniform(0.3, 0.5),
            random.uniform(0.0, 1.0),
            random.choice([0, 1]),
            0
        ]
    
    # Default for Execution
    elif next_technique in EXECUTION_TECHNIQUES:
        return [
            random.randint(1, 5),
            random.randint(2500, 25000),
            random.uniform(0.3, 1.5),
            random.uniform(0.95, 1.0),
            random.uniform(0.0, 0.05),
            0.0,
            1,
            1,
            0,
            1,
            1,
            random.uniform(0.2, 0.8),
            0,
            random.randint(1, 3),
            random.uniform(0.5, 0.8),
            random.uniform(0.0, 1.0),
            random.choice([0, 1]),
            0
        ]
    
    # Default for Impact
    elif next_technique in IMPACT_TECHNIQUES:
        return [
            random.randint(250, 750),      # packet_count (3s window)
            random.randint(25000, 100000),
            random.uniform(80, 250),
            random.uniform(0.7, 0.9),
            random.uniform(0.1, 0.25),
            random.uniform(0.0, 0.1),
            random.randint(3, 10),
            random.choice([0, 1]),
            1,
            random.randint(1, 5),
            random.randint(5, 15),
            random.uniform(30, 100),
            random.randint(5, 20),
            random.randint(100, 400),
            random.uniform(0.1, 0.3),
            random.uniform(0.0, 1.0),
            random.choice([0, 1]),
            0
        ]
    
    # Default for Exfiltration
    elif next_technique in EXFILTRATION_TECHNIQUES:
        return [
            random.randint(4, 13),
            random.randint(15000, 75000),
            random.uniform(1.5, 5),
            random.uniform(0.9, 1.0),
            random.uniform(0.0, 0.1),
            0.0,
            random.randint(1, 2),
            0,
            1,
            1,
            random.randint(1, 2),
            random.uniform(1, 5),
            random.randint(0, 2),
            random.randint(1, 5),
            random.uniform(0.7, 0.95),
            random.uniform(0.0, 1.0),
            random.choice([0, 1]),
            0
        ]
    
    # Generic default
    else:
        return [
            random.randint(3, 10),
            random.randint(1000, 5000),
            random.uniform(1, 5),
            random.uniform(0.8, 0.95),
            random.uniform(0.05, 0.15),
            random.uniform(0.0, 0.05),
            random.randint(1, 4),
            random.choice([0, 1]),
            random.choice([0, 1]),
            1,
            random.randint(1, 5),
            random.uniform(1, 5),
            random.randint(0, 2),
            random.randint(1, 5),
            random.uniform(0.3, 0.6),
            random.uniform(0.0, 1.0),
            random.choice([0, 1]),
            0
        ]


def augment_sequences_with_network_features(sequences, num_variants=10):
    """
    Augment sequences by generating multiple variants with post-detection network features.
    
    Each variant represents different attacker behaviors (aggressive, stealthy, normal).
    
    Creates training pairs: (current_attack, next_attack, aggregated_network_features)
    
    Args:
        sequences: List of attack sequences
        num_variants: Number of variants per sequence
    
    Returns:
        training_pairs: List of (current_attack → next_attack + network_features) pairs
    """
    training_pairs = []
    variant_types = ['normal', 'stealthy', 'aggressive']
    
    for seq_data in sequences:
        sequence = seq_data['sequence']
        source = seq_data['source']
        seq_type = seq_data.get('type', 'unknown')
        
        # Generate multiple variants
        for variant in range(num_variants):
            # Choose variant type
            if variant < 6:
                variant_type = 'normal'
            elif variant < 8:
                variant_type = 'stealthy'
            else:
                variant_type = 'aggressive'
            
            # For each transition in the sequence, create a training pair
            for i in range(len(sequence) - 1):
                current_tech = sequence[i]
                next_tech = sequence[i + 1]
                
                # Generate post-detection features for this transition
                features = generate_post_detection_features(
                    current_tech, 
                    next_tech,
                    variant_type
                )
                
                # Create training pair
                training_pairs.append({
                    'source': f"{source}_variant_{variant}_{variant_type}",
                    'type': seq_type,
                    'current_attack': current_tech,
                    'next_attack': next_tech,
                    'network_features': features,
                    'variant_type': variant_type,
                    'sequence_position': i,
                    'sequence_length': len(sequence)
                })
    
    return training_pairs


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description='Generate realistic network features for attack sequences')
    parser.add_argument('--input', type=str, default='data/all_attack_sequences.json',
                       help='Input attack sequences file')
    parser.add_argument('--output', type=str, default='data/sequences_with_network_features.json',
                       help='Output file with network features')
    parser.add_argument('--variants', type=int, default=10,
                       help='Number of variants per sequence')
    args = parser.parse_args()
    
    print("=" * 70)
    print("REALISTIC NETWORK FEATURE GENERATION")
    print("=" * 70)
    
    # Setup paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    input_file = project_dir / args.input
    output_file = project_dir / args.output
    
    # Load attack sequences
    print(f"\n📥 Loading attack sequences from {input_file}...")
    with open(input_file) as f:
        sequences = json.load(f)
    print(f"✅ Loaded {len(sequences)} sequences")
    
    # Show sequence types
    type_counts = Counter(s.get('type', 'unknown') for s in sequences)
    print(f"\n📊 Sequence types:")
    for seq_type, count in type_counts.items():
        print(f"   {seq_type:20s}: {count:3d} sequences")
    
    # Generate network features
    print(f"\n🔨 Generating post-detection network features...")
    print(f"   Variants per sequence: {args.variants}")
    print(f"   Variant types: normal (60%), stealthy (20%), aggressive (20%)")
    
    training_pairs = augment_sequences_with_network_features(sequences, args.variants)
    print(f"✅ Generated {len(training_pairs)} training pairs")
    print(f"   Format: (current_attack → next_attack + aggregated_network_features)")
    
    # Save training pairs
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(training_pairs, f, indent=2)
    print(f"\n💾 Saved to {output_file}")
    
    # Print statistics
    print("\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70)
    print(f"Original sequences: {len(sequences)}")
    print(f"Training pairs: {len(training_pairs)}")
    print(f"Pairs per sequence: {len(training_pairs) / len(sequences):.1f}")
    
    # Analyze transitions
    transitions = [(p['current_attack'], p['next_attack']) for p in training_pairs]
    unique_transitions = len(set(transitions))
    print(f"Unique technique transitions: {unique_transitions}")
    
    # Show top transitions
    transition_counts = Counter(transitions)
    print(f"\n📈 Top 10 most common transitions:")
    for (curr, nxt), count in transition_counts.most_common(10):
        print(f"   {curr} → {nxt}: {count:3d} times")
    
    # Variant distribution
    variant_counts = Counter(p['variant_type'] for p in training_pairs)
    print(f"\n📊 Variant distribution:")
    for variant, count in variant_counts.items():
        print(f"   {variant:15s}: {count:5d} pairs ({100*count/len(training_pairs):.1f}%)")
    
    # Show example
    print("\n" + "=" * 70)
    print("EXAMPLE TRAINING PAIR")
    print("=" * 70)
    example = training_pairs[0]
    print(f"Source: {example['source']}")
    print(f"Type: {example['type']}")
    print(f"Variant: {example['variant_type']}")
    print(f"Position in sequence: {example['sequence_position'] + 1}/{example['sequence_length']}")
    print(f"\n🎯 Training Pair:")
    print(f"   Current Attack: {example['current_attack']}")
    print(f"   Next Attack:    {example['next_attack']}")
    print(f"   (Features represent traffic AFTER {example['current_attack']} detection)")
    
    features = example['network_features']
    feature_names = [
        'packet_count', 'byte_count', 'packets_per_sec',
        'tcp_ratio', 'udp_ratio', 'icmp_ratio',
        'unique_dest_ports', 'is_ics_port', 'common_ports',
        'unique_sources', 'unique_destinations',
        'connection_rate', 'failed_connections', 'syn_count',
        'payload_entropy', 'time_of_day', 'is_weekend', 'time_since_last'
    ]
    
    print("\n   Aggregated Network Features (60s post-detection):")
    for name, value in zip(feature_names, features):
        if isinstance(value, float):
            print(f"     {name:20s}: {value:.4f}")
        else:
            print(f"     {name:20s}: {value}")
    
    # Interpret behavior
    print(f"\n   Interpretation:")
    if features[0] > 5000:
        print(f"     - High volume ({int(features[0])} packets) → Aggressive behavior")
    elif features[0] < 100:
        print(f"     - Low volume ({int(features[0])} packets) → Stealthy behavior")
    else:
        print(f"     - Moderate volume ({int(features[0])} packets) → Normal behavior")
    
    if features[7] == 1:
        print(f"     - Targeting ICS ports → Industrial focus")
    
    if features[10] > 50:
        print(f"     - Many destinations ({int(features[10])}) → Spreading/DDoS")
    elif features[10] <= 5:
        print(f"     - Few destinations ({int(features[10])}) → Targeted attack")
    
    print("\n✅ Network feature generation complete!")
    print(f"\nNext step: python scripts/build_graph.py")


if __name__ == '__main__':
    main()
