# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np


def convert_with_scapy(pcap_path, output_path):
    try:
        from scapy.all import rdpcap, IP, TCP, UDP
    except ImportError:
        raise ImportError("Scapy is required: pip install scapy")

    packets = rdpcap(str(pcap_path))
    flows = {}

    for pkt in packets:
        if IP not in pkt:
            continue

        src = pkt[IP].src
        dst = pkt[IP].dst
        proto = pkt[IP].proto

        if TCP in pkt:
            sport = pkt[TCP].sport
            dport = pkt[TCP].dport
            flags = pkt[TCP].flags
            window = pkt[TCP].window
        elif UDP in pkt:
            sport = pkt[UDP].sport
            dport = pkt[UDP].dport
            flags = 0
            window = 0
        else:
            continue

        flow_key = (src, dst, sport, dport, proto)

        if flow_key not in flows:
            flows[flow_key] = {
                'src': src, 'dst': dst, 'sport': sport, 'dport': dport,
                'proto': proto, 'start_time': float(pkt.time),
                'end_time': float(pkt.time), 'fwd_packets': 0, 'bwd_packets': 0,
                'fwd_bytes': 0, 'bwd_bytes': 0, 'syn': 0, 'rst': 0, 'fin': 0,
                'psh': 0, 'pkt_sizes': [],
                'fwd_init_win': 0, 'bwd_init_win': 0,
            }

        flow = flows[flow_key]
        flow['end_time'] = float(pkt.time)
        flow['pkt_sizes'].append(len(pkt))

        if src == flow['src']:
            flow['fwd_packets'] += 1
            flow['fwd_bytes'] += len(pkt)
            if flow['fwd_init_win'] == 0:
                flow['fwd_init_win'] = window
        else:
            flow['bwd_packets'] += 1
            flow['bwd_bytes'] += len(pkt)
            if flow['bwd_init_win'] == 0:
                flow['bwd_init_win'] = window

        if TCP in pkt:
            if 'S' in str(flags):
                flow['syn'] += 1
            if 'R' in str(flags):
                flow['rst'] += 1
            if 'F' in str(flags):
                flow['fin'] += 1
            if 'P' in str(flags):
                flow['psh'] += 1

    rows = []
    for flow in flows.values():
        duration = flow['end_time'] - flow['start_time']
        iat = duration / max(flow['fwd_packets'] + flow['bwd_packets'], 1)

        rows.append({
            'Flow Duration': duration * 1e6,
            'Total Fwd Packet': flow['fwd_packets'],
            'Total Bwd packets': flow['bwd_packets'],
            'Total Length of Fwd Packet': flow['fwd_bytes'],
            'Total Length of Bwd Packet': flow['bwd_bytes'],
            'Flow IAT Mean': iat * 1e6,
            'Flow IAT Std': 0,
            'Fwd IAT Mean': iat * 1e6,
            'Bwd IAT Mean': iat * 1e6,
            'Down/Up Ratio': flow['bwd_bytes'] / max(flow['fwd_bytes'], 1),
            'Fwd Packets/s': flow['fwd_packets'] / max(duration, 1),
            'Bwd Packets/s': flow['bwd_packets'] / max(duration, 1),
            'SYN Flag Count': flow['syn'],
            'RST Flag Count': flow['rst'],
            'FIN Flag Count': flow['fin'],
            'PSH Flag Count': flow['psh'],
            'Packet Length Mean': np.mean(flow['pkt_sizes']) if flow['pkt_sizes'] else 0,
            'Packet Length Std': np.std(flow['pkt_sizes']) if flow['pkt_sizes'] else 0,
            'Average Packet Size': np.mean(flow['pkt_sizes']) if flow['pkt_sizes'] else 0,
            'Protocol': flow['proto'],
            'FWD Init Win Bytes': flow['fwd_init_win'],
            'Bwd Init Win Bytes': flow['bwd_init_win'],
            'Dst Port': flow['dport'],
            'Src Port': flow['sport'],
            'Label': 'BENIGN',
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return len(df)


def convert_with_cicflowmeter(pcap_path, output_path):
    try:
        from cicflowmeter.sniffer import Sniffer
    except ImportError:
        raise ImportError("cicflowmeter is required: pip install cicflowmeter")

    sniffer = Sniffer(
        input_file=str(pcap_path),
        output_file=str(output_path),
        output_format='csv',
    )
    sniffer.start()
    sniffer.join()
    return 0


def main():
    parser = argparse.ArgumentParser(description="WARDEN-T PCAP to CSV converter")
    parser.add_argument("input", help="Input PCAP file")
    parser.add_argument("output", help="Output CSV file")
    parser.add_argument("--backend", choices=["scapy", "cicflowmeter"], default="scapy")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: {args.input} not found")
        sys.exit(1)

    print(f"Converting {args.input} -> {args.output} (backend: {args.backend})")

    if args.backend == "scapy":
        n_flows = convert_with_scapy(args.input, args.output)
    else:
        n_flows = convert_with_cicflowmeter(args.input, args.output)

    print(f"Done. {n_flows} flows written to {args.output}")


if __name__ == "__main__":
    main()
