import pyshark
import pandas as pd

# Define the input PCAP file and output CSV file
input_pcap = "/Users/dhruvanavinchander1/Downloads/SUEE1.pcap-anon.pcap"
output_csv = "output.csv"


print("Reading PCAP file...")
capture = pyshark.FileCapture(input_pcap)


packets_list = []


for packet in capture:
    try:
        packet_info = {
            "No.": packet.number,  # Packet number
            "Time": packet.sniff_time,  # Timestamp
            "Source": packet.ip.src if hasattr(packet, 'ip') else None,
            "Destination": packet.ip.dst if hasattr(packet, 'ip') else None,
            "Protocol": packet.highest_layer,  # Protocol type
            "Length": packet.length,  # Packet length
            "Info": str(packet)[:100]  # Packet summary (truncated for readability)
        }
        packets_list.append(packet_info)
    except Exception as e:
        print(f"Error processing packet: {e}")


capture.close()


print("Converting to DataFrame...")
df = pd.DataFrame(packets_list)


print(f"Saving to CSV: {output_csv}...")
df.to_csv(output_csv, index=False)

print("Conversion complete!")
