import requests
import time
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from colorama import Fore, Style, init
from engine import SpoilageEngine

init(autoreset=True)

ESP_IP = "10.181.139.163"
ENDPOINT = f"http://{ESP_IP}/data"

# --- Pretty print (copied from replayer.py) ---
def pretty_print(res):
    ts = int(res["ts"]) if res["ts"] is not None else "?"
    product = res["product"]

    risk = res['risk_level'].upper()
    if risk == "OK":
        risk_str = Fore.GREEN + f"{risk} 🟢" + Style.RESET_ALL
    elif risk == "WARNING":
        risk_str = Fore.YELLOW + f"{risk} ⚠️" + Style.RESET_ALL
    else:
        risk_str = Fore.RED + f"{risk} 🔴" + Style.RESET_ALL

    print(f"\n{Fore.CYAN}[TS={ts}s] Product: {product}{Style.RESET_ALL}")
    print(Fore.CYAN + "-" * 45 + Style.RESET_ALL)
    print(f" Instant Spoilage : {res['instant_spoilage_pct']:.2f} %")
    print(f" Cumulative Index : {res['cumulative_spoilage_pct']:.2f} %")
    print(f" Risk Level       : {risk_str}")
    anomalies = res["anomalies"]
    print(f" Anomalies        : zscore={'Yes' if anomalies['zscore'] else 'No'} "
          f"| ewma={'Yes' if anomalies['ewma'] else 'No'}")
    print(Fore.MAGENTA + "\n Contributions:" + Style.RESET_ALL)
    for k, v in res["contributions"].items():
        print(f"   • {k.capitalize():<11}: {v*100:.1f} %")
    thr = res["adaptive_thresholds"]
    print(Fore.BLUE + "\n Thresholds:" + Style.RESET_ALL)
    print(f"   • Warning : {thr['warn']:.2f} %")
    print(f"   • Critical: {thr['crit']:.2f} %")
    print(Fore.LIGHTBLACK_EX + "\n Notes: " + " | ".join(res["notes"]) + Style.RESET_ALL)
    print(Fore.CYAN + "=" * 45 + Style.RESET_ALL)

# --- fetch JSON from ESP ---
def fetch_data():
    try:
        r = requests.get(ENDPOINT, timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"[WARN] fetch error: {e}")
    return None

# --- Live loop ---
def main():
    engine = SpoilageEngine(product="vaccine", window=30, mode="adaptive")

    xs, inst, cum, warn, crit = [], [], [], [], []
    step = 0
    all_results = []

    plt.ion()
    fig, ax = plt.subplots()

    print("📡 Streaming live data from ESP8266... Press Ctrl+C to stop")
    try:
        while True:
            data = fetch_data()
            if not data:
                time.sleep(3)
                continue

            step += 1
            # map ESP JSON to expected fields
            reading = {
                "ts": data.get("ts", step),
                "product": data.get("product", "vaccine"),
                "temp_inside_c": float(data.get("temp_inside_c", 0.0)),
                "temp_outside_c": float(data.get("temp_outside_c", 0.0)),
                "humidity_pct": float(data.get("humidity_pct", 0.0)),
                "door_open": int(data.get("door_open", 0)),
                "gas_ppm": float(data.get("gas_ppm", 0.0)),
            }

            out = engine.update(reading)
            pretty_print(out)

            xs.append(step)
            inst.append(out["instant_spoilage_pct"])
            cum.append(out["cumulative_spoilage_pct"])
            warn.append(out["adaptive_thresholds"]["warn"])
            crit.append(out["adaptive_thresholds"]["crit"])
            all_results.append(out)

            # update plot
            ax.cla()
            ax.plot(xs, inst, label="Instant")
            ax.plot(xs, cum, label="Cumulative")
            ax.plot(xs, warn, "--", label="Warn")
            ax.plot(xs, crit, "--", label="Crit")
            ax.set_ylim(0, 105)
            ax.set_xlabel("Step")
            ax.set_ylabel("Spoilage Index / %")
            ax.set_title("Spoilage Monitor (ESP Live)")
            ax.legend()
            plt.pause(0.1)

            # save every 10 samples
            if step % 10 == 0:
                df = pd.DataFrame(all_results)
                df.to_excel("esp_live_output.xlsx", index=False)
                df.to_csv("esp_live_output.csv", index=False)

            time.sleep(3)

    except KeyboardInterrupt:
        print("\n🛑 Stopping stream, saving logs...")
        if all_results:
            df = pd.DataFrame(all_results)
            df.to_excel("esp_live_output.xlsx", index=False)
            df.to_csv("esp_live_output.csv", index=False)
        print("✅ Logs saved.")

if __name__ == "__main__":
    main()
