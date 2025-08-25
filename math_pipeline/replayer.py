# replayer.py
# Usage:
#   python replayer.py readings.csv adaptive        -> batch mode (CSV)
#   python replayer.py --live adaptive              -> realtime simulation (1s/step)

import csv
import sys
import time
import pandas as pd   # for Excel export
try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

from colorama import Fore, Style, init
from engine import SpoilageEngine

init(autoreset=True)  # enable colored output on all platforms


def pretty_print(res):
    """Nicely formatted console output."""
    ts = int(res["ts"]) if res["ts"] is not None else "?"
    product = res["product"]

    # Risk level with color + emoji
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


def run_reading(engine, reading, xs, inst, cum, warn, crit, step, all_results):
    """Process one reading, print + update plot, store result."""
    out = engine.update(reading)
    pretty_print(out)

    xs.append(step)
    inst.append(out["instant_spoilage_pct"])
    cum.append(out["cumulative_spoilage_pct"])
    warn.append(out["adaptive_thresholds"]["warn"])
    crit.append(out["adaptive_thresholds"]["crit"])
    all_results.append(out)

    if HAVE_MPL:
        plt.clf()
        plt.plot(xs, inst, label="Instant")
        plt.plot(xs, cum, label="Cumulative")
        plt.plot(xs, warn, "--", label="Warn")
        plt.plot(xs, crit, "--", label="Crit")
        plt.ylim(0, 105)
        plt.xlabel("Step")
        plt.ylabel("Spoilage Index / %")
        plt.title("Spoilage Replayer (Live)")
        plt.legend()
        plt.pause(0.1)


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python replayer.py readings.csv [adaptive|simple]   # batch mode")
        print("  python replayer.py --live [adaptive|simple]         # realtime simulation")
        sys.exit(1)

    live_mode = (sys.argv[1] == "--live")
    mode = sys.argv[2] if len(sys.argv) > 2 else "adaptive"

    engine = SpoilageEngine(product="vaccine", window=30, mode=mode)

    xs, inst, cum, warn, crit = [], [], [], [], []
    step = 0
    all_results = []  # store every output dict

    if live_mode:
        # Simulated live streaming
        with open("readings.csv", "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                step += 1
                reading = {
                    "ts": step,  # simulate timestamp
                    "product": row.get("product", "vaccine"),
                    "temp_inside_c": float(row["temp_inside_c"]),
                    "temp_outside_c": float(row["temp_outside_c"]),
                    "humidity_pct": float(row["humidity_pct"]),
                    "door_open": int(row["door_open"]),
                    "gas_ppm": float(row.get("gas_ppm", 0.0)),
                }
                run_reading(engine, reading, xs, inst, cum, warn, crit, step, all_results)
                time.sleep(1)  # 1s delay to mimic realtime sensor
    else:
        # Batch mode from CSV
        path = sys.argv[1]
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                step += 1
                reading = {
                    "ts": float(row["ts"]) if row.get("ts") else None,
                    "product": row.get("product", "vaccine"),
                    "temp_inside_c": float(row["temp_inside_c"]),
                    "temp_outside_c": float(row["temp_outside_c"]),
                    "humidity_pct": float(row["humidity_pct"]),
                    "door_open": int(row["door_open"]),
                    "gas_ppm": float(row.get("gas_ppm", 0.0)),
                }
                run_reading(engine, reading, xs, inst, cum, warn, crit, step, all_results)

        if HAVE_MPL:
            plt.show()
        else:
            print("\n(matplotlib not available – plotted data skipped)")

    # === EXPORT RESULTS ===
    if all_results:
        # Excel
        df = pd.DataFrame(all_results)
        df.to_excel("replayer_output.xlsx", index=False)

        # Text log
        with open("replayer_output.txt", "w") as f:
            for r in all_results:
                f.write(str(r) + "\n")

        print(Fore.GREEN + "\nResults exported to: replayer_output.xlsx and replayer_output.txt" + Style.RESET_ALL)


if __name__ == "__main__":
    main()
