# generate_sample_csv.py
import csv
import random

def generate_csv(path="readings.csv", n_steps=40):
    """
    Generate a sample readings.csv with realistic variations:
      - Steps 0-9: stable cold room
      - Steps 10-19: door open events
      - Steps 20-29: outside temp rises (heat wave)
      - Steps 30-39: back to stable
    """
    fieldnames = ["ts","product","temp_inside_c","temp_outside_c","humidity_pct","door_open","gas_ppm"]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        ts = 0
        for step in range(n_steps):
            if step < 10:
                # stable
                ti = 5 + random.uniform(-0.3,0.3)
                to = 28 + random.uniform(-1,1)
                h = 60 + random.uniform(-2,2)
                d = 0
                g = 400 + random.uniform(-50,50)
            elif step < 20:
                # door open
                ti = 7 + random.uniform(-0.5,0.5)
                to = 30 + random.uniform(-2,2)
                h = 70 + random.uniform(-3,3)
                d = random.choice([0,1])  # door intermittently open
                g = 600 + random.uniform(-80,80)
            elif step < 30:
                # hot outside
                ti = 8 + random.uniform(-0.5,0.5)
                to = 38 + random.uniform(-2,2)
                h = 65 + random.uniform(-3,3)
                d = 0
                g = 700 + random.uniform(-100,100)
            else:
                # return to stable
                ti = 5 + random.uniform(-0.3,0.3)
                to = 29 + random.uniform(-1,1)
                h = 60 + random.uniform(-2,2)
                d = 0
                g = 450 + random.uniform(-50,50)

            writer.writerow({
                "ts": ts,
                "product": "vaccine",
                "temp_inside_c": round(ti,2),
                "temp_outside_c": round(to,2),
                "humidity_pct": round(h,1),
                "door_open": d,
                "gas_ppm": round(g,1)
            })
            ts += 60  # 1 minute step

    print(f"Sample CSV written to {path}")

if __name__ == "__main__":
    generate_csv()
