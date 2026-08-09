import csv
import time
from pathlib import Path

from telemetry import Telemetry
from input_reader import InputReader


class TelemetryRecorder:

    def __init__(self):
        self.telemetry = Telemetry()
        self.input = InputReader()

    def record(self, duration=60):
        Path("data/sessions").mkdir(
            parents=True,
            exist_ok=True
        )

        filename = (
            f"data/sessions/"
            f"session_{int(time.time())}.csv"
        )

        print(f"Gravando: {filename}")
        print(f"Duração: {duration}s")
        print("Dirija normalmente.")
        print("Ctrl+C para parar.\n")

        start_time = time.perf_counter()

        first_state = self.telemetry.read()
        first_input = self.input.read()

        fieldnames = [
            "timestamp",
            *first_state.keys(),
            *first_input.keys()
        ]

        with open(
            filename,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            writer.writeheader()

            try:
                while True:
                    elapsed = time.perf_counter() - start_time

                    if elapsed >= duration:
                        break

                    state = self.telemetry.read()
                    inputs = self.input.read()

                    writer.writerow({
                        "timestamp": elapsed,
                        **state,
                        **inputs
                    })

                    time.sleep(0.01)

            except KeyboardInterrupt:
                print("\nGravação interrompida.")

            finally:
                self.input.close()

        print(f"\nDataset salvo em:")
        print(filename)