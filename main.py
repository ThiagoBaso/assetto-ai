import sys
import time
import time
import os
from lib.sim_info import info
from telemetry import Telemetry

MODE = 2
sys.path.append("lib")

if MODE == 0:
    while True:
        p = info.physics
        g = info.graphics

        os.system('cls' if os.name == 'nt' else 'clear')

        print("=== FÍSICA ===")
        print(f"Velocidade: {p.speedKmh:.1f} km/h")
        print(f"RPM: {p.rpms:.0f}  Marcha: {p.gear}")
        print(f"Volante: {p.steerAngle:.1f}°")
        print(f"Acelerador: {p.gas:.3f}  Freio: {p.brake:.3f}  Embreagem: {p.clutch:.3f}")
        print(f"Aceleração G: X={p.accG[0]:.2f}  Y={p.accG[1]:.2f}  Z={p.accG[2]:.2f}")
        print(f"Slip (rodas): {p.wheelSlip[0]:.2f} {p.wheelSlip[1]:.2f} {p.wheelSlip[2]:.2f} {p.wheelSlip[3]:.2f}")
        print(f"Vel. Angular local: {p.localAngularVel[0]:.3f} {p.localAngularVel[1]:.3f} {p.localAngularVel[2]:.3f}")
        print(f"Temp. núcleo pneus: {p.tyreCoreTemperature[0]:.0f} {p.tyreCoreTemperature[1]:.0f} {p.tyreCoreTemperature[2]:.0f} {p.tyreCoreTemperature[3]:.0f} °C")
        print(f"Desgaste pneus: {p.tyreWear[0]:.2f} {p.tyreWear[1]:.2f} {p.tyreWear[2]:.2f} {p.tyreWear[3]:.2f}")
        print(f"Curso suspensão: {p.suspensionTravel[0]:.3f} {p.suspensionTravel[1]:.3f} {p.suspensionTravel[2]:.3f} {p.suspensionTravel[3]:.3f}")
        print(f"Pneus fora: {p.numberOfTyresOut}  Aderência superfície: {g.surfaceGrip:.2f}")  # 1.0 = asfalto, menos = grama/areia

        print("\n=== POSIÇÃO E CORRIDA ===")
        print(f"Progresso na volta: {g.normalizedCarPosition:.3f}")
        print(f"Setor: {g.currentSectorIndex}  Voltas compl.: {g.completedLaps}/{g.numberOfLaps}")
        print(f"Posição na corrida: {g.position}")
        print(f"Tempo atual: {g.iCurrentTime/1000:.3f}s  Última volta: {g.iLastTime/1000:.3f}s  Melhor volta: {g.iBestTime/1000:.3f}s")
        print(f"Bandeira: {g.flag}  Status: {g.status}")
        print(f"Coordenadas: X={g.carCoordinates[0]:.2f} Y={g.carCoordinates[1]:.2f} Z={g.carCoordinates[2]:.2f}")

        print("\nPressione Ctrl+C para sair.")
        time.sleep(0.1)
elif MODE == 1:

    telemetry = Telemetry()

    try:

        while True:

            state = telemetry.read()

            print(
                f"Velocidade: {state['speed']:.1f} km/h | "
                f"RPM: {state['rpm']} | "
                f"Marcha: {state['gear']} | "
                f"Volante: {state['steer_angle']:.2f} | "
                f"Gas: {state['gas']:.2f} | "
                f"Freio: {state['brake']:.2f}"
            )

            time.sleep(0.1)

    except KeyboardInterrupt:

        print("\nEncerrando...")

    finally:

        info.close()

elif MODE == 2:
    from recorder import TelemetryRecorder

    recorder = TelemetryRecorder()

    recorder.record(duration=1200)

elif MODE == 3:
    from input_reader import InputReader


    controller = InputReader()

    try:
        while True:
            inputs = controller.read()

            print(
                f"\r"
                f"Steer: {inputs['steering_input']:+.3f} | "
                f"Brake: {inputs['brake_input']:.3f} | "
                f"Throttle: {inputs['throttle_input']:.3f}",
                end=""
            )

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nEncerrando...")

    finally:
        controller.close()