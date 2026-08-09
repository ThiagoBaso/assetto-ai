from lib.sim_info import info


class Telemetry:

    def read(self):
        p = info.physics
        g = info.graphics

        return {
            "speed": float(p.speedKmh),
            "rpm": int(p.rpms),
            "gear": int(p.gear),

            "steer_angle": float(p.steerAngle),
            "gas": float(p.gas),
            "brake": float(p.brake),
            "clutch": float(p.clutch),

            "acc_g_x": float(p.accG[0]),
            "acc_g_y": float(p.accG[1]),
            "acc_g_z": float(p.accG[2]),

            "local_vel_x": float(p.localVelocity[0]),
            "local_vel_y": float(p.localVelocity[1]),
            "local_vel_z": float(p.localVelocity[2]),

            "angular_vel_x": float(p.localAngularVel[0]),
            "angular_vel_y": float(p.localAngularVel[1]),
            "angular_vel_z": float(p.localAngularVel[2]),

            "slip_fl": float(p.wheelSlip[0]),
            "slip_fr": float(p.wheelSlip[1]),
            "slip_rl": float(p.wheelSlip[2]),
            "slip_rr": float(p.wheelSlip[3]),

            "tyre_temp_fl": float(p.tyreCoreTemperature[0]),
            "tyre_temp_fr": float(p.tyreCoreTemperature[1]),
            "tyre_temp_rl": float(p.tyreCoreTemperature[2]),
            "tyre_temp_rr": float(p.tyreCoreTemperature[3]),

            "pos_x": float(g.carCoordinates[0]),
            "pos_y": float(g.carCoordinates[1]),
            "pos_z": float(g.carCoordinates[2]),

            "lap_progress": float(g.normalizedCarPosition),

            "lap": int(g.completedLaps),
            "sector": int(g.currentSectorIndex),

            "surface_grip": float(g.surfaceGrip),

            "tyres_out": int(p.numberOfTyresOut),
        }