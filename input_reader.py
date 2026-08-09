import pygame


class InputReader:

    def __init__(self):
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            raise RuntimeError("Nenhum controle encontrado.")

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

    def read(self):
        pygame.event.pump()

        steering = self.joystick.get_axis(0)
        brake = self.joystick.get_axis(4)
        throttle = self.joystick.get_axis(5)

        # Converte -1..+1 para 0..1
        brake = (brake + 1) / 2
        throttle = (throttle + 1) / 2

        return {
            "steering_input": steering,
            "brake_input": brake,
            "throttle_input": throttle,
        }

    def close(self):
        pygame.quit()