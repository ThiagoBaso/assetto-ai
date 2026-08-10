import time
import pyvjoy

vj = pyvjoy.VJoyDevice(1)

print("Testando vJoy - movendo eixos...")
# Mexer volante (X) - centro -> esquerda -> direita -> centro
print("Volante para esquerda...")
vj.set_axis(pyvjoy.HID_USAGE_X, 0x2000)  # 25% do range esquerda
time.sleep(1)
print("Volante para direita...")
vj.set_axis(pyvjoy.HID_USAGE_X, 0x6000)  # 75% do range direita
time.sleep(5)
vj.set_axis(pyvjoy.HID_USAGE_X, 0x4000)  # centro

# Testar acelerador (Y) - valores 0x0000 = solto, 0x8000 = fundo
print("Acelerando...")
vj.set_axis(pyvjoy.HID_USAGE_Y, 0x6000)  # ~75%
time.sleep(5)
vj.set_axis(pyvjoy.HID_USAGE_Y, 0x0000)

# Testar freio (Z)
print("Freando...")
vj.set_axis(pyvjoy.HID_USAGE_Z, 0x6000)
time.sleep(5)
vj.set_axis(pyvjoy.HID_USAGE_Z, 0x0000)

vj.set_axis(pyvjoy.HID_USAGE_X, 0x0000)  # volante todo à esquerda
time.sleep(1)
vj.set_axis(pyvjoy.HID_USAGE_X, 0x8000)  # volante todo à direita
time.sleep(1)
vj.set_axis(pyvjoy.HID_USAGE_X, 0x4000)  # centro

vj.set_axis(pyvjoy.HID_USAGE_Y, 0x8000)  # acelerador máximo
time.sleep(1)
vj.set_axis(pyvjoy.HID_USAGE_Y, 0x0000)  # solto

print("Fim do teste.")