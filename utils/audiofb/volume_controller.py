import serial

# Arduino code:
# # include <Wire.h>
#
# String message;
#
# void setup(){
#     Serial.begin(115200);
#     Wire.begin();
# }
#
# void loop(){
#     while (Serial.available()) {
#         char incomingChar = Serial.read();
#         if (incomingChar >= '0' & & incomingChar <= '9'){
#             message += incomingChar;
#         } else if (incomingChar == '\n') {
#             Wire.beginTransmission(0x44);
#             Wire.write(message.toInt() / 1000);
#             Wire.write(message.toInt() % 1000);
#             Wire.endTransmission();
#             message = "";
#         }
#     }
# }

def level_to_codes(level):
    level = min(max(0, level), 100)
    level = 100 - level
    bd = int((level/100*79))
    return 224+bd//10, 208+bd%10

def codes_to_bytes(x):
    return '{}\n'.format(x[0]*1000+x[1]).encode()

class VolumeController:
    def __init__(self):
        self.ser = serial.Serial(port='COM4', baudrate=115200)
        self.ser.write(b'216\n')

    def close(self):
        self.ser.close()

    def set_volume(self, volume):
        self.ser.write(codes_to_bytes(level_to_codes(volume)))


if __name__ == '__main__':
    vc = VolumeController()
    vc.set_volume(100)
