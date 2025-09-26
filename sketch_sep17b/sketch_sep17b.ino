#include <Wire.h>
#include <Adafruit_MMC56x3.h>

#define TCAADDR 0x70  // Default I2C address for TCA9548A

void tcaselect(uint8_t i) {
  if (i > 7) return;
  Wire.beginTransmission(TCAADDR);
  Wire.write(1 << i);
  Wire.endTransmission();
}

// Create an array for 8 sensor objects
Adafruit_MMC5603 mmc[8] = {
  Adafruit_MMC5603(0),
  Adafruit_MMC5603(1),
  Adafruit_MMC5603(2),
  Adafruit_MMC5603(3),
  Adafruit_MMC5603(4),
  Adafruit_MMC5603(5),
  Adafruit_MMC5603(6),
  Adafruit_MMC5603(7)
};

void setup() {
  Serial.begin(115200);
  Wire.begin();
    // Speed up I2C
  for (int i = 0; i < 8; i++) {
    tcaselect(i);
    mmc[i].begin(MMC56X3_DEFAULT_ADDRESS, &Wire);
  }
}

void loop() {
  // Print sensor readings in a single line: Y X Z Y X Z ...
  for (int i = 0; i < 8; i++) {
    tcaselect(i);
    sensors_event_t event;
    if (mmc[i].getEvent(&event)) {
      Serial.print(event.magnetic.y); Serial.print(" ");
      Serial.print(event.magnetic.x); Serial.print(" ");
      Serial.print(event.magnetic.z); Serial.print(" ");
    } else {
      // If the sensor read fails, print zeros as placeholders
      Serial.print("0 0 0 ");
    }
    
  }
  Serial.println(); // End the line for this frame
  delay(20);
  // Optionally add a small delay here if needed, e.g. delay(1);
}
