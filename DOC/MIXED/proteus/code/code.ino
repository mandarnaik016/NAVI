const int pingPin = 7; // Trigger Pin of Ultrasonic Sensor
const int echoPin = 6; // Echo Pin of Ultrasonic Sensor


void setup()
{
  pinMode(13, OUTPUT);
  Serial.begin(9600); // Starting Serial Terminal
}

void loop()
{
  long duration, inches, cm;
  pinMode(pingPin, OUTPUT);
  digitalWrite(pingPin, LOW);
  delayMicroseconds(2);
  digitalWrite(pingPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(pingPin, LOW);

  pinMode(echoPin, INPUT);
  duration = pulseIn(echoPin, HIGH);
  cm = microsecondsToCentimeters(duration);

  Serial.print(cm);
  Serial.print("cm");
  Serial.println();

  if( cm < 100 ){
    digitalWrite(13, HIGH);
  }
  else{
    digitalWrite(13, LOW);
  }

  delay(100);
}

long microsecondsToCentimeters(long microseconds)
{
  return microseconds / 29 / 2;
}






