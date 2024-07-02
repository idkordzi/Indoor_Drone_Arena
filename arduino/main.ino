// initialize digital pin LED_BUILTIN as an output.
void setup() {
  pinMode(2, OUTPUT);
  pinMode(3, OUTPUT);
  pinMode(4, OUTPUT);
  pinMode(5, OUTPUT);
}

int ms_delay = 20

// the loop function executes the main program in infinite loop
void loop() {
  digitalWrite(2, HIGH);  // set rising edge
  digitalWrite(3, HIGH);
  digitalWrite(4, HIGH);
  digitalWrite(5, HIGH);
  delay(ms_delay);       // wait
  digitalWrite(2, LOW);  // set falling edge
  digitalWrite(3, LOW);
  digitalWrite(4, LOW);
  digitalWrite(5, LOW);
  delay(ms_delay);      // wait
}