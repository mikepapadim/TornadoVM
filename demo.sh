# Temurin Example
sdk list java
sdk use java 21.0.2-tem
make
tornado -version && tornado --version
tornado --threadInfo -m tornado.examples/uk.ac.manchester.tornado.examples.VectorAddInt 256


# OpenJDK Example
sdk list java
sdk use java 21.ea.35-open
make
tornado -version && tornado --version
tornado --threadInfo -m tornado.examples/uk.ac.manchester.tornado.examples.VectorAddInt 256


