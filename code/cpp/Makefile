CXX = g++
CXXFLAGS = -std=c++17 -O3 -I./include
TARGET = simple_benchmark
SOURCES = simple_benchmark.cpp src/matrix_mul.cpp

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: clean run