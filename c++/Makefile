# Nome do executável
TARGET = a.out

# Compilador
CXX = g++
CXXFLAGS = -O2 -Wall -std=c++17 -fopenmp  # <-- adicionada flag -fopenmp

# Fontes
SRC = main.cpp cell.cpp cell_lattice.cpp obstacle.cpp config.cpp simulation.cpp

# Cabeçalhos
HEADERS = cell.h cell_lattice.h obstacle.h config.h utils.h simulation.cpp

# Objetos
OBJ = $(SRC:.cpp=.o)

# Regra principal
all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean
