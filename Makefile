build:
	cmake -S . -B build
	cmake --build build

clean:
	rm -rf build

run: build
	./build/asteroids

.PHONY: build clean run
