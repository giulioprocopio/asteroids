build:
	cmake -S . -B build
	cmake --build build

build-web:
	emcmake cmake -S . -B build-web
	cmake --build build-web

clean:
	rm -rf build build-web

run: build
	./build/asteroids

run-web: build-web
	python -m http.server -d build-web 8080

.PHONY: build build-web clean run run-web
