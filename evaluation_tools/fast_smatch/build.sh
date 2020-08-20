python3 setup.py build

cp build/*/_smatch.so .

#clean up by removing build files
rm -rf build/
