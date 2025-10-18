export LD_LIBRARY_PATH=/home/farhan-akhtar/libtorch-cuda128/libtorch/lib:/usr/local/cuda/lib64
export LD_PRELOAD=/home/farhan-akhtar/libtorch-cuda128/libtorch/lib/libc10_cuda.so:/home/farhan-akhtar/libtorch-cuda128/libtorch/lib/libtorch_cuda.so
RUST_BACKTRACE=1 ./target/release/EfficientNetTrain "$@"
chmod +x run_cuda.sh