import tensorflow as tf


def main() -> None:
    print(f"tensorflow={tf.__version__}")
    print(f"built_with_cuda={tf.test.is_built_with_cuda()}")
    devices = tf.config.list_physical_devices()
    gpus = tf.config.list_physical_devices("GPU")
    print(f"devices={devices}")
    print(f"gpus={gpus}")
    if not gpus:
        raise SystemExit("TensorFlow cannot see a GPU.")

    with tf.device("/GPU:0"):
        a = tf.random.normal([1024, 1024])
        b = tf.random.normal([1024, 1024])
        c = tf.matmul(a, b)
    print(f"gpu_matmul_sum={float(tf.reduce_sum(c).numpy()):.4f}")


if __name__ == "__main__":
    main()
