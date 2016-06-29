## Distributed Tensorflow Tips

I've been doing some experimenting with [Distributed Tensorflow](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/distributed_runtime) recently and got it to work on a Google Compute Engine cluster. I just wanted to mention a few of the issues I ran across in case anyone else wants to try setting it up in the future:

1. I had to build tensorflow from source, following the instructions [here](https://www.tensorflow.org/versions/master/get_started/os_setup.html#create-the-pip-package-and-install) before I [build the distributed runtime](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/distributed_runtime).

2. You will need to open up port "2222" (or whichever port you're using) in the GCE firewall. I followed the instructions [here](https://cloud.google.com/compute/docs/networking#addingafirewall).

3. I ran into some confusion as to how to start the server on the different machines. As it [turns out](https://github.com/tensorflow/tensorflow/issues/1418), you need to run almost the exact same command on all the machines to start servers that communicate with each other. The only thing I changed was the job_name for different instances. 

So for instance on VM#1, you could run:

```
bazel-bin/tensorflow/core/distributed_runtime/rpc/grpc_tensorflow_server \
--cluster_spec='worker|192.168.170.193:2500,prs|192.168.170.226:2500' --job_name=worker --task_id=0 &'
```

And on VM#2 run:

```
bazel-bin/tensorflow/core/distributed_runtime/rpc/grpc_tensorflow_server \
--cluster_spec='worker|192.168.170.193:2500,prs|192.168.170.226:2500' --job_name=prs --task_id=0 &'
```