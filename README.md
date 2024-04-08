The NVIDIA Triton Inference Server (formerly known as TensorRT Inference Server) is an open-source software solution developed by NVIDIA. It provides a cloud inference solution optimized for NVIDIA GPUs. Triton simplifies the deployment of AI models at scale in production.

In Summary :
    Triton Inference Server is designed to deploy a variety of AI models in production. It supports a wide range of deep learning and machine learning frameworks, including TensorFlow, PyTorch, ONNX Runtime, and many others. Its primary use cases are:
    - Serving multiple models from a single server instance.
    - Dynamic model loading and unloading without server restart.
    - Ensemble inference, allowing multiple models to be used together to achieve results.
    - Model versioning for A/B testing and rolling updates.

**HOW TO SERVE MULTIPLE MODELS FROM A SINGLE TRITON SERVER INSTANCE : A GUIDE**

Before you can use the Triton Docker image you must install Docker. 
If you plan on using a GPU for inference you must also install the NVIDIA Container Toolkit.

Prerequisites:

        Docker
        TritonClient
        `pip install tritonclient[all]`

Steps: 

Pull the image using the following command.
        `$ docker pull nvcr.io/nvidia/tritonserver:<yy.mm>-py3`
        Where <yy.mm> is the version of Triton that you want to pull.  For a complete list of all the variants and versions of the Triton Inference Server Container, visit the NGC Page.

For example: 
`docker pull nvcr.io/nvidia/tritonserver:23.09-py3`

Setup a folder structure like the below :

        # Example repository structure
        <model-repository>/
          <model-name>/
            [config.pbtxt]
            [<output-labels-file> ...]
            <version>/
              <model-definition-file>
            <version>/
              <model-definition-file>
            ...
          <model-name>/
            [config.pbtxt]
            [<output-labels-file> ...]
            <version>/
              <model-definition-file>
            <version>/
              <model-definition-file>
            ...
          ...


Folder Structure example : 

    model_repository/
    ├── text_detection
    │   ├── 1
    │   │   └── model.onnx
    │   ├── 2
    │   │   └── model.onnx
    │   └── config.pbtxt
    └── text_recognition
        ├── 1
        │   └── model.py
        └── config.pbtxt

Run the container using the following command

`docker run -d -p 8000:8000 -v {triton_repo_path}:/models {tag} /bin/bash`

where “triton_repo_path” should be absolute path to the “model repository” for above example it would be ‘../../model_repository’

Inside the container some libraries need to be installed by running the following commands
`apt-get update && apt-get install -y libgl1`
`pip3 install torch==1.13.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html torchvision==0.14.0+cu117 pillow opencv-python pandas`

Start the server using the below
`tritonserver --model-repository /models`

All the above can be run using the following python code

    container_id = subprocess.check_output(
                        f'docker run -d -p 8000:8000 -v {triton_repo_path}:/models {tag} /bin/bash -c "apt-get update && apt-get install -y libgl1 && pip3 install torch==1.13.0 -f https://download.pytorch.org/whl/torch_stable.html torchvision==0.14.0 pillow opencv-python pandas && tritonserver --model-repository /models"', 
                        shell=True).decode('utf-8').strip()

And run the inference using the below
`python client.py`


