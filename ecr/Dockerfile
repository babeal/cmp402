FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference-neuronx:1.13.1-neuronx-py310-sdk2.14.1-ubuntu20.04

RUN pip install --upgrade \ 
    torchserve==0.9.0 \
    neuronx-cc==2.11.0.34 \
    sentencepiece==0.1.99 \
    torch-neuronx==1.13.1.1.12.0 \
    transformers==4.34.1 \
    transformers-neuronx==0.8.268 \
    torchvision \
    --extra-index-url=https://pip.repos.neuron.amazonaws.com
