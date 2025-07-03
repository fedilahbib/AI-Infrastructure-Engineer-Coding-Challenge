FROM nvcr.io/nvidia/tritonserver:23.03-py3

# Copy the model repository to the container
COPY ../models /models

# Run Triton with the correct model repo path
CMD ["tritonserver", "--model-repository=/models"]
