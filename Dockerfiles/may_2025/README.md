docker buildx create --name my-builder --use
docker buildx ls
docker login -u mshaikh

# For multiplatform build and pushing to the dockerhub 
docker buildx build --platform   linux/amd64,linux/arm64 -t ds-torch -f Dockerfile.torch --push .
# For arm64 build and adding local image for testing and debugging
docker buildx build --platform   linux/arm64 -t ds-torch -f Dockerfile.torch --load .

# Running an interactive session with GPU support
docker run --rm -ti --gpus all ds-torch:latest


