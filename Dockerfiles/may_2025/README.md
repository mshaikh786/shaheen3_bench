docker buildx create --name my-builder --use
docker buildx ls
docker login -u mshaikh
docker buildx build --platform   linux/amd64,linux/arm64 -t ds-torch -f Dockerfile.torch --push .
