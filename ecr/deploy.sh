repository_name="${ECR_REPOSITORY_NAME-neuronx-inference}"

account=$(aws sts get-caller-identity --query Account --output text)

# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=$(aws configure get region)
region=${region:-us-west-2}

source_image_name=$(awk -F ' ' '/^FROM/ { gsub(/"/, "", $2); print $2 }' Dockerfile)

target_image_name="${account}.dkr.ecr.${region}.amazonaws.com/${repository_name}:latest"

# If the repository doesn't exist in ECR, create it.
aws ecr describe-repositories --repository-names "${repository_name}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${repository_name}" > /dev/null
fi

aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin ${source_image_name}

# Get the login command from ECR and execute it directly
aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin ${target_image_name}

# Build the Docker image locally with the image name and then push it to ECR
docker build -q -t ${repository_name} .
docker tag ${repository_name} ${target_image_name}

docker push ${target_image_name}