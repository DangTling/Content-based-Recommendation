# My Flask App with Qdrant Integration

## Build and Run the Application

Follow the steps below to build and run the Flask app along with the Qdrant service.

### 1. Build the Flask Application

To build the Flask app Docker image, run the following command:

```bash
docker build -t my-flask-app .
```
### 2. Pull the Qdrant Docker Image

```bash
docker pull qdrant/qdrant
```
### 3. Create a Docker Network

Create a custom Docker network where both containers will be connected:

```bash
docker network create my_network
```
### 4. Run the Qdrant Container

Now, run the Qdrant container in detached mode. The container will be connected to a custom Docker network, and port 6333 inside the container will be mapped to port 6333 on the host machine:

```bash
docker run -d --name qdrant --network my_network -p 6333:63333 qdrant/qdrant
```
### 5. Run the Flask Application Container

Finally, run the Flask application container, mapping port 5000 inside the container to port 5000 on your host machine:

```bash
docker run -d --name my_app --network my_network -p 5000:5000 my-flask-app
```
