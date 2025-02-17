# My Flask App with Qdrant Integration

## Build and Run the Application

Follow the steps below to build and run the Flask app along with the Qdrant service.

### 1. Build the Flask Application

To build the Flask app Docker image, run the following command:

```bash
docker build -t my-flask-app .
docker pull qdrant/qdrant
docker run -d --name qdrant --network my_network -p 6333:63333 qdrant/qdrant
docker run -d --name my_app --network my_network -p 5000:5000 my-flask-app
