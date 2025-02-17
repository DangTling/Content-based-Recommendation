To build, please run:
  docker build -t my-flask-app . 
  docker pull qdrant/qdrant
  docker run -d --name qdrant --network my_network -p 6333:63333 qdrant/qdrant
  docker run -d --name my_app --network my_network -p 5000:5000 my-flask-app
