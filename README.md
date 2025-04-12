Just run

`docker-compose up --build`


to restart

`docker-compose down && docker-compose up --build -d`



to see logs of frontend

`docker-compose logs frontend`


to restart just the frontend
`docker-compose restart frontend`


you need to serve ollama yourself, after having pulled relevant models

`ollama serve`