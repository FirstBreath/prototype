FROM ubuntu:20.04

# Mettre à jour le système et installer ffmpeg
RUN apt-get update && apt-get install -y ffmpeg

# Définissez les variables d'environnement avec des valeurs par défaut
ENV INPUT_FILE=/video.mp4
ENV RTSP_PORT=8554
ENV RTSP_ADDRESS=localhost
ENV RTSP_PATH=/test

WORKDIR /home

COPY ./video.mp4 /home/video.mp4

# Commande pour lancer ffmpeg en boucle
CMD ["sh", "-c", "ffmpeg -stream_loop -1 -re -i ${INPUT_FILE} -c copy -f rtsp rtsp://${RTSP_ADDRESS}:${RTSP_PORT}${RTSP_PATH}"]
