import cv2
import yt_dlp

# URL du live YouTube
url = "https://www.youtube.com/live/qam4ytA2uGo?si=bUGqn2c6yn1EwxHf"

# Utilisation de yt-dlp pour obtenir le meilleur flux vidéo
ydl_opts = {
    'format': 'best',
    'quiet': True,
    'no_warnings': True,
    'force_generic_extractor': True
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info_dict = ydl.extract_info(url, download=False)
    video_url = info_dict.get('url', None)

# Capture du flux vidéo
cap = cv2.VideoCapture(video_url)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Affichage de la vidéo
    cv2.imshow('Live YouTube Stream', frame)
    
    # Quitter la fenêtre avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()