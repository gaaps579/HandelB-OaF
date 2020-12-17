from django.db import models
from .OaF import Transcribe


class Song(models.Model):
    audio = models.FileField(upload_to="musics/%Y/%m/%D/", default="Vacio")
    id_analisis = models.CharField(max_length=250, default="Vacio")
    song_title = models.CharField(max_length=250)
    uploaded = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return "Audio processed at {}".format(self.uploaded.strftime("%Y-%m-%d %H:%M"))

    def save(self, *args, **kwargs):
        try:
            # print(self.audio)
            super().save(*args, **kwargs)
            Transcribe(self.audio.path, self.id_analisis)
        except Exception as e:
            print("Process failed", e)
