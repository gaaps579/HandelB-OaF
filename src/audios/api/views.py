from rest_framework import viewsets
from .serializers import AudiosSerializer
from ..models import Song


class SongViewSet(viewsets.ModelViewSet):
    queryset = Song.objects.all().order_by('-uploaded')
    serializer_class = AudiosSerializer