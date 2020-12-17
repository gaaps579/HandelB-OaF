from ..models import Song
from rest_framework import serializers


class AudiosSerializer(serializers.ModelSerializer):
    class Meta:
        model = Song
        fields = '__all__'
