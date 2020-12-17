from .views import SongViewSet
from rest_framework import routers
from django.urls import path, include

app_name = 'api-audios'

router = routers.DefaultRouter()
router.register(r'songs', SongViewSet)

urlpatterns = [path('', include(router.urls))]
