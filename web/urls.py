from django.urls import path
from .views import vista_prediccion

urlpatterns = [
    path("", vista_prediccion, name="prediccion"),
]