from django.urls import path
from .views import *

urlpatterns = [

    path('', index),

    path('upload/', upload_pdf),

    path('chat/', chat),

]