from django.contrib import admin
from django.urls import path
from predictor import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'), # Yeh line humari app ko homepage banati hai
]