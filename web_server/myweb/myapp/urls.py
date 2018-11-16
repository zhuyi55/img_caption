from django.urls import path
from django.conf.urls import url
from . import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    url('index', views.index, name='index'),
    url('upload', views.uploadImg),
    url('show', views.showImg),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
