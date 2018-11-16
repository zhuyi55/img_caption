from django.db import models

class IMG(models.Model):
    img = models.ImageField(upload_to='img')
    name = models.CharField(max_length=20)

# Create your models here.
