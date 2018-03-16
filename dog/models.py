from django.db import models

# Create your models here.


class user(models.Model):
    mail = models.CharField(max_length=20)
    password =  models.CharField(max_length=20)