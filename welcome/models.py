from django.db import models

class Product(models.Model):
    name=models.CharField(max_length=255)
    price=models.IntegerField()
    image_url=models.CharField(max_length=2083)
    desc=models.CharField(max_length=2083)
    color=models.CharField(max_length=255)
    size=models.CharField(max_length=255)
    company=models.CharField(max_length=255)
    tag=models.CharField(max_length=255)

class UserProduct(models.Model):
    name=models.CharField(max_length=255)
    pid=models.IntegerField()
