from django.db import models
# from typing_extensions import Required


class Form(models.Model):
    name = models.CharField(max_length=255, null=False)
    phone_no = models.BigIntegerField(null=False)
    city = models.CharField(max_length=255, null=False)
    state = models.CharField(max_length=255, null=False)
    crop_name = models.CharField(max_length=255, null=False)
    quantity = models.IntegerField(null=False)
    date = models.DateTimeField(auto_now_add=True)
    price = models.IntegerField(null=False)
